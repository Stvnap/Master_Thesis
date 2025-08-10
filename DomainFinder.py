import math
import os

import h5py
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ESM_Embeddings_HP_search import (
    ESMDataset,
    load_best_model,
    objective,
)

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
if not dist.is_initialized():
    dist.init_process_group("nccl")
    if dist.get_rank() == 0:
        print("Initializing process group for DDP")
RANK = dist.get_rank()
print(f"Start running basic DDP example on rank {RANK}.")
# create model and move it to GPU with id rank
DEVICE_ID = RANK % torch.cuda.device_count()


# -------------------------
# 1. Global settings
# -------------------------

CSV_PATH = "./Dataframes/v3/FoundEntriesSwissProteins.csv"
CATEGORY_COL = "Pfam_id"
SEQUENCE_COL = "Sequence"
CACHE_PATH = "./pickle/FoundEntriesSwissProteins_domains.pkl"
PROJECT_NAME = "Optuna_uncut_t33_domains_boundary"

ESM_MODEL = "esm2_t33_650M_UR50D"

NUM_CLASSES = (
    2  # Number of classes for domain detection, set to 1 for binary classification
)
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2

EMB_BATCH = 1
NUM_WORKERS_EMB = 32
# print(f"Using {NUM_WORKERS_EMB} workers for embedding generation")
BATCH_SIZE = 20
LR = 1e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 50
STUDY_N_TRIALS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}")


# -------------------------
# 2. Transformer Model
# -------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,  # 1280 for ESM2-t33
        hidden_dims,
        num_classes,
        num_heads,
        dropout,
        activation,
        kernel_init,
        num_layers,
    ):
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.kernel_init = kernel_init
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers

        # Ensure input_dim is divisible by num_heads for attention
        assert input_dim % num_heads == 0, (
            f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Optional: Project ESM embeddings to different dimension
        # self.input_projection = nn.Linear(input_dim, hidden_dims)
        # working_dim = hidden_dims
        working_dim = input_dim  # Use ESM embeddings directly

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=working_dim,
            nhead=num_heads,
            dim_feedforward=working_dim * 4,  # Standard: 4x model dimension
            dropout=dropout,
            activation=activation,
            batch_first=True,  # Much easier to work withs
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Per-position domain boundary classifier
        self.classifier = nn.Sequential(
            nn.Linear(working_dim, working_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(working_dim // 2, num_classes),
        )

    def generate_positional_encoding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * -(math.log(10000.0) / d_model)
        )

        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # Add batch dimension [1, seq_len, d_model]

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # print(f"Input shape: {x.shape}")  # Debugging line

        batch_size, seq_len, input_dim = x.size()

        # Optional input projection
        # x = self.input_projection(x)

        # Add positional encoding
        pe = self.generate_positional_encoding(seq_len, input_dim, x.device)
        x = x + pe  # Broadcasting: [batch, seq, dim] + [1, seq, dim]

        # Apply dropout
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Transformer processing (batch_first=True, so no transpose needed)
        x = self.encoder(x)  # [batch_size, seq_len, input_dim]

        # Per-position classification for domain boundaries
        logits = self.classifier(x)  # [batch_size, seq_len, num_classes]

        return logits


# -------------------------
# 3. TRAINER
# -------------------------

# -------------------------
# 4. MAIN
# -------------------------


def main(Final_training=False):
    # os.makedirs("pickle", exist_ok=True)
    os.makedirs(f"logs/{PROJECT_NAME}", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    print("Directories created.")

    if not os.path.exists("./temp/embeddings_domain.h5"):
        ESMDataset(
            FSDP_used=False,
            domain_boundary_detection=True,
            num_classes=NUM_CLASSES,
            esm_model=ESM_MODEL,
            csv_path=CSV_PATH,
            category_col=CATEGORY_COL,
            sequence_col=SEQUENCE_COL,
        )
        print("Using preembedded ESM data from scratch")

    class DomainBoundaryDataset(Dataset):
        """Custom dataset for domain boundary detection with ESM embeddings that loads data from an H5 file."""

        def __init__(self, h5_file):
            self.h5_file = h5_file

            # Determine the total length by inspecting the H5 file
            with h5py.File(self.h5_file, "r") as f:
                self.embedding_keys = sorted(
                    [k for k in f.keys() if k.startswith("batch_")]
                )
                self.chunk_sizes = [f[key].shape[0] for key in self.embedding_keys]
                self.cumulative_sizes = np.cumsum(self.chunk_sizes)
                self.length = (
                    self.cumulative_sizes[-1] if self.cumulative_sizes.size > 0 else 0
                )

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            if idx < 0 or idx >= self.length:
                raise IndexError("Index out of range")

            # Open file for each access and close immediately
            with h5py.File(self.h5_file, "r") as f:
                # Find which chunk the index belongs to
                chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")
                if chunk_idx > 0:
                    local_idx = idx - self.cumulative_sizes[chunk_idx - 1]
                else:
                    local_idx = idx

                embedding_key = self.embedding_keys[chunk_idx]
                # Construct corresponding keys for other data
                key_suffix = embedding_key.replace("batch_", "")
                labels_key = f"labels_batch_{key_suffix}"
                # starts_key = f"starts_batch_{key_suffix}"
                # ends_key = f"ends_batch_{key_suffix}"

                embeddings = torch.tensor(
                    f[embedding_key][local_idx], dtype=torch.float32
                )
                labels = torch.tensor(f[labels_key][local_idx], dtype=torch.long)

                return embeddings, labels

    print("Creating DomainBoundaryDataset from embeddings in H5 file...")
    # Create the dataset and dataloader
    domain_boundary_dataset = DomainBoundaryDataset("./temp/embeddings_domain.h5")

    # Split indices for train and validation sets
    dataset_size = len(domain_boundary_dataset)
    indices = list(range(dataset_size))

    # Split indices into training and validation
    val_size = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, shuffle=True, random_state=42
    )

    # Create Subset datasets
    train_dataset = torch.utils.data.Subset(domain_boundary_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(domain_boundary_dataset, val_indices)

    # Safety check if the embeddings and labels have the expected dimensions for train_dataset
    # cause by the saving to the h5 file

    sample_embedding, sample_label = train_dataset[0]

    print("shapes:", sample_embedding.dim(), sample_label.dim())

    if sample_embedding.dim() == 3 and sample_label.dim() == 2:
        print(
            "Squeezing dimensions of embeddings and labels in train and val datasets."
        )
        # Create new datasets with squeezed tensors

        class SqueezedDataset(Dataset):
            def __init__(self, original_dataset, indices):
                self.original_dataset = original_dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                original_idx = self.indices[idx]
                embedding, label = self.original_dataset[original_idx]
                return embedding.squeeze(0), label.squeeze(0)

        train_dataset = SqueezedDataset(domain_boundary_dataset, train_indices)
        val_dataset = SqueezedDataset(domain_boundary_dataset, val_indices)

        if RANK == 0:
            print(
                "Squeezed embeddings and labels to correct dimensions for train and val datasets."
            )

    # sample dim size taken from the first sample
    input_dims_sample = train_dataset[0][0].unsqueeze(0)  # [1, seq_len, embedding_dim]

    if RANK == 0:
        # print(input_dims_sample)
        print(input_dims_sample.shape)

    # Create DataLoaders for each subset
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print("Datasets and DataLoaders for domain boundary detection created.")

    # Calculate class weights by iterating through the dataset
    print("Calculating class weights for training set...")
    all_labels = []
    for i in range(0, 1000):
        _, labels = train_dataset[i]
        all_labels.append(labels)

    train_labels = torch.cat(all_labels)

    # Calculate class weights for the loss function
    train_labels_flat = train_labels.view(-1)
    counts = torch.bincount(train_labels_flat, minlength=NUM_CLASSES).float()

    # Handle cases where a class might be missing in a small sample
    if torch.any(counts == 0):
        print("Warning: One or more classes have zero samples. Using uniform weights.")
        weights = torch.ones(NUM_CLASSES, device=DEVICE)
    else:
        total = train_labels_flat.size(0)
        weights = total / (NUM_CLASSES * counts)
        # Normalize weights to prevent them from scaling the loss too much
        weights = weights * (NUM_CLASSES / weights.sum())

    weights = weights.to(DEVICE)
    if RANK == 0:
        print(f"Class counts: {counts.cpu().numpy()}")
        print(f"Calculated class weights: {weights.cpu().numpy()}")

    del all_labels, train_labels, counts

    if RANK == 0:
        study = optuna.create_study(
            direction="minimize",
            storage=f"sqlite:///logs/{PROJECT_NAME}/optuna_study.db",
            load_if_exists=True,
            study_name=PROJECT_NAME,
        )

    dist.barrier()  # Ensure all ranks are synchronized before starting the study

    if RANK != 0:
        study = optuna.create_study(
            direction="minimize",
            storage=f"sqlite:///logs/{PROJECT_NAME}/optuna_study.db",
            load_if_exists=True,
            study_name=PROJECT_NAME,
        )

    def objective_wrapper(trial):
        return objective(
            trial,
            input_dims_sample,
            train_loader,
            val_loader,
            weights=weights,
            domain_task=True,
            EPOCHS=EPOCHS,
        )

    study.optimize(
        objective_wrapper,
        n_trials=STUDY_N_TRIALS,
    )

    if RANK == 0:
        print("Best trial number:", study.best_trial.number)
        print("Best trial:", study.best_trial)
        # print("Best trial params:", study.best_trial.params)

    best_trial = study.best_trial

    lit_model = load_best_model(
        best_trial, input_dims_sample, weights=None, domain_task=True
    )

    lit_model.eval()
    if RANK == 0:
        print("Best model loaded and set to eval mode")

    # Evaluate on validation set
    device = lit_model.device
    # print(device)

    all_preds = []
    all_true = []
    sequence_level_metrics = []

    with torch.no_grad():
        for batch_embeddings, batch_labels in val_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_logits = lit_model(batch_embeddings)
            batch_preds = batch_logits.argmax(dim=-1)  # [batch, seq_len]

            # Keep sequence structure for analysis
            for i in range(batch_preds.size(0)):
                seq_preds = batch_preds[i].cpu().numpy()
                seq_labels = batch_labels[i].cpu().numpy()

                # Sequence-level metrics
                boundary_positions_pred = np.where(seq_preds == 1)[0]
                boundary_positions_true = np.where(seq_labels == 1)[0]

                # Store for sequence-level analysis
                sequence_level_metrics.append(
                    {
                        "pred_boundaries": len(boundary_positions_pred),
                        "true_boundaries": len(boundary_positions_true),
                        "correct_boundaries": len(
                            np.intersect1d(
                                boundary_positions_pred, boundary_positions_true
                            )
                        ),
                    }
                )

                # Flatten for overall metrics
                all_preds.extend(seq_preds)
                all_true.extend(seq_labels)

    # Boundary-specific metrics
    boundary_prec = precision_score(all_true, all_preds, pos_label=1, zero_division=0)
    boundary_rec = recall_score(all_true, all_preds, pos_label=1, zero_division=0)

    # Sequence-level metrics
    total_sequences = len(sequence_level_metrics)
    sequences_with_correct_boundaries = sum(
        1 for m in sequence_level_metrics if m["correct_boundaries"] > 0
    )
    avg_boundary_detection_rate = np.mean(
        [
            m["correct_boundaries"] / max(m["true_boundaries"], 1)
            for m in sequence_level_metrics
        ]
    )

    if RANK == 0:
        print("\n=== Boundary-Specific Metrics ===")
        print(f"Boundary Precision: {boundary_prec:.4f}")
        print(f"Boundary Recall: {boundary_rec:.4f}")
        print("\n=== Sequence-Level Metrics ===")
        print(
            f"Sequences with detected boundaries: {sequences_with_correct_boundaries}/{total_sequences} ({sequences_with_correct_boundaries / total_sequences:.2%})"
        )
        print(f"Average boundary detection rate: {avg_boundary_detection_rate:.4f}")

    if Final_training is True:
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, mode="min", verbose=True
        )
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

        if RANK == 0:
            print("Lit model created")
        trainer = pl.Trainer(
            max_epochs=50,
            accelerator="gpu",
            devices=-1,
            strategy="ddp",
            enable_progress_bar=True,
            callbacks=[early_stop, checkpoint_callback],
            logger=TensorBoardLogger(
                save_dir=f"./logs/{PROJECT_NAME}", name=PROJECT_NAME
            ),
        )

        if RANK == 0:
            print("-" * 100)
            print("Starting training...")
            print("-" * 100)

        trainer.fit(lit_model, train_loader, val_loader)

        # save the final model
        final_model_path = f"./models/{PROJECT_NAME}.pt"
        torch.save(lit_model, final_model_path)


#############################################################################################################

if __name__ == "__main__":
    main(Final_training=False)
    if RANK == 0:
        print("\nFinished running DomainFinder\n")
    dist.barrier()  # Ensure all processes reach this point before exiting
    dist.destroy_process_group()  # Clean up the process group
