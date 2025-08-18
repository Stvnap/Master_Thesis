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
        # time_start = time.time()
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

        # if RANK == 0:
        #     print(f"Forward pass time: {time.time() - time_start:.2f} seconds")

        return logits

# -------------------------
# 4. Loader Class
# -------------------------

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

class SqueezedDataset_Usage(Dataset):
    def __init__(self, original_dataset, actual_seq_lengths):
        self.original_dataset = original_dataset
        self.actual_seq_lengths = actual_seq_lengths

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        embedding, label = self.original_dataset[idx]  # Unpack the tuple first
        actual_length = self.actual_seq_lengths[idx]
        
        # Return embedding, sequence index, and ACTUAL sequence length (not padded length)
        return embedding.squeeze(0), idx, actual_length
# -------------------------
# 6. MAIN TRAINER
# -------------------------


def main_trainer(Final_training=False):
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
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print("Datasets and DataLoaders for domain boundary detection created.")

    # Calculate class weights by iterating through the dataset
    print("Calculating class weights for training set...")
    all_labels = []
    for i in range(0, 100):
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



def loader(ESM_Model, input_file):
    import pandas as pd

    os.makedirs("tempTest/embeddings", exist_ok=True)
    if not os.path.exists("./tempTest/embeddings/embeddings_domain.h5"):
        ESMDataset(
            FSDP_used=False,
            domain_boundary_detection=True,
            num_classes=2,
            esm_model=ESM_Model,
            csv_path=input_file,
            category_col=CATEGORY_COL,
            sequence_col=SEQUENCE_COL,
        )
        print("Using preembedded ESM data from scratch")

    # Load the original CSV to get actual sequence lengths
    df = pd.read_csv(input_file)
    actual_seq_lengths = [len(seq) for seq in df[SEQUENCE_COL]]
    
    print("Creating DomainBoundaryDataset from embeddings in H5 file...")
    # Create the dataset and dataloader
    domain_boundary_dataset = DomainBoundaryDataset("./tempTest/embeddings/embeddings_domain.h5")

    sample_embedding, sample_label = domain_boundary_dataset[0]
    print("shapes:", sample_embedding.dim(), sample_label.dim())

    if sample_embedding.dim() == 3 and sample_label.dim() == 2:
        print(
            "Squeezing dimensions of embeddings and labels in train and val datasets."
        )
        # Create new datasets with squeezed tensors
        domain_boundary_dataset_squeezed = SqueezedDataset_Usage(domain_boundary_dataset, actual_seq_lengths)

        if RANK == 0:
            print(
                "Squeezed embeddings and labels to correct dimensions for train and val datasets."
            )

    # Create DataLoader for the dataset
    domain_boudnary_set_loader = DataLoader(
        domain_boundary_dataset_squeezed,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if RANK == 0:
        print(f"Dataset: {len(domain_boudnary_set_loader)} samples")
        print("Datasets and DataLoaders for domain boundary detection created.")
        print("\nLoading Model...\n")

    model = torch.load("./models/Optuna_uncut_t33_domains_boundary.pt", map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    model.eval()

    if RANK == 0:
        print("Model loaded and set to eval mode")    

    return model, domain_boudnary_set_loader

def Predictor(model, domain_boudnary_set_loader):
    """
    Function to call the Predictor script.
    This function will be used to predict domain boundaries in sequences.
    """

    all_sequence_preds = []
    sequence_metadata = []

    # print("STARTING PREDICTION...")
    with torch.no_grad():
        for batch_data in domain_boudnary_set_loader:
            # Unpack the batch data
            if len(batch_data) == 3:  # embedding, seq_idx, actual_seq_len
                batch_embeddings, batch_seq_indices, batch_actual_lengths = batch_data
            else:  # fallback for old format
                batch_embeddings = batch_data
                batch_seq_indices = list(range(len(batch_embeddings)))
                batch_actual_lengths = [emb.shape[0] for emb in batch_embeddings]
            
            batch_embeddings = batch_embeddings.to(DEVICE)
            batch_logits = model(batch_embeddings)
            batch_preds = batch_logits.argmax(dim=-1)

            # Process each sequence in the batch separately
            for i in range(batch_preds.size(0)):
                seq_preds_full = batch_preds[i].cpu().numpy()
                seq_idx = batch_seq_indices[i] if hasattr(batch_seq_indices, '__iter__') else batch_seq_indices
                actual_len = batch_actual_lengths[i] if hasattr(batch_actual_lengths, '__iter__') else batch_actual_lengths
                
                # IMPORTANT: Truncate predictions to actual sequence length (remove padding)
                seq_preds = seq_preds_full[:actual_len]
                
                # Store predictions and metadata for this sequence
                all_sequence_preds.append(seq_preds)
                sequence_metadata.append({
                    'original_seq_idx': seq_idx,
                    'actual_seq_length': actual_len,
                    'padded_length': len(seq_preds_full),
                    'batch_idx': i
                })
    
    if RANK == 0:
        print(f"Predicted {len(all_sequence_preds)} sequences with domain boundaries.")
        # Print some debug info
        for i in range(min(3, len(all_sequence_preds))):
            metadata = sequence_metadata[i]
            print(f"Sequence {i}: actual_length={len(all_sequence_preds[i])}, "
                  f"padded_length={metadata['padded_length']}, "
                  f"actual_seq_length={metadata['actual_seq_length']}")

    return all_sequence_preds, sequence_metadata

def regions_search(all_preds, sequence_metadata=None):
    """
    Function to search for domain regions in the predictions.
    This function will return the start and end positions of each domain region.
    """
    all_regions = []
    
    for seq_idx, seq_preds in enumerate(all_preds):
        regions = []
        start = None
        structured_counter = 0
        
        # Get actual sequence length from metadata
        seq_len = len(seq_preds)  # This should now be the actual length (no padding)
        if sequence_metadata and seq_idx < len(sequence_metadata):
            original_seq_idx = sequence_metadata[seq_idx]['original_seq_idx']
            actual_seq_length = sequence_metadata[seq_idx]['actual_seq_length']
            if RANK == 0 and seq_idx < 3:  # Debug first few sequences
                print(f"Processing sequence {seq_idx} (original idx: {original_seq_idx}), "
                      f"actual length: {actual_seq_length}, predictions length: {seq_len}")
        
        for i, pred in enumerate(seq_preds):
            # Update counter based on prediction - EXTREMELY HARSH
            if pred == 1:
                structured_counter = min(structured_counter + 1, 50)  # Higher max counter
            else:
                structured_counter = max(structured_counter - 10, 0)  # EXTREMELY fast decay (was -5, now -10)
        
            # Check if we're in a structured region - EXTREMELY HIGH THRESHOLD
            if structured_counter > 35:  # Increased from 20 to 35 (requires 35+ consecutive predictions)
                if start is None:  # Start of new region
                    start = i
                    if RANK == 0 and seq_idx < 3:
                        print(f"  Structured region started at index {i}")
                
            else:  # structured_counter <= 35
                if start is not None:  # End of a region
                    end = i
                    if RANK == 0 and seq_idx < 3:
                        print(f"  Structured region ended at index {i}")
                    
                    # Only add regions that are long enough - EXTREMELY LONG MINIMUM
                    if end - start >= 100:  # Increased minimum domain length from 50 to 100 residues
                        regions.append((start, end))
                    elif RANK == 0 and seq_idx < 3:
                        print(f"    Rejected short region: length {end - start}")
                    
                    start = None
        
        # Handle case where sequence ends while in a structured region
        if start is not None:
            if seq_len - start >= 100:  # Increased minimum length check to 100
                regions.append((start, seq_len))
                if RANK == 0 and seq_idx < 3:
                    print(f"  Region ended at sequence end: {seq_len}")
            elif RANK == 0 and seq_idx < 3:
                print(f"  Rejected short final region: length {seq_len - start}")
        
        all_regions.append(regions)
    
    if RANK == 0:
        print(f"Found {len(all_regions)} sequences with domain regions.")
        for seq_idx in range(min(3, len(all_regions))):  # Only print first 3 for debugging
            regions = all_regions[seq_idx]
            metadata = sequence_metadata[seq_idx] if sequence_metadata else {}
            actual_length = metadata.get('actual_seq_length', 'unknown')
            print(f"Sequence {seq_idx} (actual length: {actual_length}) has {len(regions)} regions: {regions}")

    return all_regions

# -------------------------
# 6. MAIN USAGE
# -------------------------


def main(input_file):
    """
    Main function to run the DomainFinder script.
    This function will be used to transform the input data into a format suitable for the model.
    """

    # Call the loader function to prepare the dataset and model
    model, domain_boudnary_set_loader = loader(ESM_MODEL, input_file)

    # Call the Predictor function to predict domain boundaries
    all_preds, sequence_metadata = Predictor(model, domain_boudnary_set_loader)

    
    

    
    
    if RANK == 0:
        print("First sequence sum shape:", len(all_preds[0]) if all_preds else "No predictions")


    


    # Search for domain regions with metadata
    all_regions = regions_search(all_preds, sequence_metadata)

    if RANK == 0:
        print("First sequence regions:", all_regions[0] if all_regions else "No regions")

    # Save all_regions to file
    output_file = "./tempTest/predicted_domain_regions.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_regions, f)
    
    return all_regions

    






















#############################################################################################################
from main import input_file
import pickle

if __name__ == "__main__":

    main(input_file)
    dist.barrier()  # Ensure all processes reach this point before exiting
    dist.destroy_process_group()  # Clean up the process group
    quit()


    main_trainer(Final_training=False)
    if RANK == 0:
        print("\nFinished running DomainFinder\n")
    dist.barrier()  # Ensure all processes reach this point before exiting
    dist.destroy_process_group()  # Clean up the process group

else:
    main(input_file)