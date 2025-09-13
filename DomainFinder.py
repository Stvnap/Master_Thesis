import math
import os
import argparse
import h5py
import numpy as np
import optuna
import psutil
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# from main import input_file
import pickle
from ESM_Embeddings_HP_search import (
    ESMDataset,
    load_best_model,
    objective,
)

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))# BATCH_SIZE = 10000

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

EMB_BATCH = 64
NUM_WORKERS = min(16, os.cpu_count())
# print(f"Using {NUM_WORKERS_EMB} workers for embedding generation")
VRAM = psutil.virtual_memory().total // (1024 ** 3)  # in GB
BATCH_SIZE = 40 if VRAM >= 24 else 20 if VRAM >= 16 else 10 if VRAM >= 8 else 5
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
            batch_first=True,  
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
            
            # Read idx_multiplied data if available
            self.idx_multiplied = None
            if "idx_multiplied" in f:
                self.idx_multiplied = list(f["idx_multiplied"][:])
                if RANK == 0:
                    print(f"Found idx_multiplied data: {len(self.idx_multiplied)} entries")
                    # Show which original sequences were windowed
                    from collections import Counter
                    idx_counts = Counter(self.idx_multiplied)
                    windowed_sequences = [idx for idx, count in idx_counts.items() if count > 1]
                    print(f"Original sequences that were windowed: {windowed_sequences}")
            else:
                if RANK == 0:
                    print("No idx_multiplied data found in H5 file")

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

            embeddings = torch.tensor(
                f[embedding_key][local_idx], dtype=torch.float32
            )
            labels = torch.tensor(f[labels_key][local_idx], dtype=torch.long)

            return embeddings, labels
    
    def get_original_sequence_index(self, idx):
        """Get the original sequence index for a given dataset index"""
        if self.idx_multiplied is not None and idx < len(self.idx_multiplied):
            return self.idx_multiplied[idx]
        else:
            return idx  # fallback if no mapping available
    
    def get_windowed_sequences_info(self):
        """Get information about which sequences were windowed"""
        if self.idx_multiplied is None:
            return {}
        
        from collections import defaultdict, Counter
        
        # Count how many windows each original sequence has
        idx_counts = Counter(self.idx_multiplied)
        
        # Group dataset indices by original sequence index
        original_to_dataset_indices = defaultdict(list)
        for dataset_idx, original_idx in enumerate(self.idx_multiplied):
            original_to_dataset_indices[original_idx].append(dataset_idx)
        
        return {
            'windowed_sequences': [idx for idx, count in idx_counts.items() if count > 1],
            'sequence_window_counts': dict(idx_counts),
            'original_to_dataset_mapping': dict(original_to_dataset_indices)
        }



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
        
        # Get windowing information if available
        if hasattr(original_dataset, 'get_windowed_sequences_info'):
            self.windowing_info = original_dataset.get_windowed_sequences_info()
        else:
            self.windowing_info = {}

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        embedding, label = self.original_dataset[idx]
        actual_length = self.actual_seq_lengths[idx] if idx < len(self.actual_seq_lengths) else 1000
        
        # Get original sequence index if available
        original_seq_idx = self.original_dataset.get_original_sequence_index(idx) if hasattr(self.original_dataset, 'get_original_sequence_index') else idx
        
        # Return embedding, dataset index, actual length, and original sequence index
        return embedding.squeeze(0), idx, actual_length, original_seq_idx
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
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    print("Datasets and DataLoaders for domain boundary detection created.")

    # Calculate class weights by iterating through the dataset
    print("Calculating class weights for training set...")
    all_labels = []
    for i in range(0, 20):
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

    del all_labels, train_labels, counts, train_labels_flat, total

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

    if RANK == 0:
        print(f"Best model loaded from trial: {best_trial.number}")

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
            print()
            print("-" * 100)
            print("Starting training...")
            print("-" * 100)
            print()

        trainer.fit(lit_model, train_loader, val_loader)

        # save the final model
        final_model_path = f"./models/{PROJECT_NAME}.pt"
        torch.save(lit_model, final_model_path)


    lit_model.eval()
    if RANK == 0:
        print("Set to eval mode")

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
            emb_batch=1,
        )
        print("Using preembedded ESM data from scratch")

    print("Creating DomainBoundaryDataset from embeddings in H5 file...")
    # Create the dataset and dataloader
    domain_boundary_dataset = DomainBoundaryDataset("./tempTest/embeddings/embeddings_domain.h5")

    # Load the original CSV to get actual sequence lengths
    df = pd.read_csv(input_file)
    csv_seq_lengths = [len(seq) for seq in df[SEQUENCE_COL]]
    
    # Ensure the actual_seq_lengths matches the dataset size
    dataset_size = len(domain_boundary_dataset)
    if len(csv_seq_lengths) != dataset_size:
        print(f"Warning: CSV has {len(csv_seq_lengths)} sequences but dataset has {dataset_size} samples")
        # Truncate or pad the sequence lengths to match dataset size
        actual_seq_lengths = csv_seq_lengths[:dataset_size]

    else:
        actual_seq_lengths = csv_seq_lengths

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
    else:
        # If dimensions are correct, still need to create the usage dataset
        domain_boundary_dataset_squeezed = SqueezedDataset_Usage(domain_boundary_dataset, actual_seq_lengths)

    # Create DataLoader for the dataset
    domain_boudnary_set_loader = DataLoader(
        domain_boundary_dataset_squeezed,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    if RANK == 0:
        print(f"Dataset: {len(domain_boudnary_set_loader)} samples")
        print("Datasets and DataLoaders for domain boundary detection created.")
        print("\nLoading Model...\n")

    model = torch.load("./models/FINAL/Optuna_uncut_t33_domains_boundary.pt", map_location=DEVICE, weights_only=False)
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

    with torch.no_grad():
        # Create progress bar that only shows on rank 0
        for batch_data in tqdm(domain_boudnary_set_loader, desc="Predicting domain boundaries", disable=RANK != 0,unit="Batches", position=0, leave=True):
            # Unpack the batch data
            if len(batch_data) == 4:  # embedding, dataset_idx, actual_seq_len, original_seq_idx
                batch_embeddings, batch_dataset_indices, batch_actual_lengths, batch_original_indices = batch_data
            elif len(batch_data) == 3:  # embedding, seq_idx, actual_seq_len
                batch_embeddings, batch_dataset_indices, batch_actual_lengths = batch_data
                batch_original_indices = batch_dataset_indices  # fallback
            else:  # fallback for old format
                batch_embeddings = batch_data
                batch_dataset_indices = list(range(len(batch_embeddings)))
                batch_actual_lengths = [emb.shape[0] for emb in batch_embeddings]
                batch_original_indices = batch_dataset_indices
            
            batch_embeddings = batch_embeddings.to(DEVICE)
            batch_logits = model(batch_embeddings)
            batch_preds = batch_logits.argmax(dim=-1)

            # Process each sequence in the batch separately
            for i in range(batch_preds.size(0)):
                seq_preds_full = batch_preds[i].cpu().numpy()
                dataset_idx = batch_dataset_indices[i] if hasattr(batch_dataset_indices, '__iter__') else batch_dataset_indices
                original_idx = batch_original_indices[i] if hasattr(batch_original_indices, '__iter__') else batch_original_indices
                actual_len = batch_actual_lengths[i] if hasattr(batch_actual_lengths, '__iter__') else batch_actual_lengths
                
                # IMPORTANT: Truncate predictions to actual sequence length (remove padding)
                seq_preds = seq_preds_full[:actual_len]
                
                # Store predictions and metadata for this sequence
                all_sequence_preds.append(seq_preds)
                sequence_metadata.append({
                    'dataset_idx': dataset_idx,
                    'original_seq_idx': original_idx,
                    'actual_seq_length': actual_len,
                    'padded_length': len(seq_preds_full),
                    'batch_idx': i
                })
    
    if RANK == 0:
        print(f"Predicted {len(all_sequence_preds)} sequences with domain boundaries.")
        
        # Show information about windowed sequences
        original_indices = [meta['original_seq_idx'] for meta in sequence_metadata]
        from collections import Counter
        original_counts = Counter(original_indices)
        windowed_originals = [idx for idx, count in original_counts.items() if count > 1]
        
        if windowed_originals:
            print(f"Original sequences with multiple windows: {windowed_originals}")
            for orig_idx in windowed_originals[:3]:  # Show first 3
                windows = [i for i, meta in enumerate(sequence_metadata) if meta['original_seq_idx'] == orig_idx]
                print(f"  Original sequence {orig_idx} -> dataset indices {windows}")
        
        # Print debug info for first few sequences
        for i in range(min(3, len(all_sequence_preds))):
            metadata = sequence_metadata[i]
            print(f"Dataset sequence {i}: original_seq={metadata['original_seq_idx']}, "
                  f"actual_length={len(all_sequence_preds[i])}, "
                  f"padded_length={metadata['padded_length']}")

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
        in_structured_region = False
        
        # Get actual sequence length from metadata
        seq_len = len(seq_preds)
        if sequence_metadata and seq_idx < len(sequence_metadata):
            original_seq_idx = sequence_metadata[seq_idx]['original_seq_idx']
            actual_seq_length = sequence_metadata[seq_idx]['actual_seq_length']
            if RANK == 0 and seq_idx < 3:
                print(f"Processing sequence {seq_idx} (original idx: {original_seq_idx}), "
                      f"actual length: {actual_seq_length}, predictions length: {seq_len}")
        
        # Find first occurrence of just 2 consecutive 1's to start structured region (more liberal)
        consecutive_ones_start = None
        for i in range(seq_len - 1):  # Changed from seq_len - 2 to seq_len - 1
            if seq_preds[i] == 1 and seq_preds[i + 1] == 1:  # Changed from 3 consecutive to 2 consecutive
                consecutive_ones_start = i + 1  
                break
        
        # If we found 2 consecutive 1's at the beginning, start region there
        if consecutive_ones_start is not None:
            start = consecutive_ones_start
            in_structured_region = True
            structured_counter = 20  # Increased from 15 to 20 for more stability
            if RANK == 0 and seq_idx < 3:
                print(f"  Structured region started at first consecutive 1's at position {consecutive_ones_start} (counter: {structured_counter})")
        
        for i, pred in enumerate(seq_preds):
            # Update counter based on prediction (even more generous)
            if pred == 1:
                structured_counter = min(structured_counter + 3, 60)  # Changed from +2 to +3, max from 50 to 60
            else:
                structured_counter = max(structured_counter - 5, 0)   # Changed from -10 to -5 (much less penalty)
        
            # Check if we should start a new region (much more liberal threshold)
            if structured_counter > 5 and not in_structured_region:   # Changed from 8 to 5
                start = i+1
                in_structured_region = True
                if RANK == 0 and seq_idx < 3:
                    print(f"  Structured region started at index {i} (counter: {structured_counter})")
            
            # Check if we should end the current region (very liberal ending)
            elif structured_counter <= 1 and in_structured_region:  # Changed from 2 to 1
                end = i+1
                in_structured_region = False
                if RANK == 0 and seq_idx < 3:
                    print(f"  Structured region ended at index {i} (counter: {structured_counter})")
                
                # Only add regions that are long enough (very liberal minimum length)
                region_length = end - start
                if region_length >= 15:  # Changed from 20 to 15 amino acids
                    regions.append((start, end))
                    if RANK == 0 and seq_idx < 3:
                        print(f"    Added region: ({start}, {end}) length: {region_length}")
                elif RANK == 0 and seq_idx < 3:
                    print(f"    Rejected short region: length {region_length}")
                
                start = None
        
        # Handle case where sequence ends while in a structured region (very liberal)
        if in_structured_region and start is not None:
            region_length = seq_len - start
            if region_length >= 10:  # Changed from 20 to 10 (very short regions allowed at end)
                regions.append((start, seq_len))
                if RANK == 0 and seq_idx < 3:
                    print(f"  Region ended at sequence end: ({start}, {seq_len}) length: {region_length}")
            elif RANK == 0 and seq_idx < 3:
                print(f"  Rejected short final region: length {region_length}")
        
        all_regions.append(regions)
    
    if RANK == 0:
        print(f"Found {len(all_regions)} sequences with domain regions.")
        for seq_idx in range(min(3, len(all_regions))):
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
        pickle.dump(sequence_metadata, f)
    
    return all_regions

    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Domain Finder Script")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, default="./tempTest/predicted_domain_regions.pkl", help="Output file path")
    parser.add_argument("--model", type=str, required=True, help="ESM model name")
    parser.add_argument("--TrainerMode", type=str, default="False", help="Set to True to run the trainer, False to run the predictor")
    return parser.parse_args()

# Parse arguments at the beginning
args = parse_arguments()

# Convert TrainerMode string to boolean
TRAINER_MODE = args.TrainerMode.lower() == "true"
input_file = args.input
#############################################################################################################

if __name__ == "__main__":
    if TRAINER_MODE is True:
        main_trainer(Final_training=False)
    
    else:
        main(input_file)

        if RANK == 0:
            print("\nFinished running DomainFinder\n")
        dist.barrier()  # Ensure all processes reach this point before exiting
        dist.destroy_process_group()  # Clean up the process group
        quit()