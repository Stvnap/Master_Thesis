"""
DomainFinder.py

Table of Contents:
===================
1. Imports
2. Global settings & basic setup
3. Transformer Class
4. Dataset Class
5. Helper Functions for Main Trainer
6. MAIN TRAINER
7. Helper Functions for Usage Main
8. USAGE MAIN
9. Main Execution

CLASSES:
--------
- Transformer
- DomainBoundaryDataset
- SqueezedDataset
- SqueezedDataset_Usage

FUNCTIONS:
--------
- opener
- class_weights_calculator
- HP_search
- final_trainer
- evaluator
- main_trainer
- usage_opener
- Predictor
- regions_search
- main
- parse_arguments
"""

# -------------------------
# 1. Imports
# -------------------------
import argparse
import math
import os
import pickle
from collections import Counter, defaultdict

import h5py
import numpy as np
import optuna
import pandas as pd
import psutil
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.ndimage import binary_closing, binary_opening
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ESM_Embeddings_HP_search import (
    ESMDataset,
    load_best_model,
    objective,
)

# -------------------------
# 2. Global settings & basic setup
# -------------------------

# ddp setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device setup
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
if not dist.is_initialized():
    dist.init_process_group("nccl")
    if dist.get_rank() == 0:
        print("Initializing process group for DDP")
RANK = dist.get_rank()  # rank of the current process

# GLOBAL SETTINGS AND PATHS
CSV_PATH = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/FoundEntriesSwissProteins.csv"  # path to the csv file with sequences
CATEGORY_COL = "Pfam_id"  # category column in the csv file
SEQUENCE_COL = "Sequence"  # sequence column in the csv file
CACHE_PATH = "./pickle/FoundEntriesSwissProteins_domains.pkl"  # path to cache pickle file
MODEL_NAME = "Optuna_uncut_t33_domains_boundary"  # model name for saving
PROJECT_NAME = (
    "Optuna_uncut_t33_domains_boundary"  # project name for logging and model saving
)
ESM_MODEL = "esm2_t33_650M_UR50D"  # ESM model to use
NUM_CLASSES = (
    2  # Number of classes for domain detection. Keep 2 for boundary vs non-boundary
)
TRAIN_FRAC = 0.6  # Fraction of data used for training
VAL_FRAC = 0.2  # ... for validation
TEST_FRAC = 0.2  # ... for testing
NUM_WORKERS = min(16, os.cpu_count())  # number of DataLoader workers
VRAM = psutil.virtual_memory().total // (1024**3)  # systems VRAM in GB
BATCH_SIZE = (
    40 if VRAM >= 24 else 20 if VRAM >= 16 else 10 if VRAM >= 8 else 5
)  # adjust batch size based on available VRAM
EPOCHS = 50  # number of training epochs
STUDY_N_TRIALS = 3  # number of optuna trials


# -------------------------
# 3. Transformer Class
# -------------------------
class Transformer(nn.Module):
    """
    Main Transformer model for domain boundary detection using ESM embeddings. Based on PyTorch's nn.Transformer.
    Based on BERT architecture (encoder only transformer) for sequence labeling tasks.
    Uses:
        - Multi-head self-attention
        - Positional encoding
        - Feedforward neural network layers
    Outputs per-position class logits for domain boundary classification.

    Args:
        input_dim (int): Dimension of input ESM embeddings (e.g., 1280 for ESM2-t33).
        hidden_dims (int): Dimension of hidden layers in the transformer.
        num_classes (int): Number of output classes (always 2 for boundary vs non-boundary).
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        activation (str): Activation function to use.
        kernel_init (str): Kernel initialization method.
        num_layers (int): Number of transformer encoder layers.
    """

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
        # class based on https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        super(Transformer, self).__init__()
        # init parameters
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.kernel_init = kernel_init
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers

        # Ensure input_dim is divisible by num_heads for attention mechanism
        assert input_dim % num_heads == 0, (
            f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # input dimension
            nhead=num_heads,  # number of attention heads
            dim_feedforward=input_dim * 4,  # Standard: 4x model dimension
            dropout=dropout,  # dropout rate
            activation=activation,  # activation function
            batch_first=True,  # batch dimension first
        )

        # Stack multiple encoder layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Per-position domain boundary classifier, simple FFN on top of transformer outputs with 2 layers and 2 outputs (boundary vs non-boundary)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),  # reduce dimension to half
            activation,  # activation given as argument
            nn.Dropout(dropout),  # dropout
            nn.Linear(input_dim // 2, num_classes),  # final output layer
        )

    def generate_positional_encoding(self, seq_len, d_model, device):
        """
        Generate positional encodings for input sequences. Basic sinusoidal positional encoding based on the paper "Attention is All You Need".
        Args:
            seq_len (int): Length of the input sequence.
            d_model (int): Dimension of the model (input embeddings).
            device (torch.device): Device to create the positional encodings on.
        returns:
            pe (torch.Tensor): Positional encodings of shape [1, seq_len, d_model].
        """
        # Create positional encodings
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
        """
        Forward pass of the transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
        returns:
            logits (torch.Tensor): Output logits of shape [batch_size, seq_len, num_classes].
        """

        # unpack input dimensions
        batch_size, seq_len, input_dim = x.size()

        # Add positional encoding
        pe = self.generate_positional_encoding(seq_len, input_dim, x.device)
        x = x + pe  # Broadcasting: [batch, seq, dim] + [1, seq, dim]

        # Apply dropout
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Transformer processing, encoding the input sequence for contextual information
        x = self.encoder(x)  # [batch_size, seq_len, input_dim]

        # Put per residue information through classifier to get logits for boundary vs non-boundary
        logits = self.classifier(x)  # [batch_size, seq_len, num_classes]

        return logits


# -------------------------
# 4. Dataset Class
# -------------------------


class DomainBoundaryDataset(Dataset):
    """
    Custom dataset for domain boundary detection with ESM embeddings that loads data from the embedded H5 file.
    Args:
        h5_file (str): Path to the H5 file containing ESM embeddings and labels.
    """

    def __init__(self, h5_file):
        self.h5_file = h5_file

        # Determine the total length by inspecting the H5 file
        with h5py.File(self.h5_file, "r") as f:
            # get embedding keys and their sizes
            self.embedding_keys = sorted(
                [k for k in f.keys() if k.startswith("batch_")]
            )
            # chunk sizes and cumulative sizes for indexing
            self.chunk_sizes = [f[key].shape[0] for key in self.embedding_keys]
            self.cumulative_sizes = np.cumsum(self.chunk_sizes)
            # based on those get length
            self.length = (
                self.cumulative_sizes[-1] if self.cumulative_sizes.size > 0 else 0
            )

            # Read idx_multiplied data if available used to track windowed sequences (if len(sequence) > 1000)
            self.idx_multiplied = None
            if "idx_multiplied" in f:
                self.idx_multiplied = list(f["idx_multiplied"][:])
                if RANK == 0:
                    print(
                        f"Found idx_multiplied data: {len(self.idx_multiplied)} entries"
                    )

                    # Show which original sequences were windowed
                    idx_counts = Counter(self.idx_multiplied)
                    windowed_sequences = [
                        idx for idx, count in idx_counts.items() if count > 1
                    ]
                    print(
                        f"Original sequences that were windowed: {windowed_sequences}"
                    )

    def __len__(self):
        # simple return from precomputed length
        return self.length

    def __getitem__(self, idx):
        """
        Get embedding and label for a given index
        """
        # Check index bounds
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range")

        # Open file for each access
        with h5py.File(self.h5_file, "r") as f:
            # Find which chunk the index belongs to
            chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")
            # Calculate local index within the chunk
            if chunk_idx > 0:
                local_idx = idx - self.cumulative_sizes[chunk_idx - 1]
            else:
                local_idx = idx

            # get embedding_key and labels_key
            embedding_key = self.embedding_keys[chunk_idx]
            # remove batch from embedding_key to get corresponding labels key
            key_suffix = embedding_key.replace("batch_", "")
            labels_key = f"labels_batch_{key_suffix}"

            # Load embedding and label
            embeddings = torch.tensor(f[embedding_key][local_idx], dtype=torch.float32)
            labels = torch.tensor(f[labels_key][local_idx], dtype=torch.long)

            return embeddings, labels

    def get_original_sequence_index(self, idx):
        """
        Get the original sequence index for a given dataset index. Used for tracking windowed sequences.
        """
        # Return original index if idx_multiplied mapping is available
        if self.idx_multiplied is not None and idx < len(self.idx_multiplied):
            return self.idx_multiplied[idx]
        # fallback if no mapping available
        else:
            return idx

    def get_windowed_sequences_info(self):
        """
        Get information about which sequences were windowed
        """
        # Return empty dict if no idx_multiplied data
        if self.idx_multiplied is None:
            return {}

        # Count how many windows each original sequence has
        idx_counts = Counter(self.idx_multiplied)

        # Group dataset indices by original sequence index
        original_to_dataset_indices = defaultdict(list)
        for dataset_idx, original_idx in enumerate(self.idx_multiplied):
            original_to_dataset_indices[original_idx].append(dataset_idx)

        # Return the information
        return {
            "windowed_sequences": [
                idx for idx, count in idx_counts.items() if count > 1
            ],
            "sequence_window_counts": dict(idx_counts),
            "original_to_dataset_mapping": dict(original_to_dataset_indices),
        }


class SqueezedDataset(Dataset):
    """
    Dataset to squeeze embeddings and labels for correct dimensions after loading from H5 file.
    """

    def __init__(self, original_dataset, indices):
        # init with original dataset and indices
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        # nothing special just return length
        return len(self.indices)

    def __getitem__(self, idx):
        # get item from original dataset and squeeze dimensions when returning
        original_idx = self.indices[idx]
        embedding, label = self.original_dataset[original_idx]
        return embedding.squeeze(0), label.squeeze(0)


class SqueezedDataset_Usage(Dataset):
    """
    Analog of SqueezedDataset but returns additional information for tracking windowed sequences.
    Needed for final dataframe creation after prediction.
    """

    def __init__(self, original_dataset, actual_seq_lengths):
        # init with original dataset and actual sequence lengths
        self.original_dataset = original_dataset
        self.actual_seq_lengths = actual_seq_lengths

        # Get windowing information if available
        if hasattr(original_dataset, "get_windowed_sequences_info"):
            self.windowing_info = original_dataset.get_windowed_sequences_info()
        else:
            self.windowing_info = {}

    def __len__(self):
        # basic len return
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # get item from original dataset
        embedding, label = self.original_dataset[idx]
        # get actual sequence length
        actual_length = (
            self.actual_seq_lengths[idx] if idx < len(self.actual_seq_lengths) else 1000
        )

        # Get original sequence index if available
        original_seq_idx = (
            self.original_dataset.get_original_sequence_index(idx)
            if hasattr(self.original_dataset, "get_original_sequence_index")
            else idx
        )

        # Return embedding, dataset index, actual length, and original sequence index
        return embedding.squeeze(0), idx, actual_length, original_seq_idx


# -------------------------
# 5. Helper Functions for Main Trainer
# -------------------------


def opener():
    """
    Opens and prepares the dataset and dataloader for domain boundary detection.
    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        train_dataset (Dataset): Training dataset. Used for class weight calculation.
        sample_embedding (torch.Tensor): Sample embedding tensor for input dimension reference.
    """
    # Create necessary directories
    os.makedirs(f"logs/{PROJECT_NAME}", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("temp", exist_ok=True)

    # Check if embeddings H5 file exists, if not create it from scratch
    if not os.path.exists(
        "./temp/embeddings_domain.h5"
    ):
        if RANK == 0:
            print("Creating DomainBoundaryDataset from embeddings in H5 file...")
        ESMDataset(
            FSDP_used=False,
            domain_boundary_detection=True,
            num_classes=NUM_CLASSES,
            esm_model=ESM_MODEL,
            csv_path=CSV_PATH,
            category_col=CATEGORY_COL,
            sequence_col=SEQUENCE_COL,
        )
    # Create the dataset and dataloader
    if RANK == 0:
        print("Using preembedded ESM data from scratch")
    domain_boundary_dataset = DomainBoundaryDataset(
        "./temp/embeddings_domain.h5"
    )

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

    # Check and squeeze dimensions if necessary
    sample_embedding, sample_label = train_dataset[0]
    if sample_embedding.dim() == 3 and sample_label.dim() == 2:
        # Create new datasets with squeezed dimensions in SqueezedDataset
        train_dataset = SqueezedDataset(domain_boundary_dataset, train_indices)
        val_dataset = SqueezedDataset(domain_boundary_dataset, val_indices)
        if RANK == 0:
            print(
                "Squeezed embeddings and labels to correct dimensions for train and val datasets."
            )

    # Create DataLoaders for each subset
    train_loader = DataLoader(
        train_dataset,  # train dataset
        batch_size=BATCH_SIZE,  # batch size from global setting
        shuffle=True,  # shuffle for training
        num_workers=NUM_WORKERS,  # number of workers from global setting
        pin_memory=True,  # pin memory for faster transfers
        persistent_workers=True,  # keep workers alive, faster
        prefetch_factor=2,  # prefetch factor for workers
    )
    val_loader = DataLoader(
        val_dataset,  # val dataset
        batch_size=BATCH_SIZE,  # batch size from global setting
        shuffle=False,  # no shuffle for validation, recommended by PyTorch
        num_workers=NUM_WORKERS,  # number of workers from global setting
        pin_memory=True,  # pin memory for faster transfers
        persistent_workers=True,  # keep workers alive, faster
        prefetch_factor=2,  # prefetch factor for workers
    )

    # Print dataset sizes
    if RANK == 0:
        print(f"Train set: {len(train_dataset)} samples")
        print(f"Val set: {len(val_dataset)} samples")
        print("Datasets and DataLoaders for domain boundary detection created.")
        # Calculate class weights by iterating through the dataset
        print("Calculating class weights for training set...")

    return train_loader, val_loader, train_dataset, sample_embedding


def class_weights_calculator(train_dataset):
    """
    Calculate class weights based on the frequency of each class in the training dataset. These weights can be used to handle class imbalance during training in the loss function.
    Args:
        train_dataset (Dataset): Training dataset to calculate class weights from.
    Returns:
        weights (torch.Tensor): Tensor of class weights (2 floats for boundary vs non-boundary).
    """
    # iterating through a fraction of the training dataset to get a fast estimate
    all_labels = []
    for i in range(0, 20):
        _, labels = train_dataset[i]
        all_labels.append(labels)

    # transform list of tensors to single tensor
    train_labels = torch.cat(all_labels)

    # Calculate class weights for the loss function
    train_labels_flat = train_labels.view(-1)
    # bin count for the 2 classes
    counts = torch.bincount(train_labels_flat, minlength=NUM_CLASSES).float()

    # Handle cases where a class might be missing in a small sample
    if torch.any(counts == 0):
        if RANK == 0:
            print(
                "Warning: One or more classes have zero samples. Using uniform weights."
            )
        weights = torch.ones(NUM_CLASSES, device=DEVICE)
    # normal case
    else:
        total = train_labels_flat.size(0)
        # compute weights inversely proportional to class frequencies
        weights = total / (NUM_CLASSES * counts)
        # Normalize weights to prevent them from scaling the loss too much
        weights = weights * (NUM_CLASSES / weights.sum())
    # move weights to device
    weights = weights.to(DEVICE)

    # print class counts and weights
    if RANK == 0:
        print(f"Class counts: {counts.cpu().numpy()}")
        print(f"Calculated class weights: {weights.cpu().numpy()}")

    # free memory
    del all_labels, train_labels, counts, train_labels_flat, total

    return weights


def HP_search(train_loader, val_loader, sample_embedding, weights):
    """
    Perform hyperparameter optimization using Optuna to find the best model configuration for domain boundary detection.
    Uses some helper functions from ESM_Embeddings_HP_search.py, look there for full explanations.
    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        sample_embedding (torch.Tensor): Sample embedding tensor for input dimension reference.
        weights (torch.Tensor): Class weights for the loss function during training.
    Returns:
        lit_model (pl.LightningModule): The best model found during hyperparameter optimization.
    """
    # rank 0 creates the study
    if RANK == 0:
        study = optuna.create_study(
            direction="minimize",
            storage=f"sqlite:///logs/{PROJECT_NAME}/optuna_study.db",
            load_if_exists=True,
            study_name=PROJECT_NAME,
        )

    # Ensure all ranks are synchronized before starting the study
    dist.barrier()

    # other ranks load the study
    if RANK != 0:
        study = optuna.create_study(
            direction="minimize",
            storage=f"sqlite:///logs/{PROJECT_NAME}/optuna_study.db",
            load_if_exists=True,
            study_name=PROJECT_NAME,
        )

    # get sample input dimensions for objective function
    input_dims_sample = sample_embedding.shape[1]

    def objective_wrapper(trial):
        # Wrapper to pass additional arguments to the objective function, loaded from ESM_Embeddings_HP_search.py.
        return objective(
            trial,
            input_dims_sample,
            train_loader,
            val_loader,
            weights=weights,
            domain_task=True,
            EPOCHS=EPOCHS,
        )

    # starting the hyperparameter optimization
    study.optimize(
        objective_wrapper,
        n_trials=STUDY_N_TRIALS,
    )

    # print best trial info
    if RANK == 0:
        print("Best trial number:", study.best_trial.number)
        print("Best trial:", study.best_trial)

    # Load the best model from the best trial
    best_trial = study.best_trial

    # load the best model using helper function from ESM_Embeddings_HP_search.py
    lit_model = load_best_model(
        best_trial, input_dims_sample, weights=None, domain_task=True
    )

    # print info
    if RANK == 0:
        print(f"Best model loaded from trial: {best_trial.number}")

    return lit_model


def final_trainer(lit_model, train_loader, val_loader):
    """
    Final training of the best model found during hyperparameter optimization. Has early stopping and model checkpointing, and tensorboard logging.
    Args:
        lit_model (pl.LightningModule): The best model found during hyperparameter optimization.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
    Returns:
        lit_model (pl.LightningModule): The final trained model.
    """
    # init early stopping and checkpoint callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    # init pytorch lightning trainer with DDP
    trainer = pl.Trainer(
        max_epochs=EPOCHS,  # number of epochs from global setting
        accelerator="gpu",  # use gpu acceleration
        devices=-1,  # use all available gpus
        strategy="ddp",  # distributed data parallel strategy
        enable_progress_bar=True,  # enable progress bar
        callbacks=[
            early_stop,
            checkpoint_callback,
        ],  # callbacks for early stopping and checkpointing
        logger=TensorBoardLogger(
            save_dir=f"./logs/{PROJECT_NAME}",
            name=PROJECT_NAME,
        ),  # tensorboard logger to save logs
    )

    # overview print
    if RANK == 0:
        print()
        print("-" * 100)
        print("Starting training...")
        print("-" * 100)
        print()

    # start final training with model and the two loaders
    trainer.fit(lit_model, train_loader, val_loader)

    return lit_model


def evaluator(final_model_path, val_loader):
    """
    Evaluate the final trained model on the validation set and compute boundary-specific and sequence-level metrics to assess performance.
    Args:
        final_model_path (str): Path to the final trained model file previously saved. Ensures correct loading and device placement.
        val_loader (DataLoader): DataLoader for the validation set. Only validation set is used for evaluation here to get a faster estimate of performance.
    """
    # reload the final model for evaluation to ensure correct loading & device placement + eval mode
    lit_model = torch.load(final_model_path, map_location=DEVICE, weights_only=False)
    lit_model = lit_model.to(DEVICE)
    lit_model.eval()

    # inference_mode for evaluation to reduce memory usage, no need to track gradients
    with torch.inference_mode():
        # Initialize metrics accumulators instead of storing all data
        total_boundary_tp = 0
        total_boundary_fp = 0
        total_boundary_fn = 0
        total_sequences = 0
        total_correct_boundaries = 0
        total_true_boundaries = 0
        sequences_with_any_detected_boundary = 0

        # Iterate through validation data loader
        for batch_idx, (batch_embeddings, batch_labels) in enumerate(
            tqdm(
                val_loader,
                desc="Evaluating model",
                disable=RANK != 0,
                unit="Batches",
                position=0,
                leave=True,
            )
        ):
            # Move batch to device
            batch_embeddings = batch_embeddings.to(DEVICE)
            # put inputs through model
            batch_logits = lit_model(batch_embeddings)
            # get predictions via argmax
            batch_preds = batch_logits.argmax(dim=-1)  # [batch, seq_len]

            # Process batch predictions and labels to update metrics
            for i in range(batch_preds.size(0)):
                # Get individual sequence predictions and labels
                seq_preds = batch_preds[i].cpu().numpy()
                seq_labels = batch_labels[i].cpu().numpy()

                # Calculate sequence-level metrics immediately
                boundary_positions_pred = np.where(seq_preds == 1)[0]
                boundary_positions_true = np.where(seq_labels == 1)[0]

                # Check if any boundaries were detected in this sequence
                if len(boundary_positions_pred) > 0:
                    sequences_with_any_detected_boundary += 1

                # Accumulate metrics without storing individual sequences
                correct_boundaries = len(
                    np.intersect1d(boundary_positions_pred, boundary_positions_true)
                )
                total_correct_boundaries += correct_boundaries
                total_true_boundaries += len(boundary_positions_true)
                total_sequences += 1

                # Calculate position-level TP, FP, FN for this sequence
                tp = correct_boundaries
                fp = len(boundary_positions_pred) - correct_boundaries
                fn = len(boundary_positions_true) - correct_boundaries

                # Accumulate position-level metrics
                total_boundary_tp += tp
                total_boundary_fp += fp
                total_boundary_fn += fn

            # Clear batch data from memory
            del batch_embeddings, batch_logits, batch_preds, batch_labels

    # Calculate final metrics from accumulators
    boundary_prec = total_boundary_tp / max(total_boundary_tp + total_boundary_fp, 1)
    boundary_rec = total_boundary_tp / max(total_boundary_tp + total_boundary_fn, 1)

    # average boundary detection rate
    avg_boundary_detection_rate = total_correct_boundaries / max(
        total_true_boundaries, 1
    )

    # Print final metrics
    if RANK == 0:
        print("\n=== Boundary-Specific Metrics ===")
        print(f"Boundary Precision: {boundary_prec:.4f}")
        print(f"Boundary Recall: {boundary_rec:.4f}")
        print("\n=== Sequence-Level Metrics ===")
        print(
            f"Sequences with detected boundaries: {sequences_with_any_detected_boundary}/{total_sequences} ({sequences_with_any_detected_boundary / total_sequences:.2%})"
        )
        print(f"Average boundary detection rate: {avg_boundary_detection_rate:.4f}")


# -------------------------
# 6. MAIN TRAINER
# -------------------------


def main_trainer(Final_training=False):
    """
    Main Funtion for training the domain boundary detection model with optuna hyperparameter optimization. Execution of all helper functions and classes happens here for training.
    Args:
        Final_training (bool): If True, performs final training of the best model after hyperparameter optimization.
    """

    # open datasets and dataloaders
    train_loader, val_loader, train_dataset, sample_embedding = opener()

    # calculate class weights for loss function
    weights = class_weights_calculator(train_dataset)

    # hyperparameter optimization with optuna
    lit_model = HP_search(train_loader, val_loader, sample_embedding, weights)

    if Final_training is True:
        lit_model = final_trainer(lit_model, train_loader, val_loader)
    # save the final model with dynamic file name
    final_model_path = f"./models/{MODEL_NAME}.pt"
    torch.save(lit_model, final_model_path)

    # evaluate the final model
    evaluator(final_model_path, val_loader)


# -------------------------
# 7. Helper Functions for Usage Main
# -------------------------


def usage_opener(ESM_Model, input_file):
    """
    Opener to load in the user data, embedd them and load the model specified to use during prediction. Creates the dataset and dataloader for prediction. Similar to the basic opener but adapted for usage.
    Args:
        ESM_Model (str): ESM model to use for embedding generation.
        input_file (str): Path to the input CSV file with sequences to predict on.
    Returns:
        model (nn.Module): Loaded domain boundary detection model.
        domain_boudnary_set_loader (DataLoader): DataLoader for the domain boundary detection dataset.
    """
    # Create necessary directories
    os.makedirs("tempUsage/embeddings", exist_ok=True)
    os.makedirs("/global/scratch2/sapelt/tempUsage/embeddings", exist_ok=True)
    # Check if embeddings H5 file exists, if not create it from scratch
    if not os.path.exists(
        "/global/scratch2/sapelt/tempUsage/embeddings/embeddings_domain.h5"
    ):
        if RANK == 0:
            print("Generating embeddings with ESM model...")
        ESMDataset(
            FSDP_used=False,
            domain_boundary_detection=True,
            num_classes=2,
            esm_model=ESM_Model,
            csv_path=input_file,
            category_col=CATEGORY_COL,
            sequence_col=SEQUENCE_COL,
            
        )
        if RANK == 0:
            print("Embeddings generated and saved.")
    else:
        if RANK == 0:
            print("Using preembedded ESM data from scratch")

    # Create the dataset and dataloader
    domain_boundary_dataset = DomainBoundaryDataset(
        "/global/scratch2/sapelt/tempUsage/embeddings/embeddings_domain.h5"
    )

    # Load the original CSV to get actual sequence lengths neeeded for usage dataset
    df = pd.read_csv(input_file)
    csv_seq_lengths = [len(seq) for seq in df[SEQUENCE_COL]]

    # Ensure the actual_seq_lengths matches the dataset size
    dataset_size = len(domain_boundary_dataset)
    if len(csv_seq_lengths) != dataset_size:
        if RANK == 0:
            print(
                f"Warning: CSV has {len(csv_seq_lengths)} sequences but dataset has {dataset_size} samples. Truncuating based on possible doublication..."
            )
        # Truncate or pad the sequence lengths to match dataset size. Artifact of windowing, so truncation is logically correct.
        actual_seq_lengths = csv_seq_lengths[:dataset_size]
    # normal case, keep unchanged
    else:
        actual_seq_lengths = csv_seq_lengths

    # Create the usage dataset (handles squeezing)
    domain_boundary_dataset_squeezed = SqueezedDataset_Usage(
        domain_boundary_dataset, actual_seq_lengths
    )

    # Create DataLoader for the dataset
    domain_boudnary_set_loader = DataLoader(
        domain_boundary_dataset_squeezed,  # squeezed dataset for usage
        batch_size=BATCH_SIZE,  # Global batch size
        shuffle=False,  # no shuffle for usage / prediction. important to keep order
        num_workers=NUM_WORKERS,  # Global number of workers
        pin_memory=True,  # pin memory for faster transfers
        persistent_workers=True,  # keep workers alive, faster
        prefetch_factor=2,  # prefetch factor for workers
    )

    # Load the trained model for prediction, move to device and set to eval mode
    model = torch.load(
        "./models/FINAL/Optuna_uncut_t33_domains_boundary.pt",
        map_location=DEVICE,
        weights_only=False,
    )
    model = model.to(DEVICE)
    model.eval()

    return model, domain_boudnary_set_loader


def Predictor(model, domain_boudnary_set_loader):
    """
    Function to call the Predictor script. This function will be used to predict domain boundaries in sequences. It also returns metadata for each sequence to help with downstream processing to keep track of the order and windowed sequences.
    Args:
        model (nn.Module): The trained domain boundary detection model.
        domain_boudnary_set_loader (DataLoader): DataLoader for the domain boundary detection dataset.
    Returns:
        all_sequence_preds (list of np.ndarray): List of predicted domain boundary arrays for each sequence.
        sequence_metadata (list of dict): List of metadata dictionaries for each sequence.
    """

    # Initialize lists to store all predictions and metadata
    all_sequence_preds = []
    sequence_metadata = []

    # inference mode for prediction to reduce memory usage, no need to track gradients
    with torch.inference_mode():
        # Create progress bar that only shows on rank 0
        for batch_data in tqdm(
            domain_boudnary_set_loader,
            desc="Predicting domain boundaries",
            disable=RANK != 0,
            unit="Batches",
            position=0,
            leave=True,
        ):
            # Unpack the batch data based on different possible formats
            if (
                len(batch_data) == 4
            ):  # embedding, dataset_idx, actual_seq_len, original_seq_idx
                (
                    batch_embeddings,
                    batch_dataset_indices,
                    batch_actual_lengths,
                    batch_original_indices,
                ) = batch_data
            elif len(batch_data) == 3:  # embedding, seq_idx, actual_seq_len
                batch_embeddings, batch_dataset_indices, batch_actual_lengths = (
                    batch_data
                )
                batch_original_indices = batch_dataset_indices  # lastfallback
            else:  # fallback for old formats if still needed
                batch_embeddings = batch_data
                batch_dataset_indices = list(range(len(batch_embeddings)))
                batch_actual_lengths = [emb.shape[0] for emb in batch_embeddings]
                batch_original_indices = batch_dataset_indices

            # Move embeddings to device
            batch_embeddings = batch_embeddings.to(DEVICE)
            # put embeddings through model to get logits
            batch_logits = model(batch_embeddings)
            # get predictions via argmax
            batch_preds = batch_logits.argmax(dim=-1)

            # Process each prediction in the batch separately
            for i in range(batch_preds.size(0)):
                # Get full sequence predictions and associated metadata
                seq_preds_full = batch_preds[i].cpu().numpy()
                # Get dataset index, original index, and actual length for this sequence
                dataset_idx = (
                    batch_dataset_indices[i]
                    if hasattr(batch_dataset_indices, "__iter__")
                    else batch_dataset_indices
                )
                original_idx = (
                    batch_original_indices[i]
                    if hasattr(batch_original_indices, "__iter__")
                    else batch_original_indices
                )
                actual_len = (
                    batch_actual_lengths[i]
                    if hasattr(batch_actual_lengths, "__iter__")
                    else batch_actual_lengths
                )

                # Truncate predictions to actual sequence length (remove padding)
                seq_preds = seq_preds_full[:actual_len]

                # Store predictions and metadata for this sequence
                all_sequence_preds.append(seq_preds)
                sequence_metadata.append(
                    {
                        "dataset_idx": dataset_idx,
                        "original_seq_idx": original_idx,
                        "actual_seq_length": actual_len,
                        "padded_length": len(seq_preds_full),
                        "batch_idx": i,
                    }
                )

        # After prediction, analyze windowed sequences if any
        original_indices = [meta["original_seq_idx"] for meta in sequence_metadata]
        original_counts = Counter(original_indices)
        windowed_originals = [
            idx for idx, count in original_counts.items() if count > 1
        ]
        # Print info about windowed sequences
        if windowed_originals:
            if RANK == 0:
                print(f"Original sequences with multiple windows: {windowed_originals}")

    return all_sequence_preds, sequence_metadata


def regions_search(all_preds):
    """
    Function to search for domain regions using morphological operations, based on scipy.ndimage. This function processes the predicted boundary arrays to identify contiguous domain regions.    This function will return the start and end positions of each domain region.
    Could use more tuning or switch to more advanced methods in the future, if recall
    Args:
        all_preds (list of np.ndarray): List of predicted domain boundary arrays for each sequence.
    Returns:
        all_regions (list of list of tuples): List of domain regions for each sequence, where each region is represented as a tuple (start, end).
    """
    # initialize list to store regions for all sequences
    all_regions = []

    # Define the structure for morphological operations.
    # Based on overall domain size expectations and noise reduction.
    # Smaller values preserve shorter domains, larger values reduce noise
    opening_structure = np.ones(
        30
    )  # Removes regions smaller than 30 residues (minimum viable domain)
    closing_structure = np.ones(
        15
    )  # Fills gaps smaller than 10 residues (allows for flexible boundaries)

    for seq_idx, seq_preds in enumerate(all_preds):
        # Ensure seq_preds is a numpy array
        preds_array = np.array(seq_preds)

        # 1. Apply binary opening to remove small, noisy positive predictions
        cleaned_preds = binary_opening(preds_array, structure=opening_structure)

        # 2. Apply binary closing to fill small gaps within domain regions
        final_preds = binary_closing(cleaned_preds, structure=closing_structure)

        # 3. Find contiguous regions of '1's in the final processed array
        regions = []
        if np.any(final_preds):
            # Find where regions of 1s start and end
            diff = np.diff(np.concatenate(([0], final_preds, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            # zip starts and ends to get region tuples
            for start, end in zip(starts, ends):
                region_length = end - start
                # Filter for regions of a minimum length for biological reasonableness
                if region_length >= 30:
                    regions.append((int(start), int(end)))

        # Append regions for this sequence to the overall list
        all_regions.append(regions)

        # Print debug info for first few sequences
        # if RANK == 0 and seq_idx < 3:
        #     print(
        #         f"Processing sequence {seq_idx}: Found {len(regions)} regions: {regions}"
        #     )

    # final print of total sequences with regions found
    if RANK == 0:
        print(f"Found {len(all_regions)} sequences with domain regions.")

    return all_regions


# -------------------------
# 8. MAIN USAGE
# -------------------------


def main(input_file):
    """
    Main function to run the DomainFinder script in usage mode.
    This function will be used to transform the users input data into a format suitable for the model.
    Args:
        input_file (str): Path to the input CSV file with sequences to predict on.
    Returns:
        all_regions (list of list of tuples): List of domain regions for each sequence, where each region is represented as a tuple (start, end).
    """

    # Call the loader function to prepare the dataset and model
    model, domain_boudnary_set_loader = usage_opener(ESM_MODEL, input_file)

    # Call the Predictor function to predict domain boundaries
    all_preds, sequence_metadata = Predictor(model, domain_boudnary_set_loader)

    # Search for domain regions with metadata
    all_regions = regions_search(all_preds)

    # Save all_regions to cache file for interrupted use cases
    output_file = "./tempUsage/predicted_domain_regions.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(all_regions, f)

    return all_regions


# -------------------------
# 9. Argument Parsing
# -------------------------


def parse_arguments():
    """
    Parse command-line arguments for the Domain Finder script. Used to switch between trainer and usage modes, and specify input/output files and model names.
    Returns:
        args: Parsed command-line arguments.
    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Domain Finder Script")

    # Input argument for Usage mode only
    parser.add_argument("--input", type=str, required=False, help="Input file path")
    # Output argument for Usage mode only
    parser.add_argument(
        "--output",
        type=str,
        default="./tempUsage/predicted_domain_regions.pkl",
        help="Output file path",
    )
    # Embedding model argument, not tested
    parser.add_argument("--model", type=str, required=False, help="ESM model name")

    # TrainerMode argument to switch between modes of training or usage
    parser.add_argument(
        "--TrainerMode",
        type=str,
        default="False",
        help="Set to True to run the trainer, False to run the predictor",
    )
    return parser.parse_args()


#############################################################################################################

# Parse arguments at the beginning
args = parse_arguments()

# Convert TrainerMode string to boolean
TRAINER_MODE = args.TrainerMode.lower() == "true"
# whether to use users parsed input file for usage mode or default global for training mode
if args.input:
    input_file = args.input
else:
    input_file = CSV_PATH

if __name__ == "__main__":
    # Either enter trainer or usage mode based on arguments in command line
    if TRAINER_MODE is True:
        main_trainer(Final_training=False)
    else:
        main(input_file)

    if RANK == 0:
        print("\nFinished running DomainFinder\n")
    dist.barrier()  # Ensure all processes reach this point before exiting
    dist.destroy_process_group()  # Clean up the process group
    quit()  # exit the script
