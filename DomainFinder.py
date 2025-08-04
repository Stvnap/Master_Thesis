import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import math
import pickle
import optuna
import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score
from ESM_Embeddings_HP_search import ESMDataset, LitClassifier
from ESM_Embeddings_HP_search import objective,load_best_model

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

NUM_CLASSES = 2 # Number of classes for domain detection, set to 1 for binary classification
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2

EMB_BATCH = 1
NUM_WORKERS_EMB = 32
# print(f"Using {NUM_WORKERS_EMB} workers for embedding generation")
BATCH_SIZE = 8
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
        input_dim,      # 1280 for ESM2-t33
        hidden_dims,    
        num_classes,   
        num_heads,     
        dropout,
        activation,
        kernel_init,
        num_layers
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
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        
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
            batch_first=True  # Much easier to work withs
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Per-position domain boundary classifier
        self.classifier = nn.Sequential(
            nn.Linear(working_dim, working_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(working_dim // 2, num_classes)
        )

    def generate_positional_encoding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension [1, seq_len, d_model]

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
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
    os.makedirs("pickle", exist_ok=True)
    os.makedirs(f"logs/{PROJECT_NAME}", exist_ok=True)  
    os.makedirs("models", exist_ok=True)  

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            train_embeddings, train_labels, val_embeddings, val_labels = pickle.load(f)
        if RANK == 0:
            print("Loaded cached embeddings & labels from disk.")
            print("Train embedding shape:", train_embeddings.shape)  # <-- Add this line
            print("Val embedding shape:", val_embeddings.shape)      # <-- Add this line
            
            print(train_labels.shape, val_labels.shape)


        train_ds = TensorDataset(train_embeddings, train_labels)
        val_ds = TensorDataset(val_embeddings, val_labels)

    else:
        esm_data = ESMDataset(FSDP_used=False,domain_boundary_detection=True,num_classes=NUM_CLASSES,esm_model=ESM_MODEL,
                              csv_path=CSV_PATH, category_col=CATEGORY_COL, sequence_col=SEQUENCE_COL)

        train_embeddings = esm_data.train_embeddings
        train_labels = esm_data.train_labels
        val_embeddings = esm_data.val_embeddings
        val_labels = esm_data.val_labels

        train_ds = TensorDataset(train_embeddings, train_labels)
        val_ds = TensorDataset(val_embeddings, val_labels)

        with open(CACHE_PATH, "wb") as f:
            pickle.dump(
                (train_embeddings, train_labels, val_embeddings, val_labels),
                f,
            )
        if RANK == 0:
            print("Computed embeddings & labels, then wrote them to cache.")


    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_EMB, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_EMB, pin_memory=True
    )


    train_labels_flat = train_labels.view(-1)  
    counts = torch.bincount(train_labels_flat, minlength=NUM_CLASSES).float()
    total = train_labels_flat.size(0)
    weights = total / (NUM_CLASSES * counts)
    weights = weights * (NUM_CLASSES / weights.sum())
    weights = weights.to(DEVICE)
    print(f"Class weights: {weights}")


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
        return objective(trial, train_embeddings, train_loader, val_loader, weights=weights,domain_task=True,EPOCHS=EPOCHS)
    study.optimize(objective_wrapper, n_trials=STUDY_N_TRIALS,)

    if RANK == 0:
        print("Best trial number:", study.best_trial.number)
        print("Best trial:", study.best_trial)
        # print("Best trial params:", study.best_trial.params)

    best_trial = study.best_trial


    lit_model = load_best_model(best_trial, train_embeddings, weights=None,domain_task=True)

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
                sequence_level_metrics.append({
                    'pred_boundaries': len(boundary_positions_pred),
                    'true_boundaries': len(boundary_positions_true),
                    'correct_boundaries': len(np.intersect1d(boundary_positions_pred, boundary_positions_true))
                })
                
                # Flatten for overall metrics
                all_preds.extend(seq_preds)
                all_true.extend(seq_labels)

    
    # Boundary-specific metrics
    boundary_prec = precision_score(all_true, all_preds, pos_label=1, zero_division=0)
    boundary_rec = recall_score(all_true, all_preds, pos_label=1, zero_division=0)
    
    # Sequence-level metrics
    total_sequences = len(sequence_level_metrics)
    sequences_with_correct_boundaries = sum(1 for m in sequence_level_metrics if m['correct_boundaries'] > 0)
    avg_boundary_detection_rate = np.mean([m['correct_boundaries'] / max(m['true_boundaries'], 1) for m in sequence_level_metrics])
    
    if RANK == 0:
        print(f"\n=== Boundary-Specific Metrics ===")
        print(f"Boundary Precision: {boundary_prec:.4f}")
        print(f"Boundary Recall: {boundary_rec:.4f}")
        print(f"\n=== Sequence-Level Metrics ===")
        print(f"Sequences with detected boundaries: {sequences_with_correct_boundaries}/{total_sequences} ({sequences_with_correct_boundaries/total_sequences:.2%})")
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