import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import math
import pickle
import optuna
import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
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
PROJECT_NAME = "Optuna_100d_uncut_t33_domains_boundary"

ESM_MODEL = "esm2_t33_650M_UR50D"

NUM_LAYERS = 8
NUM_HEADS = 8


NUM_CLASSES = 2 # Number of classes for domain detection, set to 1 for binary classification
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2

EMB_BATCH = 1
NUM_WORKERS_EMB = 1
print(f"Using {NUM_WORKERS_EMB} workers for embedding generation")
BATCH_SIZE = 8
LR = 1e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 50
STUDY_N_TRIALS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# -------------------------
# 2. Transformer Model
# -------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,      # 1280 for ESM2-t33
        hidden_dims,    # e.g., 512
        num_classes,    # 1 for binary domain/non-domain
        num_heads,      # 8
        dropout,
        activation,
        kernel_init,
    ):
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.kernel_init = kernel_init
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

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
            batch_first=True  # Much easier to work with
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )
        
        # Per-position domain boundary classifier
        self.classifier = nn.Sequential(
            nn.Linear(working_dim, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims // 2, num_classes)
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


    lit_model = load_best_model(best_trial, train_embeddings, weights=None)

    lit_model.eval()
    if RANK == 0:
        print("Best model loaded and set to eval mode")

    # Evaluate on validation set
    device = lit_model.device
    with torch.no_grad():
        val_embeddings_gpu = val_embeddings.to(device)
        # Permute to [seq_len, batch_size, input_dim]
        val_embeddings_gpu = val_embeddings_gpu.permute(1, 0, 2)
        val_logits = lit_model(val_embeddings_gpu)
    val_preds = val_logits.argmax(dim=1).cpu().numpy()
    val_true = val_labels.numpy()

    acc = accuracy_score(val_true, val_preds)
    if RANK == 0:
        print(f"Validation Accuracy: {acc:.4f}")

    if Final_training is True:
        early_stop = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", verbose=True
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
            mixed_precision="16-mixed",
            callbacks=[early_stop, checkpoint_callback],
            logger=TensorBoardLogger(
                save_dir=f"./logs/{PROJECT_NAME}", name=PROJECT_NAME
            ),
        )

        trainer.fit(lit_model, train_loader, val_loader)

        # save the final model
        final_model_path = f"./models/{PROJECT_NAME}.pt"
        torch.save(lit_model, final_model_path)

#############################################################################################################

if __name__ == "__main__":
    main(Final_training=False)