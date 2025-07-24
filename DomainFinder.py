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

NUM_LAYERS = 6
NUM_HEADS = 8


NUM_CLASSES = 1 # Number of classes for domain detection, set to 1 for binary classification
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2

EMB_BATCH = 1
NUM_WORKERS_EMB = max(16, os.cpu_count())
print(f"Using {NUM_WORKERS_EMB} workers for embedding generation")
BATCH_SIZE = 64
LR = 1e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 150
STUDY_N_TRIALS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# -------------------------
# 2. Transformer Model
# -------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_classes,
        num_heads,
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

        model_dims = {
            "esm2_t6_8M_UR50D": 320,
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
            "esm2_t48_15B_UR50D": 5120,
        }
        expected_dim = model_dims.get(ESM_MODEL, None) 
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # assert hidden_dims % num_heads == 0, "Hidden dim must be divisible by num heads"
        
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, num_classes)
        )


        self.encoder = nn.TransformerEncoder(
            self.encoder_layer(input_dim, hidden_dims, num_heads, dropout, activation),
            num_layers=NUM_LAYERS)
        
        
        # self.positional_encoding(input_dim, hidden_dims)
        


    # padding tokens are ignored, not needed if batch size is 1
    def padding_mask(self,seq, pad_token=0):
        mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
        return mask

    # POSITIONAL ENCODING:
    # add a order so the model knows how to read it
    def generate_positional_encoding(self,seq_len, d_model,device):
        '''
        Standard pe found in most transformer papers.
        It uses sine and cosine functions of different frequencies.
        '''
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)                            # create a column vector of positions, shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # early positions have higher frequencies, later positions have lower frequencies
        pe = torch.zeros(seq_len, d_model, device=device) # empty tensor for positional encoding

        pe[:, 0::2] = torch.sin(position * div_term)  # apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cosine to odd indices

        pe = pe.unsqueeze(1)  # add batch dimension, so shape becomes (seq_len, 1, d_model)

        return pe


    # ENCODER:
    # the part that encodes the input sequence into a fixed-size representation
    def encoder_layer(self,input_dim, hidden_dims, num_heads, dropout, activation):
        return nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dims,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x):
        seq_len, batch_size, input_dim = x.size()
        pe = self.generate_positional_encoding(seq_len, input_dim, x.device)
        x = x + pe  # add positional encoding to input embeddings

        x = nn.Dropout(self.dropout)(x)  # apply dropout to input embeddings

        print("Input shape:", x.shape)      # before encoder
        x = self.encoder(x)
        print("Encoder output:", x.shape)
        logits = self.classifier(x)
        print("Classifier output:", logits.shape)

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
        return objective(trial, train_embeddings, train_loader, val_loader, weights=None,domain_task=True)
    study.optimize(objective_wrapper, n_trials=STUDY_N_TRIALS,)

    if RANK == 0:
        print("Best trial number:", study.best_trial.number)
        print("Best trial:", study.best_trial)
        # print("Best trial params:", study.best_trial.params)

    best_trial = study.best_trial


    lit_model = load_best_model(best_trial, train_embeddings, weights)

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