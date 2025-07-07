import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import pickle
from ESM_Embeddings_HP_search import ESMDataset


# -------------------------
# 1. Global settings
# -------------------------

CSV_PATH = "./Dataframes/v3/FoundEntriesSwissProteins.csv"
CATEGORY_COL = "Pfam_id"
SEQUENCE_COL = "Sequence"
CACHE_PATH = "./pickle/FoundEntriesSwissProteins_100d.pkl"
PROJECT_NAME = "Optuna_100d_uncut_t33"

ESM_MODEL = "esm2_t33_650M_UR50D"


NUM_CLASSES = 101  # 100 Pfam classes + 1 for "other" category
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
# 2. MAIN
# -------------------------

def main(Final_training=False):
    os.makedirs("pickle", exist_ok=True)
    os.makedirs(f"logs/{PROJECT_NAME}", exist_ok=True)  
    os.makedirs("models", exist_ok=True)  

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            train_embeddings, train_labels, val_embeddings, val_labels = pickle.load(f)
        print("Loaded cached embeddings & labels from disk.")

        train_ds = TensorDataset(train_embeddings, train_labels)
        val_ds = TensorDataset(val_embeddings, val_labels)

    else:
        esm_data = ESMDataset(FSDP_used=False,domain_boundary_detection=True,num_classes=NUM_CLASSES,esm_model=ESM_MODEL, csv_path=CSV_PATH, category_col=CATEGORY_COL, sequence_col=SEQUENCE_COL)

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
        print("Computed embeddings & labels, then wrote them to cache.")

    counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    total = train_labels.size(0)
    weights = total / (NUM_CLASSES * counts)
    weights = weights * (NUM_CLASSES / weights.sum())
    weights = weights.to(DEVICE)

    print("Dataset building complete")


    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_EMB, pin_memory=True
    )
    print("Train loader created")

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_EMB, pin_memory=True
    )
    print("Val loader created")


if __name__ == "__main__":
    main(Final_training=False)