import os
import pickle
import glob

import esm
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics import Precision, Recall

torch.set_float32_matmul_precision("medium")

# -------------------------
# 1. GLobal settings
# -------------------------
CSV_PATH = "./Dataframes/DataTrainSwissPro_esm_shuffled.csv"
CATEGORY_COL = "categories"
SEQUENCE_COL = "Sequences"
CACHE_PATH = "./pickle/SwissProtTrainEmbeddings.pkl"


TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15

NUM_CLASSES = 3
BATCH_SIZE = 128
EMB_BATCH = 4
LR = 1e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 50

# DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# 2. FFW classifier head
# -------------------------
class FFNClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_classes,
        dropout,
        activation,
        kernel_init,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hdim),          # Linear layer
                nn.BatchNorm1d(hdim),               # Batch normalization
                activation,                         # ReLU activation
                nn.Dropout(dropout),                # Dropout layer
            ]
            prev_dim = hdim
        layers.append(
            nn.Linear(prev_dim, num_classes)
        )                                           # Final linear layer for classification
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                kernel_init(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# -------------------------
# 3. PyTorch-Lightning module
# -------------------------


class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_classes,
        optimizer_class,
        activation,
        kernel_init,
        lr,
        weight_decay,
        dropout=0.3,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimicer_class = optimizer_class
        self.lr = lr
        self.weight_decay = weight_decay

        # base model
        self.model = FFNClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            activation=activation,
            kernel_init=kernel_init,
        )

        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.precision_metric = Precision(
            task="multiclass", num_classes=num_classes, average=None, ignore_index=None
        )
        self.recall_metric = Recall(
            task="multiclass", num_classes=num_classes, average=None, ignore_index=None
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = (preds == y).float().mean()

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

        self.precision_metric(preds, y)
        self.recall_metric(preds, y)

    def on_validation_epoch_end(self):
        prec_all = self.precision_metric.compute()
        rec_all = self.recall_metric.compute()

        val_prec_1 = prec_all[1].item()
        val_rec_1 = rec_all[1].item()
        val_prec_2 = prec_all[2].item()
        val_rec_2 = rec_all[2].item()

        self.log("val_prec_1", val_prec_1, prog_bar=True, sync_dist=True)
        self.log("val_rec_1", val_rec_1, prog_bar=True, sync_dist=True)
        self.log("val_prec_2", val_prec_2, prog_bar=True, sync_dist=True)
        self.log("val_rec_2", val_rec_2, prog_bar=True, sync_dist=True)

        self.precision_metric.reset()
        self.recall_metric.reset()

    def configure_optimizers(self):
        if self.optimicer_class == torch.optim.SGD:
            return self.optimicer_class(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            return self.optimicer_class(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )


# -------------------------
# 4. Dataset & embedding
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return f"seq{idx}", self.seqs[idx]


class ESMDataset:
    def __init__(self):
        def map_label(cat):
            return {i: i + 1 for i in range(NUM_CLASSES)}.get(cat, 0)

        df = pd.read_csv(CSV_PATH)
        df["label"] = df[CATEGORY_COL].apply(map_label)
        df.drop(columns=[CATEGORY_COL, "ID"], inplace=True)

        print("Data loaded")

        self.train_df, tmp = train_test_split(
            df, train_size=TRAIN_FRAC, stratify=df["label"], random_state=42
        )
        val_frac = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
        self.val_df, self.test_df = train_test_split(
            tmp, train_size=val_frac, stratify=tmp["label"], random_state=42
        )

        print("Train/val/test splits created")

        self.model, self.alphabet, self.batch_converter = self.esm_loader()

        self.train_embeddings = self._embed(self.train_df[SEQUENCE_COL].tolist()).cpu()
        print("Train embeddings computed")
        self.val_embeddings = self._embed(self.val_df[SEQUENCE_COL].tolist()).cpu()
        print("Validation embeddings computed")

        self.train_labels = torch.tensor(
            self.train_df["label"].values, dtype=torch.long
        )
        self.val_labels = torch.tensor(self.val_df["label"].values, dtype=torch.long)

        print("Labels prepared")

    def esm_loader(self):
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.to(DEVICE).eval()
        return model, alphabet, alphabet.get_batch_converter()

    def _embed(self, seqs):
        """
        Tokenize & run ESM in minibatches.
        The pooling below is done in one vectorized step per batch,
        """
        loader = DataLoader(
            SeqDataset(seqs),
            batch_size=EMB_BATCH,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.batch_converter,
            drop_last=False,
        )
        all_vecs = []
        total_batches = len(loader)
        for idx, (_, _, batch_tokens) in enumerate(
            loader
        ):  # loop based on chatGPT code
            if idx % 100 == 0 or idx == total_batches - 1:
                print(f"Embedding batch {idx + 1}/{total_batches}")

            # print(batch_tokens[0])

            batch_tokens = batch_tokens.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                results = self.model(
                    batch_tokens, repr_layers=[33], return_contacts=False
                )
            token_repr = results["representations"][33]

            mask = batch_tokens != self.alphabet.padding_idx

            lengths = mask.sum(dim=1).unsqueeze(-1)

            masked_repr = token_repr * mask.unsqueeze(-1)

            pooled = masked_repr.sum(dim=1) / lengths

            all_vecs.append(pooled)

        return torch.cat(all_vecs, dim=0)


# -------------------------
# 5. Main
# -------------------------
def main(Final_training=False):
    os.makedirs("pickle", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            train_embeddings, train_labels, val_embeddings, val_labels = pickle.load(f)
        print("Loaded cached embeddings & labels from disk.")

        train_ds = TensorDataset(train_embeddings, train_labels)
        val_ds = TensorDataset(val_embeddings, val_labels)

    else:
        esm_data = ESMDataset()

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
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )
    print("Train loader created")

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True
    )
    print("Val loader created")

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")



    def objective(trial):
        n_neurons = trial.suggest_int("num_neurons", 64, 512, step=64)
        hidden_dims = [
            n_neurons for _ in range(trial.suggest_int("num_hidden_layers", 1, 10))
        ]

        drop = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        if optimizer == "adam":
            optimizer_class = torch.optim.Adam
        elif optimizer == "adamw":
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.SGD

        activation = trial.suggest_categorical(
            "activation", ["relu", "gelu", "leaky_relu"]
        )
        if activation == "relu":
            activation = nn.ReLU(inplace=True)
            kernel_init = nn.init.kaiming_normal_
        elif activation == "gelu":
            activation = nn.GELU()
            kernel_init = nn.init.xavier_normal_
        else:
            activation = nn.LeakyReLU(inplace=True)
            kernel_init = nn.init.kaiming_normal_

        print(
            f"\n,\n,\n,Trial {trial.number}: hidden_dims={len(hidden_dims)}, dropout={drop}, lr={lr}, wd={wd}, optimizer={optimizer}, activation={activation}\n,\n,\n,"
        )

        model = LitClassifier(
            input_dim=train_embeddings.size(1),
            hidden_dims=hidden_dims,
            num_classes=NUM_CLASSES,
            optimizer_class=optimizer_class,
            activation=activation,
            kernel_init=kernel_init,
            lr=lr,
            weight_decay=wd,
            dropout=drop,
            class_weights=weights,
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=1, mode="min", verbose=True
        )
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=1,
            # strategy="ddp",
            enable_progress_bar=True,
            callbacks=[early_stop, checkpoint_callback],
            logger=TensorBoardLogger(
                save_dir="./logs/optuna", name=f"optuna_trial_{trial.number}"
            ),
        )
        # print(model)

        trainer.fit(model, train_loader, val_loader)
        return checkpoint_callback.best_model_score.item()

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///logs/optuna/optuna_study.db",
        load_if_exists=True,
        study_name="esm_2d_hp_search",
    )
    study.optimize(objective, n_trials=300)


    print("Best trial number:", study.best_trial.number)
    print("Best trial:", study.best_trial)
    # print("Best trial params:", study.best_trial.params)


    best_trial = study.best_trial


    def load_best_model(trial):
        p = trial.params 

        n_neurons   = p["num_neurons"]
        n_layers    = p["num_hidden_layers"]
        hidden_dims = [n_neurons] * n_layers

        opt_name = p["optimizer"]
        if opt_name == "adam":
            optimizer_class = torch.optim.Adam
        elif opt_name == "adamw":
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.SGD

        act_name = p["activation"]
        if act_name == "relu":
            activation_fn = nn.ReLU(inplace=True)
            kernel_init   = nn.init.kaiming_normal_
        elif act_name == "gelu":
            activation_fn = nn.GELU()
            kernel_init   = nn.init.xavier_normal_
        else:  # "leaky_relu"
            activation_fn = nn.LeakyReLU(inplace=True)
            kernel_init   = nn.init.kaiming_normal_

        drop = p["dropout"]
        lr   = p["lr"]
        wd   = p["weight_decay"]

        model = LitClassifier(
            input_dim=train_embeddings.size(1),
            hidden_dims=hidden_dims,
            num_classes=NUM_CLASSES,
            optimizer_class=optimizer_class,
            activation=activation_fn,
            kernel_init=kernel_init,
            lr=lr,
            weight_decay=wd,
            dropout=drop,
            class_weights=weights,
        )


        ckpt_dir = f"./logs/optuna/optuna_trial_{trial.number}/version_0/checkpoints/"
        print(ckpt_dir)
        pattern  = os.path.join(ckpt_dir, "*.ckpt")
        matches = glob.glob(pattern)
        chosen_ckpt = matches[0]

        ckpt_path = chosen_ckpt
        print(ckpt_dir)
        model = LitClassifier.load_from_checkpoint(ckpt_path)


        torch.save(model,'./models/optuna_bestmodel.pt')
        model = model.to(DEVICE).eval()
        
        return model



    lit_model=load_best_model(best_trial)



    lit_model.eval()
    print(lit_model)

    print("First 5 embeddings:\n", val_embeddings[:5])
    print("First 5 labels:    \n", val_labels[:5].cpu().numpy())



    device = lit_model.device
    with torch.no_grad():
        val_embeddings = val_embeddings.to(device)
        val_logits     = lit_model(val_embeddings)
    val_preds = val_logits.argmax(dim=1).cpu().numpy()
    val_true  = val_labels.numpy()

    print(val_preds[:30])
    print(val_true[:30])

    if getattr(lit_model, "global_rank", 0) == 0:
        acc  = accuracy_score(val_true, val_preds)
        prec = precision_score(val_true, val_preds, labels=[1, 2], average=None)
        rec  = recall_score(   val_true, val_preds, labels=[1, 2], average=None)

        print(f"\nFinal Validation Accuracy: {acc:.4f}")
        print(f"Precision (class 1): {prec[0]:.4f}")
        print(f"Precision (class 2): {prec[1]:.4f}")
        print(f"Recall    (class 1): {rec[0]:.4f}")
        print(f"Recall    (class 2): {rec[1]:.4f}")

    sd = lit_model.state_dict()


    # Print a single parameter (e.g. the first layer’s weights)
    first_key = next(iter(sd.keys()))
    print("First key:", first_key)
    print("First-weight tensor:\n", sd[first_key])



    if Final_training is True:

        early_stop = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", verbose=True
        )
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")


        print("Lit model created")
        trainer = pl.Trainer(
            max_epochs=200,
            accelerator="gpu",
            devices=-1,
            # strategy="ddp",
            enable_progress_bar=True,
            callbacks=[early_stop, checkpoint_callback],
            logger=TensorBoardLogger(
                save_dir="./logs/optuna", name="optuna_trial_Final"
            ),
        )

        print("Trainer created")

        trainer.fit(lit_model, train_loader, val_loader)

#############################################################################################################

if __name__ == "__main__":
    main(Final_training=True)  
