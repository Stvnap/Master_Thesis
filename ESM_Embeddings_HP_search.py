# -------------------------
# 1. Imports & basic setup
# -------------------------
import glob
import os
import pickle

import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Precision, Recall

from ESM_Embedder import DEVICE, RANK, ESMDataset

os.environ["NCCL_P2P_DISABLE"] = "1"

torch.set_float32_matmul_precision("high")  # More stable than "high"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

pd.set_option("display.max_rows", None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

# -------------------------
# 1. GLOBALS
# -------------------------

CSV_PATH = "./Dataframes/v3/FoundEntriesSwissProteins.csv"
CATEGORY_COL = "Pfam_id"
SEQUENCE_COL = "Sequence"
CACHE_PATH = "./pickle/FoundEntriesALLProteins_1000d_.pkl"
PROJECT_NAME = "Optuna_1000d_uncut_t33_ALL"

ESM_MODEL = "esm2_t33_650M_UR50D"

NUM_CLASSES = 1001  # classes + 1 for "other" class
EPOCHS = 1
STUDY_N_TRIALS = 0
BATCH_SIZE = 64
NUM_WORKERS_EMB = max(16, os.cpu_count())


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
                nn.Linear(prev_dim, hdim),  # Linear layer
                nn.BatchNorm1d(hdim),  # Batch normalization
                activation,  # activation
                nn.Dropout(dropout),  # Dropout layer
            ]
            prev_dim = hdim
        layers.append(
            nn.Linear(prev_dim, num_classes)
        )  # Final linear layer for classification
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
        domain_task=False,
        num_heads=8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer_class = optimizer_class
        self.lr = lr
        self.weight_decay = weight_decay
        self.domain_task = domain_task

        # base model
        if domain_task is False:
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

        else:
            from DomainFinder import Transformer

            self.model = Transformer(
                input_dim,
                hidden_dims,
                num_classes,
                num_heads,
                dropout,
                activation,
                kernel_init,
            )

            self.loss_fn = nn.BCEWithLogitsLoss(
                reduction="none"
            )  # for domain boundary detection

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

        # Addition to check for NaN values
        assert not torch.isnan(x).any(), "Input contains NaN values"
        logits = self(x)
        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        try:
            loss = self.loss_fn(logits.view(-1, 2), y.view(-1))         # for domain boundary detection
        except:
            loss = self.loss_fn(logits, y)
        # print(loss)
        assert not torch.isnan(loss), "Validation loss is NaN"
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # Addition to check for NaN values in validation step
        assert not torch.isnan(x).any(), "Input contains NaN values"
        logits = self(x)
        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        try:
            loss = self.loss_fn(logits.view(-1, 2), y.view(-1))         # for domain boundary detection
        except:
            loss = self.loss_fn(logits, y)
        # print(loss)
        assert not torch.isnan(loss), "Validation loss is NaN"
        preds = torch.argmax(logits, dim=1)

        if self.domain_task is True:  # for domain boundary detection
            preds = preds.view(-1)  # Flatten predictions
            y = y.view(-1)  # Flatten labels


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

        # Log metrics for each class individually
        for class_idx in range(NUM_CLASSES):
            if class_idx < len(prec_all):
                val_prec = (
                    prec_all[class_idx].item()
                    if not torch.isnan(prec_all[class_idx])
                    else 0.0
                )
                self.log(
                    f"val_prec_{class_idx}", val_prec, prog_bar=False, sync_dist=True
                )

            if class_idx < len(rec_all):
                val_rec = (
                    rec_all[class_idx].item()
                    if not torch.isnan(rec_all[class_idx])
                    else 0.0
                )
                self.log(
                    f"val_rec_{class_idx}", val_rec, prog_bar=False, sync_dist=True
                )

        # Log average metrics across all classes
        avg_prec = prec_all.nanmean().item() if len(prec_all) > 0 else 0.0
        avg_rec = rec_all.nanmean().item() if len(rec_all) > 0 else 0.0

        self.log("val_prec_avg", avg_prec, prog_bar=True, sync_dist=True)
        self.log("val_rec_avg", avg_rec, prog_bar=True, sync_dist=True)

        self.precision_metric.reset()
        self.recall_metric.reset()

    def configure_optimizers(self):
        if self.optimizer_class == torch.optim.SGD:
            return self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True,  # Nesterov momentum
            )
        else:
            return self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )


# -------------------------
# 5. Optuna objective func, load best model func
# -------------------------


def objective(
    trial, train_embeddings, train_loader, val_loader, weights, domain_task=False
):
    if domain_task is False:
        if RANK == 0:
            n_neurons = trial.suggest_int("num_neurons", 64, 512, step=64)
            hidden_dims = [
                n_neurons for _ in range(trial.suggest_int("num_hidden_layers", 1, 4))
            ]

            drop = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # was 1e-2
            wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)  # was 1e-2

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
                f"\n=== Trial {trial.number} ===\n"
                f"Hidden layers: {len(hidden_dims)}, Neurons: {n_neurons}\n"
                f"Dropout: {drop:.3f}, LR: {lr:.6f}, WD: {wd:.6f}\n"
                f"Optimizer: {optimizer}, Activation: {activation.__class__.__name__}\n"
                f"========================\n"
            )

            params = {
                "hidden_dims": hidden_dims,
                "dropout": drop,
                "lr": lr,
                "weight_decay": wd,
                "optimizer_class": optimizer_class,
                "activation": activation,
                "kernel_init": kernel_init,
                "drop": drop,
                "weights": weights,
            }

        else:
            params = None

        params_list = [params]
        dist.broadcast_object_list(params_list, src=0)
        params = params_list[0]

        model = LitClassifier(
            input_dim=train_embeddings.size(1),
            hidden_dims=params["hidden_dims"],
            num_classes=NUM_CLASSES,
            optimizer_class=params["optimizer_class"],
            activation=params["activation"],
            kernel_init=params["kernel_init"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            dropout=params["drop"],
            class_weights=params["weights"],
            domain_task=False,  # Set to True if using domain boundary detection
        )

        # print(model)

    else:
        # Domain boundary detection task, HPs for Transformer model
        if RANK == 0:

            d_model = 8
            n_heads = trial.suggest_int("num_heads", 2, 8, step=2)

            n_layers = trial.suggest_int("num_hidden_layers", 6, 12, 24)
            d_ff = 4 * d_model
            max_seq_len = trial.suggest_int("max_seq_len", 100, 1000, step=100)

            drop = trial.suggest_float("dropout", 0.1, 0.5)
            drop_attn = trial.suggest_float("dropout_attn", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-5, 1e2, log=True)
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

            params = {
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "d_ff": d_ff,
                "max_seq_len": max_seq_len,
                "drop": drop,
                "drop_attn": drop_attn,
                "lr": lr,
                "weight_decay": wd,
                "optimizer_class": optimizer_class,
                "activation": activation,
                "kernel_init": kernel_init,
            }

        else:
            params = None

        params_list = [params]
        dist.broadcast_object_list(params_list, src=0)
        params = params_list[0]

        model = LitClassifier(
            input_dim=train_embeddings.size(1),
            hidden_dims=params["d_model"],
            num_classes=NUM_CLASSES,
            optimizer_class=params["optimizer_class"],
            activation=params["activation"],
            kernel_init=params["kernel_init"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            dropout=params["drop"],
            class_weights=weights,
            domain_task=True,  # Set to True if using domain boundary detection
            num_heads=params["n_heads"],
        )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        enable_progress_bar=True,
        callbacks=[early_stop, checkpoint_callback],
        logger=TensorBoardLogger(
            save_dir=f"./logs/{PROJECT_NAME}", name=f"optuna_trial_{trial.number}"
        ),
    )

    trainer.fit(model, train_loader, val_loader)
    # Ensure logger is flushed
    logger = trainer.logger
    if hasattr(logger, "finalize"):
        logger.finalize("success")
    return checkpoint_callback.best_model_score.item()


def load_best_model(trial, train_embeddings, weights):
    p = trial.params

    n_neurons = p["num_neurons"]
    n_layers = p["num_hidden_layers"]
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
        kernel_init = nn.init.kaiming_normal_
    elif act_name == "gelu":
        activation_fn = nn.GELU()
        kernel_init = nn.init.xavier_normal_
    else:  # "leaky_relu"
        activation_fn = nn.LeakyReLU(inplace=True)
        kernel_init = nn.init.kaiming_normal_

    drop = p["dropout"]
    lr = p["lr"]
    wd = p["weight_decay"]

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

    ckpt_dir = (
        f"./logs/{PROJECT_NAME}/optuna_trial_{trial.number}/version_0/checkpoints/"
    )
    print(f"Looking for checkpoints in: {ckpt_dir}")
    pattern = os.path.join(ckpt_dir, "*.ckpt")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    # Choose the best checkpoint (most recent)
    chosen_ckpt = max(matches, key=os.path.getctime)
    print(f"Loading checkpoint: {chosen_ckpt}")

    # Load from checkpoint (this automatically restores all parameters)
    model = LitClassifier.load_from_checkpoint(chosen_ckpt)

    torch.save(model, f"./models/{PROJECT_NAME}.pt")
    model = model.to(DEVICE).eval()

    return model


# -------------------------
# 6. Main
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
        esm_data = ESMDataset(
            esm_model=ESM_MODEL,
            FSDP_used=False,
            domain_boundary_detection=False,
            training=True,
            num_classes=NUM_CLASSES,
            csv_path=CSV_PATH,
            category_col=CATEGORY_COL,
            sequence_col=SEQUENCE_COL,
        )

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

    # dist.destroy_process_group()                # temporary fix to move on on one GPU for training

    counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    total = train_labels.size(0)
    weights = total / (NUM_CLASSES * counts)
    weights = weights * (NUM_CLASSES / weights.sum())
    weights = weights.to(DEVICE)

    print("Dataset building complete")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS_EMB,
        pin_memory=True,
    )
    print("Train loader created")

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_EMB,
        pin_memory=True,
    )
    print("Val loader created")

    print("Starting Optuna hyperparameter search...")

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
        return objective(trial, train_embeddings, train_loader, val_loader, weights)

    study.optimize(
        objective_wrapper,
        n_trials=STUDY_N_TRIALS,
    )

    print("Best trial number:", study.best_trial.number)
    print("Best trial:", study.best_trial)
    # print("Best trial params:", study.best_trial.params)

    best_trial = study.best_trial

    lit_model = load_best_model(best_trial, train_embeddings, weights)

    lit_model.eval()
    print("Best model loaded and set to eval mode")

    # Evaluate on validation set
    device = lit_model.device
    with torch.no_grad():
        val_embeddings_gpu = val_embeddings.to(device)  # Use different variable name
        val_logits = lit_model(val_embeddings_gpu)
    val_preds = val_logits.argmax(dim=1).cpu().numpy()
    val_true = val_labels.numpy()

    acc = accuracy_score(val_true, val_preds)
    print(f"Validation Accuracy: {acc:.4f}")

    if Final_training is True:
        early_stop = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", verbose=True
        )
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

        print("Lit model created")
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=-1,
            strategy="ddp",
            enable_progress_bar=True,
            callbacks=[early_stop, checkpoint_callback],
            logger=TensorBoardLogger(
                save_dir=f"./logs/{PROJECT_NAME}", name=PROJECT_NAME
            ),
        )

        print("Trainer created")

        trainer.fit(lit_model, train_loader, val_loader)

        if RANK == 0:
            print("\nTraining complete, saving final model...\n")

        # save the final model
        final_model_path = f"./models/{PROJECT_NAME}.pt"
        torch.save(lit_model, final_model_path)

    if dist.is_initialized():
        # Ensure all processes are synchronized before exiting
        dist.barrier()
        if RANK == 0:
            print("Done! Exiting...")
        dist.destroy_process_group()


#############################################################################################################

if __name__ == "__main__":
    main(Final_training=True)
