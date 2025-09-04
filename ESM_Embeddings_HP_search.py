# -------------------------
# 1. Imports & basic setup
# -------------------------
import glob
import os

import h5py
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall

from ESM_Embedder import DEVICE, RANK, ESMDataset

torch.set_printoptions(threshold=float("inf"))  # Show full tensor

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
NUM_CLASSES = 1001  # classes + 1 for "other" class


CSV_PATH = "./Dataframes/v3/RemainingEntriesCompleteProteins.csv"
CATEGORY_COL = "Pfam_id"
SEQUENCE_COL = "Sequence"
CACHE_PATH = f"./temp/embeddings_classification_{NUM_CLASSES-1}d.h5"
PROJECT_NAME = f"t33_ALL_{NUM_CLASSES-1}d"

ESM_MODEL = "esm2_t33_650M_UR50D"

EPOCHS = 50
STUDY_N_TRIALS = 15
BATCH_SIZE = 256
NUM_WORKERS_EMB = min(16, os.cpu_count())

VAL_FRAC = 0.2
TRAIN_FRAC = 1 - VAL_FRAC


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
        num_layers=None,
        cosine_T_max=20,
        cosine_eta_min=1e-6,
        use_grad_clip=False,
        grad_clip_value=1.0,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        momentum=0.9,
        nesterov=True,  
        rmsprop_alpha=0.99,
        rmsprop_momentum=0.9,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer_class = optimizer_class
        self.lr = lr
        self.weight_decay = weight_decay
        self.domain_task = domain_task
        self.class_weights = class_weights
        self.num_classes = num_classes
        self.cosine_T_max = cosine_T_max
        self.cosine_eta_min = cosine_eta_min
        self.use_grad_clip = use_grad_clip
        self.grad_clip_value = grad_clip_value

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.momentum = momentum
        self.nesterov = nesterov
        self.rmsprop_alpha = rmsprop_alpha
        self.rmsprop_momentum = rmsprop_momentum

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
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
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
                num_layers,
            )


        self.precision_metric = Precision(
            task="multiclass", num_classes=num_classes, average=None, ignore_index=None
        )
        self.recall_metric = Recall(
            task="multiclass", num_classes=num_classes, average=None, ignore_index=None
        )


    def fbeta (self, precision, recall, beta=1.5):
        """
        Calculate the F-beta score.
        :param precision: Precision value
        :param recall: Recall value
        :param beta: Beta value for F-beta score
        :return: F-beta score
        """
        if precision + recall == 0:
            return 0.0
        return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    def differentiable_fbeta_loss(self, logits, labels, beta=1.5, epsilon=1e-8):
        """
        Differentiable F-beta loss using soft predictions.
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        if self.domain_task:
            # Binary classification - focus on boundary class (class 1)
            # Convert labels to one-hot
            labels_onehot = torch.zeros_like(probs)
            labels_onehot.scatter_(1, labels.unsqueeze(-1), 1)
            
            # Calculate soft TP, FP, FN for boundary class
            boundary_probs = probs[:, 1]  # Probabilities for boundary class
            boundary_labels = labels_onehot[:, 1]  # True boundary labels
            
            tp = (boundary_probs * boundary_labels).sum()
            fp = (boundary_probs * (1 - boundary_labels)).sum()
            fn = ((1 - boundary_probs) * boundary_labels).sum()
            
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            
            fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + epsilon)
            return 1.0 - fbeta  # Convert to loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # time_start = time.time()
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # Addition to check for NaN values
        assert not torch.isnan(x).any(), "Input contains NaN values"
        logits = self(x)
        assert not torch.isnan(logits).any(), "Logits contain NaN values"

        # print('SHAPES',logits.shape, y.shape)

        if self.domain_task is True:
            # Process each sequence individually
            batch_size, seq_len, num_classes = logits.shape
            losses = []

            for i in range(batch_size):
                seq_logits = logits[i]  # [seq_len, num_classes]
                seq_labels = y[i]  # [seq_len]
                seq_loss = self.differentiable_fbeta_loss(seq_logits, seq_labels)
                losses.append(seq_loss)

            loss = torch.stack(losses).mean()
        else:
            loss = self.loss_fn(logits, y)

        assert not torch.isnan(loss), "Training loss is NaN"
        self.log("train_loss", loss.detach(), prog_bar=True)
        # if RANK == 0:
        #     print(f"Training step time: {time.time() - time_start:.2f} seconds")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # Addition to check for NaN values in validation step
        assert not torch.isnan(x).any(), "Input contains NaN values"
        logits = self(x)
        assert not torch.isnan(logits).any(), "Logits contain NaN values"

        if self.domain_task is True:
            # Process each sequence individually
            batch_size, seq_len, num_classes = logits.shape
            losses = []
            all_preds = []
            all_labels = []

            for i in range(batch_size):
                seq_logits = logits[i]  # [seq_len, num_classes]
                seq_labels = y[i]  # [seq_len]
                seq_loss = self.differentiable_fbeta_loss(seq_logits, seq_labels)
                losses.append(seq_loss)

                seq_preds = torch.argmax(seq_logits, dim=-1)  # [seq_len]
                all_preds.append(seq_preds)
                all_labels.append(seq_labels)

            loss = torch.stack(losses).mean()
            preds = torch.cat(all_preds)  # Concatenate all sequence predictions
            y_flat = torch.cat(all_labels)  # Concatenate all sequence labels
        else:
            loss = self.loss_fn(logits, y)
            preds = torch.argmax(logits, dim=-1)
            y_flat = y

        assert not torch.isnan(loss), "Validation loss is NaN"

        # if RANK == 0:  # Only print for first batch to avoid spam
        # if batch_idx == 0:  # Only print for first batch to avoid spam
        #         # print(f"Batch {batch_idx}:")
        #   print(f"Preds: {preds}")
        # print(f"Labels: {y_flat}")
        #         print(f"Logits sample (first sequence, first 5 positions):\n{logits[0, :5, :]}")
        #         # print(f"Max logits: {logits.max()}, Min logits:\n{logits.min()}")
        #         # print(f"Class 0 logits mean: {logits[:, :, 0].mean()}")
        #         # print(f"Class 1 logits mean: {logits[:, :, 1].mean()}")
        #         # print(f"Loss function weights: {self.loss_fn.weight}")
        #         print(f"Prediction distribution: {torch.bincount(preds)}")
        #         print(f"True label distribution: {torch.bincount(y_flat)}")
        #         print("-" * 50)

        acc = (preds == y_flat).float().mean()

        self.log(
            "val_loss",
            loss.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            acc.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log("hp_metric", loss.detach(), on_step=False, on_epoch=True, sync_dist=True)

        self.precision_metric(preds.detach(), y_flat.detach())
        self.recall_metric(preds.detach(), y_flat.detach())

    def on_validation_epoch_end(self):
        prec_all = self.precision_metric.compute()
        rec_all = self.recall_metric.compute()

        # Log metrics for each class individually
        if self.domain_task is True:
            
            # if RANK == 0:
            #     print("Domain boundary detection task, only 2 classes")

            boundary_precision = (
                prec_all[1].item() if not torch.isnan(prec_all[1]) else 0.0
            )
            boundary_recall = rec_all[1].item() if not torch.isnan(rec_all[1]) else 0.0


            fbeta = self.fbeta(boundary_precision, boundary_recall, beta=1.5)

            self.log(
                "val_boundary_prec", boundary_precision, prog_bar=True, sync_dist=True
            )
            self.log("val_boundary_rec", boundary_recall, prog_bar=True, sync_dist=True)

            self.log("f_beta", fbeta, prog_bar=True, sync_dist=True)

        else:
            for class_idx in range(self.num_classes):
                if class_idx < len(prec_all):
                    val_prec = (
                        prec_all[class_idx].item()
                        if not torch.isnan(prec_all[class_idx])
                        else 1.0
                    )
                    self.log(
                        f"val_prec_{class_idx}",
                        val_prec,
                        prog_bar=False,
                        sync_dist=True,
                    )

                if class_idx < len(rec_all):
                    val_rec = (
                        rec_all[class_idx].item()
                        if not torch.isnan(rec_all[class_idx])
                        else 1.0
                    )
                    self.log(
                        f"val_rec_{class_idx}", val_rec, prog_bar=False, sync_dist=True
                    )


            # if RANK == 0:
            #     print(f"Precision per class: {prec_all}")
            #     print(f"Recall per class: {rec_all}")
            #     print(f"NaN count in precision: {torch.isnan(prec_all).sum()}")

            # Log average metrics across all classes
            valid_prec = prec_all[~torch.isnan(prec_all)]  # Remove NaN values
            avg_prec = valid_prec.mean().item() if len(valid_prec) > 0 else 0.0
            valid_rec = rec_all[~torch.isnan(rec_all)]  # Remove NaN values  
            avg_rec = valid_rec.mean().item() if len(valid_rec) > 0 else 0.0

            self.log("val_prec_avg", avg_prec, prog_bar=True, sync_dist=True)
            self.log("val_rec_avg", avg_rec, prog_bar=True, sync_dist=True)

        self.precision_metric.reset()
        self.recall_metric.reset()



    def configure_optimizers(self):
        # Create optimizer with tuned parameters
        if self.optimizer_class == torch.optim.SGD:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        elif self.optimizer_class in [torch.optim.Adam, torch.optim.AdamW, torch.optim.NAdam]:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.beta1, self.beta2),
                eps=self.eps,
            )
        elif self.optimizer_class == torch.optim.RMSprop:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                alpha=self.rmsprop_alpha,
                momentum=self.rmsprop_momentum,
            )
        else:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        
        # CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cosine_T_max,
            eta_min=self.cosine_eta_min
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }


# -------------------------
# 5. Optuna objective func, load best model func
# -------------------------


def objective(
    trial,
    input_dim,
    train_loader,
    val_loader,
    weights,
    domain_task=False,
    EPOCHS=EPOCHS,
):
    if domain_task is False:
        if RANK == 0:
            n_neurons = trial.suggest_int("num_neurons", 128, 1536, step=64)
            hidden_dims = [
                n_neurons for _ in range(trial.suggest_int("num_hidden_layers", 1, 2, step=1))
            ]

            drop = trial.suggest_float("dropout", 0.05, 0.5)
            lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)  
            wd = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)  

            cosine_T_max = trial.suggest_int("cosine_T_max", 10, 30)
            cosine_eta_min = trial.suggest_float("cosine_eta_min", 1e-7, 1e-5, log=True)

            use_grad_clip = trial.suggest_categorical("use_grad_clip", [True, False])
            grad_clip_value = trial.suggest_float("grad_clip_value", 0.5, 2.0)


            optimizer = trial.suggest_categorical(
                "optimizer", ["adam", "adamw", "sgd", "rmsprop", "nadam"]
            )
            
            # Tune optimizer-specific parameters
            if optimizer in ["adam", "adamw", "nadam"]:
                beta1 = trial.suggest_float("beta1", 0.85, 0.95)
                beta2 = trial.suggest_float("beta2", 0.99, 0.999)
                eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)
                
                if optimizer == "adam":
                    optimizer_class = torch.optim.Adam
                elif optimizer == "adamw":
                    optimizer_class = torch.optim.AdamW
                else:  # nadam
                    optimizer_class = torch.optim.NAdam
                    
            elif optimizer == "sgd":
                momentum = trial.suggest_float("momentum", 0.8, 0.95)
                nesterov = trial.suggest_categorical("nesterov", [True, False])
                optimizer_class = torch.optim.SGD
            else:  # rmsprop
                momentum = trial.suggest_float("rmsprop_momentum", 0.8, 0.95)
                alpha = trial.suggest_float("rmsprop_alpha", 0.9, 0.99)
                optimizer_class = torch.optim.RMSprop

            activation = trial.suggest_categorical(
                "activation", ["relu", "gelu", "leaky_relu"]
            )
            if activation == "relu":
                activation = nn.ReLU(inplace=True)
                kernel_init = nn.init.kaiming_normal_
            elif activation == "gelu":
                activation = nn.GELU()
                kernel_init = nn.init.xavier_normal_
            elif activation == "leaky_relu":
                activation = nn.LeakyReLU(inplace=True)
                kernel_init = nn.init.kaiming_normal_
            else:
                activation = nn.SiLU(inplace=True)  # SiLU is Swish
                kernel_init = nn.init.xavier_normal_

            print(
                f"\n=== Trial {trial.number} ===\n"
                f"Hidden layers: {len(hidden_dims)}, Neurons: {n_neurons}\n"
                f"Dropout: {drop:.3f}, LR: {lr:.6f}, WD: {wd:.6f}\n"
                f"Optimizer: {optimizer}, Activation: {activation.__class__.__name__}\n"
                f"Cosine T_max: {cosine_T_max}, Cosine eta_min: {cosine_eta_min}\n"
                f"Use grad clip: {use_grad_clip}, Grad clip value: {grad_clip_value}\n"
                f"Beta1: {locals().get('beta1', 0.9)}, Beta2: {locals().get('beta2', 0.999)}\n"
                f"Momentum: {locals().get('momentum', 0.9)}, Nesterov: {locals().get('nesterov', True)}\n"
                f"RMSprop alpha: {locals().get('alpha', 0.99)}, RMSprop momentum: {locals().get('rmsprop_momentum', 0.9)}\n"
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
                "cosine_T_max": cosine_T_max,
                "cosine_eta_min": cosine_eta_min,
                "use_grad_clip": use_grad_clip,
                "grad_clip_value": grad_clip_value,
                "beta1": locals().get("beta1", 0.9),
                "beta2": locals().get("beta2", 0.999),
                "eps": locals().get("eps", 1e-8),
                "momentum": locals().get("momentum", 0.9),
                "nesterov": locals().get("nesterov", True),
                "rmsprop_alpha": locals().get("alpha", 0.99),
            }

        else:
            params = None

        params_list = [params]

        if dist.is_initialized():
            # Broadcast the parameters to all ranks
            dist.broadcast_object_list(params_list, src=0)
        params = params_list[0]

        model = LitClassifier(
            input_dim=input_dim,
            hidden_dims=params["hidden_dims"],
            num_classes=NUM_CLASSES,
            optimizer_class=params["optimizer_class"],
            activation=params["activation"],
            kernel_init=params["kernel_init"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            dropout=params["drop"],
            class_weights=params["weights"],
            cosine_T_max=params["cosine_T_max"],
            cosine_eta_min=params["cosine_eta_min"],
            use_grad_clip=params["use_grad_clip"],
            grad_clip_value=params["grad_clip_value"],
            beta1=params["beta1"],
            beta2=params["beta2"],
            eps=params["eps"],
            momentum=params["momentum"],
            nesterov=params["nesterov"],
            rmsprop_alpha=params["rmsprop_alpha"],
            rmsprop_momentum=params.get("rmsprop_momentum", 0.9),  # Add this if missing
            domain_task=False,
        )

        # print(model)

    else:
        # Domain boundary detection task, HPs for Transformer model
        if RANK == 0:
            d_model = trial.suggest_categorical("d_model", [256, 512, 768, 1024])
            n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])

            while d_model % n_heads != 0:
                n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])

            n_layers = trial.suggest_int("n_layers", 2, 4, step=2)
            d_ff = 4 * d_model
            max_seq_len = trial.suggest_int("max_seq_len", 100, 1000, step=100)

            drop = trial.suggest_float("dropout", 0.1, 0.5)
            drop_attn = trial.suggest_float("dropout_attn", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
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

            print(
                f"\n=== Trial {trial.number} ===\n"
                f"Hidden layers: {params['n_layers']}, Model dim: {params['d_model']}\n"
                f"Number of heads: {params['n_heads']}, FF dim: {params['d_ff']}\n"
                f"Attention dropout: {params['drop_attn']:.3f}, weights: {weights}\n"
                f"Dropout: {params['drop']:.3f}, LR: {params['lr']:.6f}, WD: {params['weight_decay']:.6f}\n"
                f"Optimizer: {params['optimizer_class'].__name__}, Activation: {params['activation'].__class__.__name__}\n"
                f"========================\n"
            )

        else:
            params = None

        params_list = [params]
        dist.broadcast_object_list(params_list, src=0)
        params = params_list[0]

        model = LitClassifier(
            input_dim=input_dim.size(2),
            hidden_dims=params["d_model"],
            num_classes=2,  # Binary classification for domain boundary detection
            optimizer_class=params["optimizer_class"],
            activation=params["activation"],
            kernel_init=params["kernel_init"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            dropout=params["drop"],
            class_weights=weights,
            domain_task=True,
            num_heads=params["n_heads"],
            num_layers=params["n_layers"],
        )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=-1 if dist.is_initialized() else 1,
        strategy="ddp" if dist.is_initialized() else "auto",
        enable_progress_bar=True,
        callbacks=[early_stop, checkpoint_callback],
        logger=TensorBoardLogger(
            save_dir=f"./logs/FINAL/{PROJECT_NAME}/tensorboard", 
            name=f"optuna_trial_{trial.number}"
        ),
    )

    # Log hyperparameters to TensorBoard
    if RANK == 0:
        if domain_task is False:
            hparams = {
                "num_neurons": params["hidden_dims"][0] if params["hidden_dims"] else 0,
                "num_hidden_layers": int(len(params["hidden_dims"])),
                "dropout": params["drop"],
                "lr": params["lr"],
                "weight_decay": params["weight_decay"],
                "optimizer": params["optimizer_class"].__name__,
                "activation": params["activation"].__class__.__name__,
            }
        else:
            # Domain task hyperparameters
            hparams = {
                "d_model": params["d_model"],
                "num_heads": params["n_heads"],
                "num_layers": params["n_layers"],
                "dropout": params["drop"],
                "lr": params["lr"],
                "weight_decay": params["weight_decay"],
                "optimizer": params["optimizer_class"].__name__,
                "activation": params["activation"].__class__.__name__,
            }
        
        trainer.logger.log_hyperparams(hparams)

    trainer.fit(model, train_loader, val_loader)
    
    # Log final metrics as hp_metric
    best_val_loss = checkpoint_callback.best_model_score.item()
    
    val_precision = trainer.logged_metrics.get("val_prec_avg", 0.0)
    if hasattr(val_precision, 'item'):
        val_precision = val_precision.item()    
         
        
    if RANK == 0:
        print(f"Trial {trial.number}: val_loss={best_val_loss:.4f}, val_precision={val_precision:.4f}")
        trainer.logger.experiment.add_scalar('hp_metric_loss', best_val_loss, 0)
        trainer.logger.experiment.add_scalar('hp_metric_precision', val_precision, 0)
    
    if domain_task is False:
        return best_val_loss, val_precision
    else:
        return best_val_loss


def load_best_model(trial, input_dim, weights, domain_task=False):
    if domain_task is False:
        p = trial.params

        n_neurons = p["num_neurons"]
        n_layers = p["num_hidden_layers"]
        hidden_dims = [n_neurons] * n_layers

        cosine_T_max = p.get("cosine_T_max", 20)        # Default fallback
        cosine_eta_min = p.get("cosine_eta_min", 1e-6)  # Default fallback

        # FIX: Don't use trial.suggest_* here, just get from params
        use_grad_clip = p.get("use_grad_clip", False)      # Get from existing params
        grad_clip_value = p.get("grad_clip_value", 1.0)    # Get from existing params

        opt_name = p["optimizer"]
        if opt_name == "adam":
            optimizer_class = torch.optim.Adam
        elif opt_name == "adamw":
            optimizer_class = torch.optim.AdamW
        elif opt_name == "nadam":  # Add this case
            optimizer_class = torch.optim.NAdam
        elif opt_name == "rmsprop":
            optimizer_class = torch.optim.RMSprop
        else:  # sgd
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
        
        # Extract optimizer-specific parameters from trial params
        beta1 = p.get("beta1", 0.9)
        beta2 = p.get("beta2", 0.999)
        eps = p.get("eps", 1e-8)
        momentum = p.get("momentum", 0.9)
        nesterov = p.get("nesterov", True)
        rmsprop_alpha = p.get("rmsprop_alpha", 0.99)
        rmsprop_momentum = p.get("rmsprop_momentum", 0.9)

        model = LitClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=NUM_CLASSES,
            optimizer_class=optimizer_class,
            activation=activation_fn,
            kernel_init=kernel_init,
            lr=lr,
            weight_decay=wd,
            dropout=drop,
            class_weights=weights,
            cosine_T_max=cosine_T_max,
            cosine_eta_min=cosine_eta_min,
            use_grad_clip=use_grad_clip,
            grad_clip_value=grad_clip_value,
            # Add optimizer-specific parameters
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            momentum=momentum,
            nesterov=nesterov,
            rmsprop_alpha=rmsprop_alpha,
            rmsprop_momentum=rmsprop_momentum,
            domain_task=False,
        )

        print(
            f"\n=== FINAL TRAINING WITH {trial.number} ===\n"
            f"Hidden layers: {n_layers}, Neurons: {n_neurons}\n"
            f"Dropout: {drop:.3f}, LR: {lr:.6f}, WD: {wd:.6f}\n"
            f"Optimizer: {optimizer_class.__name__}, Activation: {activation_fn.__class__.__name__}\n"
            f"Cosine T_max: {cosine_T_max}, Cosine eta_min: {cosine_eta_min}\n"
            f"Use grad clip: {use_grad_clip}, Grad clip value: {grad_clip_value}\n"
            f"Beta1: {beta1}, Beta2: {beta2}, Eps: {eps}\n"
            f"Momentum: {momentum}, Nesterov: {nesterov}\n"
            f"RMSprop alpha: {rmsprop_alpha}, RMSprop momentum: {rmsprop_momentum}\n"
            f"========================\n"
        )

    else:
        # Domain task logic remains the same...
        p = trial.params
        d_model = p["d_model"]
        n_heads = p["n_heads"]
        n_layers = p["n_layers"]

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
        opt_name = p["optimizer"]
        if opt_name == "adam":
            optimizer_class = torch.optim.Adam
        elif opt_name == "adamw":
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.SGD

        model = LitClassifier(
            input_dim=input_dim.size(2),
            hidden_dims=d_model,
            num_classes=2,
            optimizer_class=optimizer_class,
            activation=activation_fn,
            kernel_init=kernel_init,
            lr=lr,
            weight_decay=wd,
            dropout=drop,
            class_weights=weights,
            domain_task=True,
            num_heads=n_heads,
            num_layers=n_layers,
        )

        print(
            f"\n=== FINAL TRAINING WITH {trial.number} ===\n"
            f"Hidden layers: {n_layers}, Hidden dim: {d_model}, n Heads: {n_heads}\n"
            f"Dropout: {drop:.3f}, LR: {lr:.6f}, WD: {wd:.6f}\n"
            f"Optimizer: {optimizer_class.__name__}, Activation: {activation_fn.__class__.__name__}\n"
            f"========================\n"
        )

    # Rest of the function remains the same...
    ckpt_dir = (
        f"./logs/FINAL/{PROJECT_NAME}/tensorboard/optuna_trial_{trial.number}/version_0/checkpoints/"
    )
    print(f"Looking for checkpoints in: {ckpt_dir}")
    pattern = os.path.join(ckpt_dir, "*.ckpt")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    chosen_ckpt = max(matches, key=os.path.getctime)
    print(f"Loading checkpoint: {chosen_ckpt}")

    model = LitClassifier.load_from_checkpoint(chosen_ckpt)
    model_path = f"./models/FINAL/{PROJECT_NAME}.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)
    model = model.to(DEVICE).eval()

    return model


# -------------------------
# 6. Class distribution checker and weights calculation
# -------------------------


def class_distribution(train_dataset, val_dataset, range_train=None, range_val=None):
    # Set defaults inside the function
    if range_train is None:
        range_train = len(train_dataset)
    if range_val is None:
        range_val = len(val_dataset)

    if range_val and range_train is not None:
        if RANK == 0:
            print("NOT USING FULL SET!")




    train_labels_all = []
    for i in range(range_train):
        _, labels = train_dataset[i]
        train_labels_all.append(labels.item())

    train_labels_array = np.array(train_labels_all)
    train_class_counts = np.bincount(train_labels_array, minlength=NUM_CLASSES)

    if RANK == 0:
        print("Training set label distribution:")
        for i in range(NUM_CLASSES):
            print(
                f"  Class {i}: {train_class_counts[i]} samples ({train_class_counts[i] / range_train * 100:.2f}%)"
            )
        print(f"  Total training samples: {range_train}")

    # Count ALL validation labels
    if RANK == 0:
        print("Extracting all validation labels for complete class distribution...")
    val_labels_all = []
    for i in range(range_val):
        _, labels = val_dataset[i]
        val_labels_all.append(labels.item())

    val_labels_array = np.array(val_labels_all)
    val_class_counts = np.bincount(val_labels_array, minlength=NUM_CLASSES)

    if RANK == 0:
        print("Validation set label distribution:")
        for i in range(NUM_CLASSES):
            print(
                f"  Class {i}: {val_class_counts[i]} samples ({val_class_counts[i] / range_val * 100:.2f}%)"
            )
        print(f"  Total validation samples: {range_val}")

    # Combined statistics
    total_class_counts = train_class_counts + val_class_counts
    total_samples = range_train + range_val

    if RANK == 0:
        print("Combined dataset label distribution:")
        for i in range(NUM_CLASSES):
            print(
                f"  Class {i}: {total_class_counts[i]} samples ({total_class_counts[i] / total_samples * 100:.2f}%)"
            )
        print(f"  Total combined samples: {total_samples}")

    # Calculate class weights based on training set
    total_count = train_class_counts.sum()
    weights = torch.tensor(
        [
            total_count / (NUM_CLASSES * count) if count > 0 else 0.0
            for count in train_class_counts
        ],
        dtype=torch.float32,
    )
    if RANK == 0:
        print("Class weights calculated:", weights.numpy())
    weights = weights.to(DEVICE)

    # Create the validation tensors for final evaluation
    val_embeddings = []
    val_labels = []
    if RANK == 0:
        print("Creating validation tensors for final evaluation...")
    for i in range(range_val):
        emb, label = val_dataset[i]
        val_embeddings.append(emb)
        val_labels.append(label.item())

    val_embeddings = torch.stack(val_embeddings)
    val_labels = torch.tensor(val_labels)

    # Clean up intermediate variables
    del train_labels_all, val_labels_all, train_labels_array, val_labels_array

    # Return the computed values
    return weights, val_embeddings, val_labels


# -------------------------
# 7. Main
# -------------------------


def main_HP(Final_training=False):
    os.makedirs("pickle", exist_ok=True)
    os.makedirs(f"logs/FINAL/{PROJECT_NAME}", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(CACHE_PATH):
        ESMDataset(
            esm_model=ESM_MODEL,
            FSDP_used=False,
            domain_boundary_detection=False,
            training=True,
            num_classes=NUM_CLASSES,
            csv_path=CSV_PATH,
            category_col=CATEGORY_COL,
            sequence_col=SEQUENCE_COL,
        )

    # Create separate datasets for train and validation data from H5 file
    class TrainDataset(torch.utils.data.Dataset):
        def __init__(self, h5_file):
            self.h5_file = h5_file

            with h5py.File(self.h5_file, "r") as f:
                # Get train embedding keys
                self.embedding_keys = sorted(
                    [k for k in f.keys() if k.startswith("train_embeddings_")]
                )
                self.chunk_sizes = [f[key].shape[0] for key in self.embedding_keys]
                self.cumulative_sizes = np.cumsum(self.chunk_sizes)
                self.length = (
                    self.cumulative_sizes[-1] if self.cumulative_sizes.size > 0 else 0
                )

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            with h5py.File(self.h5_file, "r") as f:
                chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")
                local_idx = (
                    idx - self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else idx
                )

                embedding_key = self.embedding_keys[chunk_idx]
                labels_key = embedding_key.replace("train_embeddings_", "train_labels_")

                embeddings = torch.tensor(
                    f[embedding_key][local_idx], dtype=torch.float32
                )
                labels = torch.tensor(f[labels_key][local_idx], dtype=torch.long)

                return embeddings, labels

    class ValDataset(torch.utils.data.Dataset):
        def __init__(self, h5_file):
            self.h5_file = h5_file

            with h5py.File(self.h5_file, "r") as f:
                # Get validation embedding keys
                self.embedding_keys = sorted(
                    [k for k in f.keys() if k.startswith("val_embeddings_")]
                )
                self.chunk_sizes = [f[key].shape[0] for key in self.embedding_keys]
                self.cumulative_sizes = np.cumsum(self.chunk_sizes)
                self.length = (
                    self.cumulative_sizes[-1] if self.cumulative_sizes.size > 0 else 0
                )

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            with h5py.File(self.h5_file, "r") as f:
                chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")
                local_idx = (
                    idx - self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else idx
                )

                embedding_key = self.embedding_keys[chunk_idx]
                labels_key = embedding_key.replace("val_embeddings_", "val_labels_")

                embeddings = torch.tensor(
                    f[embedding_key][local_idx], dtype=torch.float32
                )
                labels = torch.tensor(f[labels_key][local_idx], dtype=torch.long)

                return embeddings, labels

    # Create train and validation datasets
    train_dataset = TrainDataset(CACHE_PATH)
    val_dataset = ValDataset(CACHE_PATH)

    sample_embedding, sample_label = train_dataset[0]

    # print("shapes:", sample_embedding.dim(), sample_label.dim())
    # print(sample_embedding, sample_label)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        persistent_workers=True,
        num_workers=NUM_WORKERS_EMB,
        pin_memory=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        persistent_workers=True,
        num_workers=NUM_WORKERS_EMB,
        pin_memory=True,
        prefetch_factor=4,
    )

    if RANK == 0:
        print("Train and validation datasets created from cached embeddings.")
        print(
            f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}"
        )
        print("Dataset building complete")
        print("Calculating class weights and label distributions...")
        print("Extracting all training labels for complete class distribution...")

    weights, val_embeddings, val_labels = class_distribution(
        train_dataset, val_dataset, 10000, 10000
    )

    if RANK == 0:
        print("Starting Optuna hyperparameter search...")
        study = optuna.create_study(
            directions=["minimize", "maximize"],
            storage=f"sqlite:///logs/FINAL/{PROJECT_NAME}/optuna_study.db",
            load_if_exists=True,
            study_name=PROJECT_NAME,
        )


    if dist.is_initialized():
        dist.barrier()  # Ensure all ranks are synchronized before starting the study

    if RANK != 0:
        study = optuna.create_study(
            direction="minimize",
            storage=f"sqlite:///logs/FINAL/{PROJECT_NAME}/optuna_study.db",
            load_if_exists=True,
            study_name=PROJECT_NAME,
        )

    input_dim = sample_embedding.shape[
        0
    ]  # print(f"Input dimension for model: {input_dim_sample}")

    def objective_wrapper(trial):
        return objective(
            trial,
            input_dim,
            train_loader,
            val_loader,
            weights,
            domain_task=False,
            EPOCHS=EPOCHS,
        )

    study.optimize(
        objective_wrapper,
        n_trials=STUDY_N_TRIALS,
    )

    if RANK == 0:
        print("Optimization complete!")
        print(f"Number of trials: {len(study.trials)}")
        
        # For multi-objective, there's no single "best" trial
        # Get Pareto front (non-dominated solutions)
        pareto_trials = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                is_dominated = False
                for other_trial in study.trials:
                    if (other_trial.state == optuna.trial.TrialState.COMPLETE and 
                        other_trial != trial):
                        # Check if other_trial dominates trial
                        # (lower loss AND higher precision)
                        if (other_trial.values[0] <= trial.values[0] and  # loss
                            other_trial.values[1] >= trial.values[1] and  # precision
                            (other_trial.values[0] < trial.values[0] or 
                             other_trial.values[1] > trial.values[1])):
                            is_dominated = True
                            break
                if not is_dominated:
                    pareto_trials.append(trial)
        
        print(f"Pareto optimal trials: {len(pareto_trials)}")
        for i, trial in enumerate(pareto_trials):
            print(f"  Trial {trial.number}: loss={trial.values[0]:.4f}, precision={trial.values[1]:.4f}")
        
        # Choose the trial with best precision among Pareto optimal
        if pareto_trials:
            best_trial = max(pareto_trials, key=lambda t: t.values[1])
            print(f"Selected trial {best_trial.number} (highest precision): loss={best_trial.values[0]:.4f}, precision={best_trial.values[1]:.4f}")
        else:
            # Fallback if no Pareto trials found
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                best_trial = max(completed_trials, key=lambda t: t.values[1] if len(t.values) > 1 else 0)
                print(f"Fallback: Selected trial {best_trial.number}")
            else:
                print("No completed trials found!")
                best_trial = None

    else:
        # For non-rank 0 processes, initialize best_trial as None
        best_trial = None

    # Broadcast the best_trial to all ranks
    if dist.is_initialized():
        # Create a list to broadcast the best trial information
        best_trial_data = [None]
        
        if RANK == 0 and best_trial is not None:
            # Send the trial number which can be used to reconstruct the trial
            best_trial_data[0] = best_trial.number
        
        dist.broadcast_object_list(best_trial_data, src=0)
        
        # Reconstruct the best_trial on non-rank 0 processes
        if RANK != 0 and best_trial_data[0] is not None:
            trial_number = best_trial_data[0]
            # Find the trial with the matching number
            for trial in study.trials:
                if trial.number == trial_number:
                    best_trial = trial
                    break
        elif best_trial_data[0] is None:
            best_trial = None

    # Final fallback if best_trial is still None
    if best_trial is None:
        if len(study.trials) > 0:
            # Use the last completed trial as fallback
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                best_trial = completed_trials[-1]
                if RANK == 0:
                    print(f"Using fallback trial {best_trial.number}")
            else:
                raise RuntimeError("No completed trials found in the study!")
        else:
            raise RuntimeError("No trials found in the study!")




    lit_model = load_best_model(best_trial, input_dim, weights)

    lit_model.eval()
    print("Best model loaded and set to eval mode")

    if Final_training is True:
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, mode="min", verbose=True
        )
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

        print("Lit model created")
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=-1 if dist.is_initialized() else 1,
            strategy="ddp" if dist.is_initialized() else "auto",
            enable_progress_bar=True,
            callbacks=[early_stop, checkpoint_callback],
            logger=TensorBoardLogger(
                save_dir=f"./logs/FINAL/{PROJECT_NAME}/tensorboard", name=f"{PROJECT_NAME}_final"
            ),
        )

        print("Trainer created")

        trainer.fit(lit_model, train_loader, val_loader)

        # save the final model
        final_model_path = f"./models/FINAL/{PROJECT_NAME}.pt"


        if RANK == 0:
            print(f"\nTraining complete, saving final model under: {final_model_path} ...\n")


        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(lit_model, final_model_path)

    if dist.is_initialized():
        # Ensure all processes are synchronized before exiting
        dist.barrier()
        if RANK == 0:
            print("Done! Exiting...")
        dist.destroy_process_group()



# -------------------------
# 8. Helper Func for Main for usage
# -------------------------

class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file

        with h5py.File(self.h5_file, "r") as f:
            # Get embedding keys (could be train_embeddings_ or just embeddings_)
            self.embedding_keys = sorted([
                k for k in f.keys() 
                if k.startswith("embeddings_") or k.startswith("train_embeddings_")
            ])
            self.chunk_sizes = [f[key].shape[0] for key in self.embedding_keys]
            self.cumulative_sizes = np.cumsum(self.chunk_sizes)
            self.length = (
                self.cumulative_sizes[-1] if self.cumulative_sizes.size > 0 else 0
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, "r") as f:
            chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")
            local_idx = (
                idx - self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else idx
            )

            embedding_key = self.embedding_keys[chunk_idx]
            embeddings = torch.tensor(
                f[embedding_key][local_idx], dtype=torch.float32
            )

            return embeddings  # Only return embeddings, no labels


def loader(csv_path):
    if not os.path.exists("tempTest/embeddings/embeddings_domain_classifier.h5"):
        ESMDataset(
            esm_model=ESM_MODEL,
            FSDP_used=False,
            domain_boundary_detection=False,
            training=True,
            num_classes=NUM_CLASSES,
            csv_path=csv_path,
            category_col=CATEGORY_COL,
            sequence_col=SEQUENCE_COL,
        )




    # Create inference dataset
    classifier_dataset = ClassifierDataset("./tempTest/embeddings/embeddings_domain_classifier.h5")


    # Create DataLoader for inference
    classifier_loader = DataLoader(
        classifier_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        persistent_workers=True,
        num_workers=NUM_WORKERS_EMB,
        pin_memory=True,
        prefetch_factor=4,
    )


    # Load the best model
    model = torch.load("./models/Optuna_1000d_uncut_t33.pt",weights_only=False)
    model.to(DEVICE).eval()
    print("Model loaded and set to eval mode")
    print(len(classifier_loader), "batches in classifier_loader")

    return model, classifier_loader



def predicter(model, classifier_loader):
    all_predictions = []
    all_predictions_raw = []

    with torch.no_grad():
        for batch in classifier_loader:
            # ClassifierDataset only returns embeddings, not labels/starts/ends
            inputs = batch
            inputs = inputs.to(DEVICE)
            output = model(inputs)

            # Get predictions and raw scores
            preds_raw, preds = torch.max(output, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_predictions_raw.extend(preds_raw.cpu().numpy())

    return all_predictions, all_predictions_raw


# -------------------------
# 9. Main for Usage
# -------------------------

def main(csv_path):
    model, classifier_loader = loader(csv_path)

    all_predictions, all_predictions_raw = predicter(model, classifier_loader)

    return all_predictions, all_predictions_raw


def parse_args():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ESM Embeddings HP Search")
    parser.add_argument("--csv_path", type=str, default="./Dataframes/v3/RemainingEntriesCompleteProteins.csv", help="Path to input CSV file")
    parser.add_argument("--HP_mode", action="store_true", help="Use to run hyperparameter optimization")
    
    return parser.parse_args()


#############################################################################################################

if __name__ == "__main__":

    args = parse_args()

    # early exit if in HP mode, starting just HP search
    if args.HP_mode is True:
        main_HP(Final_training=True)
        exit(0)

    # continue with normal execution for usage
    csv_path = args.csv_path
    all_predictions, all_predictions_raw = main(csv_path)
    
    # Save predictions to a file that main.py can read
    import pandas as pd
    
    predictions_df = pd.DataFrame({
        'prediction': all_predictions,
        'raw_scores': all_predictions_raw
    })
    
    os.makedirs('./tempTest', exist_ok=True)
    predictions_df.to_csv('./tempTest/predictions.csv', index=False)
    
    if RANK == 0:
        print("Predictions saved to ./tempTest/predictions.csv")
        print(f"Total predictions: {len(all_predictions)}")