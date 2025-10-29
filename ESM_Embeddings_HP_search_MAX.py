# -------------------------
# 1. Imports & basic setup
# -------------------------
import glob
import os
import time
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
from torch.utils.data import DataLoader, WeightedRandomSampler,Sampler
from torchmetrics import Precision, Recall
import torch.nn.functional as F
import tempfile
from tqdm import tqdm
import numpy as np
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
NUM_CLASSES = 24381  # classes + 1 for "other" class
if RANK == 0:
    print("USING n CLASSES:",NUM_CLASSES)

CSV_PATH = "/scratch/tmp/sapelt/Master_Thesis/Dataframes/v3/RemainingEntriesCompleteProteins.csv"
CATEGORY_COL = "Pfam_id"
SEQUENCE_COL = "Sequence"
CACHE_PATH = f"/scratch/tmp/sapelt/Master_Thesis/temp/embeddings_classification_{NUM_CLASSES-1}d.h5"
PROJECT_NAME = f"t33_ALL_{NUM_CLASSES-1}d"

ESM_MODEL = "esm2_t33_650M_UR50D"

EPOCHS = 10000
STUDY_N_TRIALS = 10
BATCH_SIZE = 512
NUM_WORKERS_EMB = 8

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
        # x = F.normalize(x, dim=-1)
        return self.net(x)


# -------------------------
# 3. PyTorch-Lightning module
# -------------------------

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        # Ensure weights are on the same device as input
        if self.weight is not None and self.weight.device != input.device:
            self.weight = self.weight.to(input.device)
            
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
    
class ArcFace(torch.nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        import math
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # logits = F.normalize(logits, dim=1)
        # logits = torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7)
    
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        # print(logits)
        return logits
    

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


    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer_class = optimizer_class
        self.lr = lr
        self.weight_decay = weight_decay
        self.domain_task = domain_task
        self.class_weights = class_weights
        self.num_classes = num_classes


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


            # class_weights = None

            if class_weights is not None:
                if RANK == 0:
                    print("USING CLASS WEIGHTS IN LOSS FUNCTION")
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                if RANK == 0:
                    print("WARNING: No class weights provided, using unweighted loss function")
                self.loss_fn = nn.CrossEntropyLoss()


            # if class_weights is not None:
            #     # Use Focal Loss instead of CrossEntropyLoss
            #     self.loss_fn = FocalLoss(weight=class_weights, gamma=2.0)
            # else:
            #     print("WARNING: No class weights provided, using unweighted focal loss")
            #     self.loss_fn = FocalLoss(gamma=2.0)


            # # ArcFace margin-based loss
            # self.ArcFace = ArcFace(s=30.0, margin=0.5)

            # if class_weights is not None:
            #     self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            # else:
            #     self.loss_fn = nn.CrossEntropyLoss()

        else:
            print("Domain segmentation task activated - using Transformer model")
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

    def distance_aware_boundary_loss(self, logits, labels, alpha=2.0, beta=1.0, gamma=0.5, fbeta_weight=1.5, epsilon=1e-8):
        """
        Custom loss that heavily penalizes boundary predictions far from true boundaries.
        
        Args:
            logits: Model predictions [seq_len, num_classes]
            labels: True labels [seq_len] (0 for non-boundary, 1 for boundary)
            alpha: Weight for distance penalty (higher = more penalty for far predictions)
            beta: Weight for base classification loss
            epsilon: Small value for numerical stability
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        boundary_probs = probs[:, 1]  # Probabilities for boundary class
        binary_labels = labels.float()
        
        # Base classification loss (cross-entropy)
        ce_loss = -binary_labels * torch.log(boundary_probs + epsilon) - (1 - binary_labels) * torch.log(1 - boundary_probs + epsilon)
        
        # Find positions of true boundaries
        true_boundary_positions = torch.where(binary_labels == 1)[0]
                
        # Calculate distance penalty for false positive predictions
        false_positive_mask = (binary_labels == 0) & (boundary_probs > 0.5)
        
        # Only calculate distance penalty if there are both false positives AND true boundaries
        if false_positive_mask.any() and len(true_boundary_positions) > 0:
            fp_positions = torch.where(false_positive_mask)[0]
            
            # For each false positive, find distance to nearest true boundary
            distance_penalties = []
            for fp_pos in fp_positions:
                # Calculate distances to all true boundaries
                distances = torch.abs(fp_pos - true_boundary_positions.float())
                min_distance = torch.min(distances)
                
                # Distance penalty: exponential decay based on distance
                # Closer false positives get higher penalties
                distance_penalty = torch.exp(-min_distance / 10.0)  # Adjust denominator to control decay rate
                
                # Weight by prediction confidence
                fp_confidence = boundary_probs[fp_pos]
                weighted_penalty = distance_penalty * fp_confidence * alpha
                
                distance_penalties.append(weighted_penalty)
            
            distance_penalty_term = torch.stack(distance_penalties).mean()
        else:
            distance_penalty_term = torch.tensor(0.0, device=logits.device)
        
        tp_soft = (boundary_probs * binary_labels).sum()
        fp_soft = (boundary_probs * (1 - binary_labels)).sum() 
        fn_soft = ((1 - boundary_probs) * binary_labels).sum()
        
        precision_soft = tp_soft / (tp_soft + fp_soft + epsilon)
        recall_soft = tp_soft / (tp_soft + fn_soft + epsilon)
        
        fbeta_score_soft = (1 + fbeta_weight**2) * precision_soft * recall_soft / (fbeta_weight**2 * precision_soft + recall_soft + epsilon)
        fbeta_loss = 1.0 - fbeta_score_soft
        
        # 4. Properly combine with defined gamma
        total_loss = beta * ce_loss.mean() + distance_penalty_term + gamma * fbeta_loss
        return total_loss
        
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


        # if self.loss_fn is ArcFace:
            # Apply ArcFace transformation
        # transformed_logits = self.ArcFace(logits, y)
        # logits = transformed_logits



        # print('SHAPES',logits.shape, y.shape)

        if self.domain_task is True:
            # Process each sequence individually
            batch_size, seq_len, num_classes = logits.shape
            losses = []

            for i in range(batch_size):
                seq_logits = logits[i]  # [seq_len, num_classes]
                seq_labels = y[i]  # [seq_len]
                seq_loss = self.distance_aware_boundary_loss(seq_logits, seq_labels)
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


        # if self.loss_fn is ArcFace:
            # Apply ArcFace transformation
        # transformed_logits = self.ArcFace(logits, y)
        # logits = transformed_logits





        if self.domain_task is True:
            # Process each sequence individually
            batch_size, seq_len, num_classes = logits.shape
            losses = []
            all_preds = []
            all_labels = []

            for i in range(batch_size):
                seq_logits = logits[i]  # [seq_len, num_classes]
                seq_labels = y[i]  # [seq_len]
                seq_loss = self.distance_aware_boundary_loss(seq_logits, seq_labels)
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

        if RANK == 0:  # Only print for first batch to avoid spam
            if batch_idx == 0:  # Only print for first batch to avoid spam
                    # print(f"Batch {batch_idx}:")
                print(f"Preds: {preds[:20]}")
                print(f"Labels: {y_flat[:20]}")
                # print(f"Logits sample (first sequence, first 5 positions):\n{logits[0, :5, :]}")
                    # print(f"Max logits: {logits.max()}, Min logits:\n{logits.min()}")
                    # print(f"Class 0 logits mean: {logits[:, :, 0].mean()}")
                    # print(f"Class 1 logits mean: {logits[:, :, 1].mean()}")
                    # print(f"Loss function weights: {self.loss_fn.weight}")
                # print(f"Prediction distribution: {torch.bincount(preds)}")
                # print(f"True label distribution: {torch.bincount(y_flat)}")
                print("-" * 50)

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
                nesterov=True,
                momentum=0.9,  # Add momentum for SGD
                dampening=0,   # Set dampening to 0 for Nesterov
            )
        elif self.optimizer_class in [torch.optim.Adam, torch.optim.AdamW, torch.optim.NAdam]:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,

            )
        elif self.optimizer_class == torch.optim.RMSprop:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,

            )
        else:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,  
                "monitor": "val_loss",   
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
            n_neurons = trial.suggest_int("num_neurons", 640, 1440, step=200) #128,1536,64
            hidden_dims = [
                n_neurons for _ in range(trial.suggest_int("num_hidden_layers", 1, 3, step=1))
            ]

            drop = trial.suggest_float("dropout", 0.05, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)  
            wd = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)  



            optimizer = trial.suggest_categorical(
                "optimizer", ["adam", "adamw", "sgd", "rmsprop", "nadam"]
            )
                        

            # Tune optimizer-specific parameters
            if optimizer in ["adam", "adamw", "nadam"]:
                
                if optimizer == "adam":
                    optimizer_class = torch.optim.Adam
                elif optimizer == "adamw":
                    optimizer_class = torch.optim.AdamW
                else:  # nadam
                    optimizer_class = torch.optim.NAdam
                    
            elif optimizer == "sgd":
                optimizer_class = torch.optim.SGD
            else:  # rmsprop
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

            if RANK == 0:
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

        if RANK == 0:
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

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True, min_delta=0.01)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=-1 if dist.is_initialized() else 1,
        strategy="ddp" if dist.is_initialized() else "auto",
        enable_progress_bar=True,
        callbacks=[early_stop, checkpoint_callback],
        logger=TensorBoardLogger(
            save_dir=f"/scratch/tmp/sapelt/Master_Thesis/logs/FINAL/{PROJECT_NAME}/tensorboard", 
            name=f"optuna_trial_{trial.number}"
        ),
        limit_train_batches=250, #245
        limit_val_batches=250,   #245
        use_distributed_sampler=False,
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
        
        if RANK == 0:
            print("DOMAIN TASK:", domain_task)

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
            domain_task=False,
        )
        if RANK == 0:
            print(
                f"\n=== FINAL TRAINING WITH {trial.number} ===\n"
                f"Hidden layers: {n_layers}, Neurons: {n_neurons}\n"
                f"Dropout: {drop:.3f}, LR: {lr:.6f}, WD: {wd:.6f}\n"
                f"Optimizer: {optimizer_class.__name__}, Activation: {activation_fn.__class__.__name__}\n"
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

        if RANK == 0:
            print(
                f"\n=== FINAL TRAINING WITH {trial.number} ===\n"
                f"Hidden layers: {n_layers}, Hidden dim: {d_model}, n Heads: {n_heads}\n"
                f"Dropout: {drop:.3f}, LR: {lr:.6f}, WD: {wd:.6f}\n"
                f"Optimizer: {optimizer_class.__name__}, Activation: {activation_fn.__class__.__name__}\n"
                f"========================\n"
            )

    # Rest of the function remains the same...
    # ckpt_dir = (
    #     f"/scratch/tmp/sapelt/Master_Thesis/logs/FINAL/{PROJECT_NAME}/tensorboard/optuna_trial_{trial.number}/version_0/checkpoints/"
    # )
    # print(f"Looking for checkpoints in: {ckpt_dir}")
    # pattern = os.path.join(ckpt_dir, "*.ckpt")
    # matches = glob.glob(pattern)

    # if not matches:
    #     raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    # chosen_ckpt = max(matches, key=os.path.getctime)
    # print(f"Loading checkpoint: {chosen_ckpt}")

    # model = LitClassifier.load_from_checkpoint(chosen_ckpt)
    



    # model_path = f"/scratch/tmp/sapelt/Master_Thesis/models/FINAL/{PROJECT_NAME}.pt"
    # os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # torch.save(model, model_path)
    # model = model.to(DEVICE).eval()



    # print("MODEL",model)
    return model


# -------------------------
# 6. Class distribution checker and weights calculation
# -------------------------


def class_distribution(val_dataset):
    # Fast bulk read for train labels
    with h5py.File(CACHE_PATH, "r") as f:
        train_labels = np.concatenate([f[key][:] for key in f.keys() if key.startswith("train_labels_")])
        val_labels = np.concatenate([f[key][:] for key in f.keys() if key.startswith("val_labels_")])

    train_class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    val_class_counts = np.bincount(val_labels, minlength=NUM_CLASSES)

    total_class_counts = train_class_counts + val_class_counts
    total_samples = len(train_labels) + len(val_labels)

    # Calculate class weights
    total_count = train_class_counts.sum()
    weights = torch.tensor(
        [
            total_count / (NUM_CLASSES * count) if count > 0 else 0.0
            for count in train_class_counts
        ],
        dtype=torch.float32,
    ).to(DEVICE)




 
    # # soften weights
    # alpha = 0.5
    # weights = weights ** alpha
    # weights = weights / weights.sum() * NUM_CLASSES  # Normalize to keep mean weight = 1



    # Optionally print distributions
    if RANK == 0:
        print("Training set label distribution:")
        for i in range(NUM_CLASSES):
            print(f"  Class {i}: {train_class_counts[i]} samples ({train_class_counts[i] / len(train_labels) * 100:.2f}%)")
        print(f"  Total training samples: {len(train_labels)}")
        print("Validation set label distribution:")
        for i in range(NUM_CLASSES):
            print(f"  Class {i}: {val_class_counts[i]} samples ({val_class_counts[i] / len(val_labels) * 100:.2f}%)")
        print(f"  Total validation samples: {len(val_labels)}")
        print("Combined dataset label distribution:")
        for i in range(NUM_CLASSES):
            print(f"  Class {i}: {total_class_counts[i]} samples ({total_class_counts[i] / total_samples * 100:.2f}%)")
        print(f"  Total combined samples: {total_samples}")
        print("Class weights calculated:", weights.cpu().numpy())
        

    #with h5py.File(CACHE_PATH, "r") as f:
    #    val_embeddings = np.concatenate([f[key][:] for key in f.keys() if key.startswith("val_embeddings_")])
    #val_embeddings = torch.tensor(val_embeddings)

    val_embeddings=None

    return weights, val_embeddings, val_labels, train_labels

# -------------------------
# 7. Main
# -------------------------


def main_HP(Final_training=False):
    os.makedirs("/scratch/tmp/sapelt/Master_Thesis/pickle", exist_ok=True)
    os.makedirs(f"/scratch/tmp/sapelt/Master_Thesis/logs/FINAL/{PROJECT_NAME}", exist_ok=True)
    os.makedirs("/scratch/tmp/sapelt/Master_Thesis/models", exist_ok=True)
    # Check if cache file exists, and if not, create embeddings and log progress
    if os.path.exists(f"/scratch/tmp/sapelt/Master_Thesis/temp/progress_{NUM_CLASSES}.txt"):
        with open(f"/scratch/tmp/sapelt/Master_Thesis/temp/progress_{NUM_CLASSES}.txt", "r") as status_file:
            if "All chunks processed. Exiting." not in status_file.read():
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
    elif not os.path.exists(CACHE_PATH):
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

    print("shapes:", sample_embedding.dim(), sample_label.dim())
    print(sample_embedding, sample_label)

    

    # Create DataLoaders with optimized settings

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        persistent_workers=True,
        num_workers=NUM_WORKERS_EMB,
        pin_memory=True,
        prefetch_factor=4,
        # drop_last=True,

    )

    if RANK == 0:
        print("Train and validation datasets created from cached embeddings.")
        print(
            f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}"
        )
        print("Dataset building complete")
        print("Calculating class weights and label distributions...")
        print("Extracting all training labels for complete class distribution...")

    weights, val_embeddings, val_labels, train_labels = class_distribution(
        val_dataset
    )


    # class_counts = torch.bincount(torch.tensor(train_labels[:25*5005]))

    # num_samples_per_class = int(class_counts.min().item() * 2)  # Or whatever ratio you want
    # num_samples = num_samples_per_class * NUM_CLASSES
    
    
    # # Create sample weights inversely proportional to class frequency
    # if os.path.exists(f"/scratch/tmp/sapelt/Master_Thesis/temp/sample_weights_{NUM_CLASSES}_idx.npy"):
    #     sample_weights = np.load(f"/scratch/tmp/sapelt/Master_Thesis/temp/sample_weights_{NUM_CLASSES}_idx.npy")
    # else:
    #     sample_weights = [1.0/class_counts[label] for label in tqdm(train_labels[:25*5005], desc="Creating sample weights")]
    #     sample_weights = np.array(sample_weights)
    #     np.save(f"/scratch/tmp/sapelt/Master_Thesis/temp/sample_weights_{NUM_CLASSES}_idx.npy", sample_weights)
        
    # torch.from_numpy(sample_weights)   


    # class CustomWeightedRandomSampler(WeightedRandomSampler):
    #     """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)

    #     def __iter__(self):
    #         rand_tensor = np.random.choice(range(0, len(self.weights)),
    #                                     size=self.num_samples,
    #                                     p=self.weights.numpy() / torch.sum(self.weights).numpy(),
    #                                     replace=self.replacement)
    #         rand_tensor = torch.from_numpy(rand_tensor)
    #         return iter(rand_tensor.tolist())


    # sampler = CustomWeightedRandomSampler(
    #     weights=sample_weights, 
    #     num_samples=min(num_samples, len(train_dataset)), 
    #     replacement=False
    # )




    # class mediansampler(Sampler):
    #     """Samples each class up to the median count of all classes.
    #     """
    #     def __init__(self, data_source, class_counts, train_labels):
    #         self.data_source = data_source
    #         self.num_samples = len(self.data_source)
    #         self.median_count = class_counts.median().item()
    #         self.train_labels = train_labels
            
    #     def __iter__(self):
    #         # Group indices by their classes
    #         indices_by_class = {}
    #         for idx, label in enumerate(self.train_labels):
    #             if label not in indices_by_class:
    #                 indices_by_class[label] = []
    #             indices_by_class[label].append(idx)
            
    #         # Sample each class up to the median count
    #         sampled_indices = []
    #         for class_idx, indices in indices_by_class.items():
    #             if len(indices) > 0:
    #                 if len(indices) >= self.median_count:
    #                     # For overrepresented classes, undersample without replacement
    #                     sampled = np.random.choice(indices, size=self.median_count, replace=False)
    #                 else:
    #                     # For underrepresented classes, use all samples + oversample to reach median
    #                     sampled = indices.copy()  # Include all original samples
    #                     additional = np.random.choice(indices, size=self.median_count-len(indices), replace=True)
    #                     sampled.extend(additional)
    #                 sampled_indices.extend(sampled)
            
    #         # Shuffle the samples to avoid bias
    #         np.random.shuffle(sampled_indices)
            
    #         return iter(sampled_indices)
            
    #     def __len__(self):
    #         num_classes = len(set(self.train_labels))
    #         return int(num_classes * self.median_count)



    # median_sampler = mediansampler(train_dataset, class_counts, train_labels)




    train_loader = DataLoader(
        train_dataset,
        # sampler=sampler,
        # batch_sampler=batch_sampler,
        batch_size=BATCH_SIZE,
        shuffle=True,
        persistent_workers=True,
        num_workers=NUM_WORKERS_EMB,
        pin_memory=True,
        prefetch_factor=4,
        # drop_last=True,
    ) 


    # print("TRAIN LOADER,",train_loader[:10])



    if RANK == 0:
        print("Starting Optuna hyperparameter search...")
        study = optuna.create_study(
            directions=["minimize", "maximize"],
            storage=f"sqlite:////scratch/tmp/sapelt/Master_Thesis/logs/FINAL/{PROJECT_NAME}/optuna_study.db",
            load_if_exists=True,
            study_name=PROJECT_NAME,
            # overwrite=True,
        )


    if dist.is_initialized():
        dist.barrier()  # Ensure all ranks are synchronized before starting the study

    if RANK != 0:
        study = optuna.create_study(
            direction="minimize",
            storage=f"sqlite:////scratch/tmp/sapelt/Master_Thesis/logs/FINAL/{PROJECT_NAME}/optuna_study.db",
            load_if_exists=True,
            study_name=PROJECT_NAME,
            # overwrite=True,
        )

    input_dim = sample_embedding.shape[
        0
    ]  
    
    # print(f"Input dimension for model: {input_dim_sample}")

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
        
        if RANK == 0:
            print(f"Pareto optimal trials: {len(pareto_trials)}")
        for i, trial in enumerate(pareto_trials):
            if RANK == 0:
                print(f"  Trial {trial.number}: loss={trial.values[0]:.4f}, precision={trial.values[1]:.4f}")
        
        # Choose the trial with best precision among Pareto optimal
        if pareto_trials:
            best_trial = max(pareto_trials, key=lambda t: t.values[1])
            if "num_neurons" in best_trial.params:
                if RANK == 0:
                    print(f"Selected trial {best_trial.number} (highest precision): loss={best_trial.values[0]:.4f}, precision={best_trial.values[1]:.4f}")
            else:
                for i in range(1,10):
                    if RANK == 0:
                        print("No best trial found, searching numbers close to it...")
                    best_trial = study.trials[best_trial.number + i]
                    if "num_neurons" not in best_trial.params:
                        best_trial = study.trials[best_trial.number - i]
                    if "num_neurons" in best_trial.params:
                        if RANK == 0:
                            print(f"Selected trial {best_trial.number} (Found actual trial): loss={best_trial.values[0]:.4f}, precision={best_trial.values[1]:.4f}")
                        break


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

    #lit_model.eval()
    if RANK == 0:
        print("Best model loaded and set to eval mode")

    if Final_training is True:
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, mode="min", verbose=True, min_delta=0.01
        )
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
        if RANK == 0:
            print("Lit model created")
        
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=-1 if dist.is_initialized() else 1,
            strategy="ddp" if dist.is_initialized() else "auto",
            enable_progress_bar=True,
            callbacks=[early_stop, checkpoint_callback],
            logger=TensorBoardLogger(
                save_dir=f"/scratch/tmp/sapelt/Master_Thesis/logs/FINAL/{PROJECT_NAME}/tensorboard", name=f"{PROJECT_NAME}_final"
            ),
            limit_train_batches=250,
            limit_val_batches=250,
            use_distributed_sampler=False,
        )
        if RANK == 0:
            print("Trainer created")

        trainer.fit(lit_model, train_loader, val_loader)

        # save the final model
        final_model_path = f"/scratch/tmp/sapelt/Master_Thesis/models/FINAL/{PROJECT_NAME}.pt"

        if RANK == 0:
            print(f"\nTraining complete, saving final model under: {final_model_path} ...\n")


        if hasattr(lit_model, 'model') and hasattr(lit_model.model, '__class__') and 'Transformer' in lit_model.model.__class__.__name__:
            pass
        else:
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
    if not os.path.exists("/scratch/tmp/sapelt/Master_Thesis/tempTest/embeddings/embeddings_domain_classifier.h5"):
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
    classifier_dataset = ClassifierDataset("/scratch/tmp/sapelt/Master_Thesis/tempTest/embeddings/embeddings_domain_classifier.h5")


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
    model = torch.load("/scratch/tmp/sapelt/Master_Thesis/models/Optuna_1000d_uncut_t33.pt",weights_only=False)
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
    parser.add_argument("--csv_path", type=str, default="/scratch/tmp/sapelt/Master_Thesis/Dataframes/v3/RemainingEntriesCompleteProteins.csv", help="Path to input CSV file")
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
    
    os.makedirs('/scratch/tmp/sapelt/Master_Thesis/tempTest', exist_ok=True)
    predictions_df.to_csv('/scratch/tmp/sapelt/Master_Thesis/tempTest/predictions.csv', index=False)
    
    if RANK == 0:
        print("Predictions saved to /scratch/tmp/sapelt/Master_Thesis/tempTest/predictions.csv")
        print(f"Total predictions: {len(all_predictions)}")
