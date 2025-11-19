"""
ESM_Embeddings_HP_search.py 

Table of Contents:
===================

Classes:
--------
1. FFNClassifier
2. FocalLoss  
3. ArcFace
4. LitClassifier
5. TrainDataset
6. ValDataset
7. ClassifierDataset

Functions:
----------
1. objective()
2. class_distribution()
3. run_optuna_study()
4. find_and_broadcast_best_trial()
5. load_best_model()
6. final_training()
7. loader()
8. predictor()
9. main_usage()
10. parse_args()
11. main_HP()
12. main()
"""
# --------------------------------------------------------------------------------------------
# 1. Imports & basic setup
# --------------------------------------------------------------------------------------------

import argparse
import math
import os

import h5py
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall

from ESM_Embedder import DEVICE, RANK, ESMDataset

# Show full tensor
torch.set_printoptions(threshold=float("inf"))

# ENV SETUP
os.environ["NCCL_P2P_DISABLE"] = "1"
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
pd.set_option("display.max_rows", None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

# --------------------------------------------------------------------------------------------
# 2. GLOBALS
# --------------------------------------------------------------------------------------------

NUM_CLASSES = 1001  # classes + 1 for "other" class or FULL for full pfam classification
if RANK == 0:
    print("USING n CLASSES:", NUM_CLASSES)

# Path to Train CSV, target column names, Cache path for embeddings, project name for Optuna/tensorboard saving directory
CSV_PATH = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/RemainingEntriesCompleteProteins.csv"
CATEGORY_COL = "Pfam_id"
SEQUENCE_COL = "Sequence"
CACHE_PATH = f"/global/research/students/sapelt/Masters/MasterThesis/temp/embeddings_classification_{NUM_CLASSES - 1}d.h5"
PROJECT_NAME = f"t33_ALL_{NUM_CLASSES - 1}d"

# ESM MODEL SELECTION
ESM_MODEL = "esm2_t33_650M_UR50D"

# HYPERPARAMETER SEARCH SETTINGS & EMB SIZE
EPOCHS = 10000
STUDY_N_TRIALS = 10
VRAM = psutil.virtual_memory().total // (1024**3)  # systems VRAM in GB
BATCH_SIZE = (
    512 if VRAM >= 24 else 256 if VRAM >= 16 else 128 if VRAM >= 8 else 64
)  # adjust batch size based on available VRAM
NUM_WORKERS_EMB = 64

# Data split fractions (test is missing, splitted beforehand into another CSV)
VAL_FRAC = 0.2
TRAIN_FRAC = 1 - VAL_FRAC


# --------------------------------------------------------------------------------------------
# 3. FFW classifier head
# --------------------------------------------------------------------------------------------


class FFNClassifier(nn.Module):
    """
    Basic Feed-Forward Neural Network classifier head that gets called in the LitClassifier PyTorch-Lightning module.
    Build with
    - input_dim: dimension of input features automatically inferred based on ESM model selected
    - hidden_dims: list of hidden layer dimensions
    - num_classes: number of output classes used in the final classification layer
    - dropout: dropout rate
    - activation: activation function
    - kernel_init: kernel initialization method
    Forward func returns class logits, basic nothing special here.
    """

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
        # Input layer
        prev_dim = input_dim
        # Build hidden layers
        for hdim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hdim),  # Linear layer
                nn.BatchNorm1d(hdim),  # Batch normalization
                activation,  # activation
                nn.Dropout(dropout),  # Dropout layer
            ]
            prev_dim = hdim

        # Final linear layer for classification
        layers.append(nn.Linear(prev_dim, num_classes))
        # Create the sequential model
        self.net = nn.Sequential(*layers)

        # Initialize weights
        for m in self.net:
            if isinstance(m, nn.Linear):
                kernel_init(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x = F.normalize(x, dim=-1)
        return self.net(x)


# --------------------------------------------------------------------------------------------
# 4. Custom Loss Functions (tested during development, more or less relic code now)
# --------------------------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """
    Focal Loss used for heavy class imbalance scenarios. During our testing phase, it did not outperform standard CrossEntropyLoss.
    But kept if needed in the future as part of our research.
    Based on: https://arxiv.org/pdf/1708.02002.pdf and classic implementations found online.
    __init__ args:
    - weight: class weights tensor, allowing for further balancing of classes
    - gamma: focusing parameter, higher values focus more on hard examples
    - reduction: reduction method, "mean", "sum" or "none"
    forward args:
    - input: model predictions (logits) [batch_size, num_classes]
    - target: true labels [batch_size]
    Returns:
    - focal loss value
    """

    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Ensure weights are on the same device as input
        if self.weight is not None and self.weight.device != input.device:
            self.weight = self.weight.to(input.device)

        # Calculate standard cross-entropy loss to add focal component
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction="none")
        # calculate predicted probabilities
        pt = torch.exp(-ce_loss)
        # Basic focal loss formula with gamma as focusing parameter
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # optional reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ArcFace(torch.nn.Module):
    """
    ArcFace margin-based loss function. Also for class imbalance scenarios. Even though it is used for face recognition tasks originally, we gave it a try for protein classification.
    Worked as good as CrossEntropyLoss in our tests, but not better. So it was ditched for now and kept as a relic.
    Based on: https://arxiv.org/abs/1801.07698
    __init__ args:
    - s: scaling factor for logits
    - margin: angular margin to add between classes
    forward args:
    - logits: model predictions (cosine similarities) [batch_size, num_classes]
    - labels: true labels [batch_size]
    Returns:
    - modified logits with ArcFace margin applied
    """

    def __init__(self, s=64.0, margin=0.5):

        # Initialize ArcFace with scaling factor and margin
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Normalize logits first
        logits = F.normalize(logits, dim=1)

        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            # Convert cosine to angle
            theta = target_logit.arccos()
            # Add margin
            theta_m = theta + self.margin
            # Convert back to cosine
            target_logit_with_margin = theta_m.cos()

            # Update logits
            logits[index, labels[index].view(-1)] = target_logit_with_margin

        # Scale
        logits = logits * self.s
        return logits


# --------------------------------------------------------------------------------------------
# 5. PyTorch-Lightning module
# --------------------------------------------------------------------------------------------


class LitClassifier(pl.LightningModule):
    """
    PyTorch-Lightning module for training a Feed-Forward Neural Network classifier on ESM embeddings.
    Used FFNClassifier as the model architecture and putting it together with all HP/Final model settings.
    __init__ args:
    - input_dim: dimension of input features automatically inferred based on ESM model selected. Given on to FFNClassifier.
    - hidden_dims: list of hidden layer dimensions for FFNClassifier.
    - num_classes: number of output classes used in the final classification layer of FFNClassifier.
    - optimizer_class: optimizer class to use for FFNClassifier training.
    - activation: activation function to use in FFNClassifier.
    - kernel_init: kernel initialization method for FFNClassifier.
    - lr: learning rate for optimizer, given from HP search/Final model settings.
    - weight_decay: weight decay (L2 regularization) for optimizer, given from HP search/Final model settings.
    - dropout: dropout rate for FFNClassifier, given from HP search/Final model settings.
    - class_weights: class weights tensor for handling class imbalance in loss func, given from dataset analysis.
    - domain_task: boolean flag indicating if the task is domain segmentation (uses Transformer model) or standard classification (uses FFNClassifier).
    - num_heads: number of attention heads for Transformer model if domain_task is True.
    - num_layers: number of layers for Transformer model if domain_task is True.
    """

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
        # Save hyperparameters by Lightning
        self.save_hyperparameters()

        # Store args
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.weight_decay = weight_decay
        self.domain_task = domain_task
        self.class_weights = class_weights
        self.num_classes = num_classes

        # base model creation for standard classification task HP search
        if domain_task is False:
            self.model = FFNClassifier(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout=dropout,
                activation=activation,
                kernel_init=kernel_init,
            )

            # Loss function setup
            if class_weights is not None:
                if RANK == 0:
                    print("USING CLASS WEIGHTS IN LOSS FUNCTION")
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                if RANK == 0:
                    print(
                        "WARNING: No class weights provided, using unweighted loss function"
                    )
                self.loss_fn = nn.CrossEntropyLoss()

            # Focal Loss setup
            # if class_weights is not None:
            #     # Use Focal Loss instead of CrossEntropyLoss
            #     self.loss_fn = FocalLoss(weight=class_weights, gamma=2.0)
            # else:
            #     print("WARNING: No class weights provided, using unweighted focal loss")
            #     self.loss_fn = FocalLoss(gamma=2.0)

            # ArcFace margin-based loss setup
            # self.ArcFace = ArcFace(s=30.0, margin=0.5)

            # if class_weights is not None:
            #     self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            # else:
            #     self.loss_fn = nn.CrossEntropyLoss()

        # Transformer model creation for domain boundary task HP search
        else:
            print("Domain boundary task activated - using Transformer model")
            # import Transformer class
            from DomainFinder import Transformer

            # create Transformer model, see DomainFinder.py for details
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

        # Precision and Recall metrics setup
        self.precision_metric = Precision(
            task="multiclass", num_classes=num_classes, average=None, ignore_index=None
        )
        self.recall_metric = Recall(
            task="multiclass", num_classes=num_classes, average=None, ignore_index=None
        )

    def fbeta(self, precision, recall, beta=1.5):
        """
        Calculate the F-beta score, used as a metric for domain boundary detection.
        with a beta of 1.5 to weight recall higher than precision. % more value on recall.
        NOT USED AS LOSS FUNCTION, JUST FOR METRICS DURING VALIDATION.
        """
        # Avoid division by zero, return 0 if both precision and recall are zero and the model is completely in the wrong
        # happens during early training stages
        if precision + recall == 0:
            return 0.0
        # F-beta formula, based on https://en.wikipedia.org/wiki/F-score
        return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    def distance_aware_boundary_loss(
        self,
        logits,
        labels,
        alpha=2.0,
        beta=1.0,
        gamma=0.5,
        fbeta_weight=1.5,
        epsilon=1e-8,
    ):
        """
        Custom loss function for domain boundary prediction.

        This loss combines three components:
        1. A base binary cross-entropy loss.
        2. A distance-aware penalty for false positives, which is higher for incorrect
           predictions closer to a true boundary.
        3. An F-beta loss to balance precision and recall.

        Args:
            logits: Model predictions (raw scores) of shape [seq_len, num_classes].
            labels: True labels of shape [seq_len], where 1 is a boundary and 0 is not.
            alpha: Weight for the distance penalty component.
            beta: Weight for the base cross-entropy loss component.
            gamma: Weight for the F-beta loss component.
            fbeta_weight: The beta value for the F-beta score calculation, weighting recall more than precision.
            epsilon: A small value to prevent division by zero for numerical stability.
        Returns:
            total_loss: The combined loss value. Used for further backpropagation.

        """
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        boundary_probs = probs[:, 1]  # Probabilities for boundary class
        binary_labels = labels.float()

        # Base classification loss (cross-entropy)
        ce_loss = -binary_labels * torch.log(boundary_probs + epsilon) - (
            1 - binary_labels
        ) * torch.log(1 - boundary_probs + epsilon)

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

                # Distance penalty: exponential growth based on distance
                # Farther false positives get higher penalties.
                # The '-1' ensures the penalty is 0 for a distance of 0.
                distance_penalty = (
                    torch.exp(min_distance / 10.0) - 1.0
                )  # Adjust denominator to control growth rate

                # Weight by prediction confidence
                fp_confidence = boundary_probs[fp_pos]
                weighted_penalty = distance_penalty * fp_confidence * alpha

                # Append to list
                distance_penalties.append(weighted_penalty)

            # convert list to tensor and average
            distance_penalty_term = torch.stack(distance_penalties).mean()
        else:
            # No false positives or no true boundaries, no distance penalty
            distance_penalty_term = torch.tensor(0.0, device=logits.device)

        # F-beta loss component
        tp_soft = (boundary_probs * binary_labels).sum()
        fp_soft = (boundary_probs * (1 - binary_labels)).sum()
        fn_soft = ((1 - boundary_probs) * binary_labels).sum()

        # Calculate soft precision and recall
        precision_soft = tp_soft / (tp_soft + fp_soft + epsilon)
        recall_soft = tp_soft / (tp_soft + fn_soft + epsilon)

        # Calculate soft F-beta score
        fbeta_score_soft = (
            (1 + fbeta_weight**2)
            * precision_soft
            * recall_soft
            / (fbeta_weight**2 * precision_soft + recall_soft + epsilon)
        )
        # basic F-beta conversion to loss
        fbeta_loss = 1.0 - fbeta_score_soft

        # Properly combine all loss components together with their weights
        total_loss = beta * ce_loss.mean() + distance_penalty_term + gamma * fbeta_loss
        return total_loss

    def forward(self, x):
        """
        Basic forward func calling the model and its input x.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Train step func for PyTorch-Lightning module. Everything that needs to do during a training step happens here.
        Args:
            batch: input batch from dataloader
            batch_idx: index of the batch (not used, old relic)
        Returns:
            loss: calculated training loss for backpropagation. gets forwarded from loss function.
        """

        # unpack batch into inputs and labels and move to device
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # Assert if NaN values are present in inputs or logits. Prevents silents falues leading to not performing models and metrics.
        assert not torch.isnan(x).any(), "Input contains NaN values"
        logits = self(x)
        assert not torch.isnan(logits).any(), "Logits contain NaN values"

        # Needed to transform logits if using ArcFace loss, more or less relic from testing ArcLoss
        # if self.loss_fn is ArcFace:
        # Apply ArcFace transformation
        # transformed_logits = self.ArcFace(logits, y)
        # logits = transformed_logits

        # Debug prints for shapes
        # print('SHAPES',logits.shape, y.shape)

        # Domain boundary task loss calculation, processing each sequence individually and averaging the loss to tackle variable sequence lengths.
        if self.domain_task is True:
            # get batch_size
            batch_size, _, _ = logits.shape
            # init loss list
            losses = []

            # loop through seqs in batch
            for i in range(batch_size):
                # get seq logits and labels
                seq_logits = logits[i]
                seq_labels = y[i]
                # calculate loss for seq
                seq_loss = self.distance_aware_boundary_loss(seq_logits, seq_labels)
                # append loss to list
                losses.append(seq_loss)
            # average losses
            loss = torch.stack(losses).mean()
        # Standard classification task loss calculation using the defined loss function (CrossEntropyLoss/FocalLoss/ArcLoss).
        else:
            loss = self.loss_fn(logits, y)

        # Check for NaN in loss to prevent failures during backpropagation
        assert not torch.isnan(loss), "Training loss is NaN"
        # Log training loss for progress bar
        self.log("train_loss", loss.detach(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step func for PyTorch-Lightning module. Everything that needs to do during a validation step happens here.
        Args:
            batch: input batch from dataloader
            batch_idx: index of the batch (used for debug prints)
        Returns:
            None, all needed is logging that is done directly here.
        """
        # unpack batch into inputs and labels and move to device
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # Assert if NaN values are present in inputs or logits. Prevents silents falues leading to not performing models and metrics.
        assert not torch.isnan(x).any(), "Input contains NaN values in validation"
        logits = self(x)
        assert not torch.isnan(logits).any(), "Logits contain NaN values in validation"

        # Needed to transform logits if using ArcFace loss, more or less relic from testing ArcLoss
        # if self.loss_fn is ArcFace:
        # Apply ArcFace transformation
        # transformed_logits = self.ArcFace(logits, y)
        # logits = transformed_logits

        # Domain boundary task loss calculation, processing each sequence individually and averaging the loss to tackle variable sequence lengths.
        if self.domain_task is True:
            # get batch_size
            batch_size, _, _ = logits.shape
            # init lists
            losses = []
            all_preds = []
            all_labels = []

            for i in range(batch_size):
                # get seq logits and labels
                seq_logits = logits[i]
                seq_labels = y[i]
                # calculate loss for seq with custom los
                seq_loss = self.distance_aware_boundary_loss(seq_logits, seq_labels)
                # append loss to list
                losses.append(seq_loss)

                # get predictions for seq
                seq_preds = torch.argmax(seq_logits, dim=-1)
                # append preds and labels to lists to calculate metrics later
                all_preds.append(seq_preds)
                all_labels.append(seq_labels)

            # average losses and concatenate all preds/labels to tensors
            loss = torch.stack(losses).mean()
            preds = torch.cat(all_preds)
            y_flat = torch.cat(all_labels)
        # Standard classification task loss calculation using the defined loss function (CrossEntropyLoss/FocalLoss/ArcLoss).
        else:
            # get loss
            loss = self.loss_fn(logits, y)
            # get predictions and labels
            preds = torch.argmax(logits, dim=-1)
            y_flat = y

        # Check for NaN in loss to prevent failures during logging
        assert not torch.isnan(loss), "Validation loss is NaN"

        # Debug prints to understand model predictions better during validation
        if RANK == 0:
            if batch_idx == 0:  # Only print for first batch to avoid spam
                # print(f"Batch {batch_idx}:")
                # print(f"Preds: {preds[:20]}")
                # print(f"Labels: {y_flat[:20]}")
                # print(f"Logits sample (first sequence, first 5 positions):\n{logits[0, :5, :]}")
                # print(f"Max logits: {logits.max()}, Min logits:\n{logits.min()}")
                # print(f"Class 0 logits mean: {logits[:, :, 0].mean()}")
                # print(f"Class 1 logits mean: {logits[:, :, 1].mean()}")
                # print(f"Loss function weights: {self.loss_fn.weight}")
                # print(f"Prediction distribution: {torch.bincount(preds)}")
                # print(f"True label distribution: {torch.bincount(y_flat)}")
                print("-" * 50)

        # Calculate accuracy and mean it
        acc = (preds == y_flat).float().mean()

        # Log validation loss and accuracy for progress bar
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

        # Update precision and recall metrics for the epoch, therefore each epoch still val end
        self.precision_metric(preds.detach(), y_flat.detach())
        self.recall_metric(preds.detach(), y_flat.detach())

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to compute and log all precision, recall, loss and accuracy metrics. Basically just for logging.
        Part of PyTorch-Lightning module structure.
        """

        # Compute precision and recall for all classes
        prec_all = self.precision_metric.compute()
        rec_all = self.recall_metric.compute()

        # Domain boundary task specific logging
        if self.domain_task is True:
            # Boundary precision and recall are getting added to a variable
            boundary_precision = (
                prec_all[1].item() if not torch.isnan(prec_all[1]) else 0.0
            )
            boundary_recall = rec_all[1].item() if not torch.isnan(rec_all[1]) else 0.0

            # Calculate F-beta score with beta=1.5 to weight recall higher than precision. Precision and recall are used
            fbeta = self.fbeta(boundary_precision, boundary_recall, beta=1.5)

            # Log the boundary precision, recall, and F-beta score
            self.log(
                "val_boundary_prec", boundary_precision, prog_bar=True, sync_dist=True
            )
            self.log("val_boundary_rec", boundary_recall, prog_bar=True, sync_dist=True)

            self.log("f_beta", fbeta, prog_bar=True, sync_dist=True)

        # Standard classification task logging
        else:
            # Log precision and recall for each class individually in a loop
            for class_idx in range(self.num_classes):
                # Log per-class metrics only if within computed range to avoid index errors
                if class_idx < len(prec_all):
                    # Handle NaN values by logging 0.0 if precision is NaN (no true positives for that class)
                    val_prec = (
                        prec_all[class_idx].item()
                        if not torch.isnan(prec_all[class_idx])
                        else 0.0  # was 1.0
                    )
                    # log to tensorboard
                    self.log(
                        f"val_prec_{class_idx}",
                        val_prec,
                        prog_bar=False,
                        sync_dist=True,
                    )

                if class_idx < len(rec_all):
                    # Handle NaN values by logging 0.0 if recall is NaN (no true positives for that class)
                    val_rec = (
                        rec_all[class_idx].item()
                        if not torch.isnan(rec_all[class_idx])
                        else 0.0  # was 1.0
                    )
                    # log to tensorboard
                    self.log(
                        f"val_rec_{class_idx}", val_rec, prog_bar=False, sync_dist=True
                    )

            # debug prints
            # if RANK == 0:
            #     print(f"Precision per class: {prec_all}")
            #     print(f"Recall per class: {rec_all}")
            #     print(f"NaN count in precision: {torch.isnan(prec_all).sum()}")

            # Log average metrics across all classes
            valid_prec = prec_all[~torch.isnan(prec_all)]  # Remove NaN values
            avg_prec = valid_prec.mean().item() if len(valid_prec) > 0 else 0.0
            valid_rec = rec_all[~torch.isnan(rec_all)]  # Remove NaN values
            avg_rec = valid_rec.mean().item() if len(valid_rec) > 0 else 0.0

            # log average metrics to tensorboard
            self.log("val_prec_avg", avg_prec, prog_bar=True, sync_dist=True)
            self.log("val_rec_avg", avg_rec, prog_bar=True, sync_dist=True)

        # Reset metrics for next epoch
        self.precision_metric.reset()
        self.recall_metric.reset()

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers for training. Also a part of PyTorch-Lightning module structure.
        Return:
            - dict containing optimizer and lr_scheduler configurations used by Lightning during training.
        """

        # Optimizer setup based on selected optimizer class and its specific parameters
        # SGD with Nesterov momentum
        if self.optimizer_class == torch.optim.SGD:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,  # pass learning rate
                nesterov=True,  # Use Nesterov momentum for faster convergence
                momentum=0.9,  # Add momentum for SGD
                dampening=0,  # Set dampening to 0 for Nesterov
            )
        # Adam variants and setup with weight decay
        elif self.optimizer_class in [
            torch.optim.Adam,
            torch.optim.AdamW,
            torch.optim.NAdam,
        ]:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        # RMSprop setup with weight decay
        elif self.optimizer_class == torch.optim.RMSprop:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        # Default optimizer setup if none of the above match, should never happen
        else:
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        # Learning rate scheduler setup to reduce LR on plateau of validation loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,  # optimizer to adjust
            mode="min",  # minimize validation loss
            factor=0.5,  # reduce LR by half
            patience=3,  # wait for 3 epochs without improvement
        )

        # return optimizer and scheduler in Lightning format
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# --------------------------------------------------------------------------------------------
# 6. Helper function for main_HP to perform hyperparameter optimization and final model training
# --------------------------------------------------------------------------------------------

# -------------------------
# 6.1 Dataset and DataLoader setup
# -------------------------


# Create separate Datasets for train and validation data from H5 file
class TrainDataset(torch.utils.data.Dataset):
    """
    TrainDataset for reading in the h5 file made from the cached embeddings.
    Handles multiple chunks stored as separate datasets in the H5 file.
    As well as reads in each rank done by the distributed dataloader (each gpu used)
    Args:
        h5_file: path to the H5 file containing cached embeddings and labels.
    """

    def __init__(self, h5_file):
        """
        Init to read in the H5 file and prepare indexing for chunks.
        sorts the embedding keys and calculates cumulative sizes for indexing as well as length.
        """
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
        # previously calculated length
        return self.length

    def __getitem__(self, idx):
        """
        getitem function to read in the correct chunk and local index based on global index.
        Args:
            idx: global index to retrieve sample from.
        """
        # Open H5 file and retrieve the correct chunk and local index
        with h5py.File(self.h5_file, "r") as f:
            # find chunk index using cumulative sizes
            chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")

            # find local index within the chunk
            local_idx = (
                idx - self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else idx
            )

            # get embedding and label keys for the chunk
            embedding_key = self.embedding_keys[chunk_idx]
            labels_key = embedding_key.replace("train_embeddings_", "train_labels_")

            # read in embeddings and labels as tensors
            embeddings = torch.tensor(f[embedding_key][local_idx], dtype=torch.float32)
            labels = torch.tensor(f[labels_key][local_idx], dtype=torch.long)

            return embeddings, labels


class ValDataset(torch.utils.data.Dataset):
    """
    ValDataset for reading in the h5 file made from the cached embeddings.
    Handles multiple chunks stored as separate datasets in the H5 file.
    As well as reads in each rank done by the distributed dataloader (each gpu used)
    Args:
        h5_file: path to the H5 file containing cached embeddings and labels.
    Basically the same as TrainDataset but for validation data.
    """

    def __init__(self, h5_file):
        """
        Init to read in the H5 file and prepare indexing for chunks.
        sorts the embedding keys and calculates cumulative sizes for indexing as well as length.
        """
        self.h5_file = h5_file

        # Open H5 file and retrieve embedding keys and sizes
        with h5py.File(self.h5_file, "r") as f:
            # Get validation embedding keys
            self.embedding_keys = sorted(
                [k for k in f.keys() if k.startswith("val_embeddings_")]
            )
            # Get chunk sizes and cumulative sizes
            self.chunk_sizes = [f[key].shape[0] for key in self.embedding_keys]
            self.cumulative_sizes = np.cumsum(self.chunk_sizes)
            # calculate total length
            self.length = (
                self.cumulative_sizes[-1] if self.cumulative_sizes.size > 0 else 0
            )

    def __len__(self):
        # previously calculated length
        return self.length

    def __getitem__(self, idx):
        """
        getitem function to read in the correct chunk and local index based on global index.
        Args:
            idx: global index to retrieve sample from.
        """
        # Open H5 file and retrieve the correct chunk and local index
        with h5py.File(self.h5_file, "r") as f:
            # find chunk index using cumulative sizes
            chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")

            # find local index within the chunk
            local_idx = (
                idx - self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else idx
            )

            # get embedding and label keys for the chunk
            embedding_key = self.embedding_keys[chunk_idx]
            labels_key = embedding_key.replace("val_embeddings_", "val_labels_")

            # read in embeddings and labels as tensors
            embeddings = torch.tensor(f[embedding_key][local_idx], dtype=torch.float32)
            labels = torch.tensor(f[labels_key][local_idx], dtype=torch.long)

            return embeddings, labels


# -------------------------
# 7.2 Optuna objective function for HP search
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
    """
    Optuna objective function for hyperparameter optimization. Used to define HP search space and train/evaluate the model with suggested HPs.
    Used for both domain boundary detection task (Transformer model) and standard classification task (FFN model).
    Func is splitted into two parts based on the domain_task flag.

    Choose HP settings
    Build model with suggested HPs
    Fit model for EPOCHS, with given checkpoints and early stopping
    Evaluate model on validation set and return validation loss for Optuna to optimize.

    Args:
        trial: Optuna trial object for suggesting hyperparameters. Optuna handles this internally.
        input_dim: dimension of input features automatically inferred based on ESM model selected.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        weights: class weights tensor for handling class imbalance in loss func, given from dataset analysis.
        domain_task: boolean flag indicating if the task is domain segmentation (uses Transformer model) or standard classification (uses FFNClassifier).
        EPOCHS: number of epochs to train the model for each trial.
    Returns:
        - best validation loss achieved during training with the suggested hyperparameters.
        - val_precision_avg: average validation precision across all classes (only for standard classification task).
    """
    # Standard classification task, HPs for FFN model
    if domain_task is False:
        # only RANK 0 suggests HPs, then broadcast to other ranks via dictionary
        if RANK == 0:
            # n neurons
            n_neurons = trial.suggest_int("num_neurons", 640, 1440, step=200)
            # hidden layers
            hidden_dims = [
                n_neurons
                for _ in range(trial.suggest_int("num_hidden_layers", 1, 3, step=1))
            ]

            # dropout rate, learning rate, weight decay
            drop = trial.suggest_float("dropout", 0.05, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            wd = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

            # optimizer selection
            optimizer = trial.suggest_categorical(
                "optimizer", ["adam", "adamw", "sgd", "rmsprop", "nadam"]
            )
            if optimizer in ["adam", "adamw", "nadam"]:
                if optimizer == "adam":  # Adam optimizer
                    optimizer_class = torch.optim.Adam
                elif optimizer == "adamw":  # AdamW optimizer
                    optimizer_class = torch.optim.AdamW
                else:  # nadam
                    optimizer_class = torch.optim.NAdam
            elif optimizer == "sgd":  # SGD optimizer
                optimizer_class = torch.optim.SGD
            else:  # rmsprop
                optimizer_class = torch.optim.RMSprop

            # activation function selection
            activation = trial.suggest_categorical(
                "activation", ["relu", "gelu", "leaky_relu"]
            )
            if activation == "relu":
                activation = nn.ReLU(inplace=True)
                # use kaiming init for relu
                kernel_init = nn.init.kaiming_normal_
            elif activation == "gelu":
                activation = nn.GELU()
                # use xavier init for gelu
                kernel_init = nn.init.xavier_normal_
            elif activation == "leaky_relu":
                activation = nn.LeakyReLU(inplace=True)
                # use kaiming init for leaky relu
                kernel_init = nn.init.kaiming_normal_
            else:
                activation = nn.SiLU(inplace=True)
                # use xavier init for silu
                kernel_init = nn.init.xavier_normal_

            # print trial settings
            if RANK == 0:
                print(
                    f"\n=== Trial {trial.number} ===\n"
                    f"Hidden layers: {len(hidden_dims)}, Neurons: {n_neurons}\n"
                    f"Dropout: {drop:.3f}, LR: {lr:.6f}, WD: {wd:.6f}\n"
                    f"Optimizer: {optimizer}, Activation: {activation.__class__.__name__}\n"
                    f"========================\n"
                )

            # save params to dict for broadcasting
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

        # other ranks set params to None for broadcasting
        else:
            params = None

        # wrap params in list
        params_list = [params]
        # broadcast params to other ranks if in distributed mode
        if dist.is_initialized():
            # Broadcast the parameters to all ranks
            dist.broadcast_object_list(params_list, src=0)
        # get params from list for all ranks
        params = params_list[0]

        # build model with suggested HPs
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

    # Domain boundary detection task, HPs for Transformer model
    else:
        # only RANK 0 suggests HPs, then broadcast to other ranks via dictionary
        if RANK == 0:
            # dimensions model and number of heads
            d_model = trial.suggest_categorical("d_model", [256, 512, 768, 1024])
            n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])

            # ensure d_model is divisible by n_heads. keep choosing n_heads until valid
            while d_model % n_heads != 0:
                n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])

            # number of layers and feed-forward dimension
            n_layers = trial.suggest_int("n_layers", 2, 4, step=2)
            d_ff = 4 * d_model
            # max sequence length for positional encoding
            max_seq_len = trial.suggest_int("max_seq_len", 100, 1000, step=100)

            # dropout rates, dropout rate for attention layers, learning rate, weight decay
            drop = trial.suggest_float("dropout", 0.1, 0.5)
            drop_attn = trial.suggest_float("dropout_attn", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
            wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

            # optimizer selection
            optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
            if optimizer == "adam":
                optimizer_class = torch.optim.Adam
            elif optimizer == "adamw":
                optimizer_class = torch.optim.AdamW
            else:
                optimizer_class = torch.optim.SGD

            # activation function selection
            activation = trial.suggest_categorical(
                "activation", ["relu", "gelu", "leaky_relu"]
            )
            if activation == "relu":
                activation = nn.ReLU(inplace=True)
                # use kaiming init for relu
                kernel_init = nn.init.kaiming_normal_
            elif activation == "gelu":
                activation = nn.GELU()
                # use xavier init for gelu
                kernel_init = nn.init.xavier_normal_
            else:
                activation = nn.LeakyReLU(inplace=True)
                # use kaiming init for leaky relu
                kernel_init = nn.init.kaiming_normal_

            # save params to dict for broadcasting
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

        # print trial settings
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

        # other ranks set params to None for broadcasting
        else:
            params = None

        # wrap params in list
        params_list = [params]
        # broadcast params to other ranks if in distributed mode
        dist.broadcast_object_list(params_list, src=0)
        # get params from list for all ranks
        params = params_list[0]

        # build model with suggested HPs
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

    # setup EarlyStopping and ModelCheckpoint callbacks
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, mode="min", verbose=True, min_delta=0.01
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    # setup PyTorch-Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,  # set max epochs
        accelerator="gpu",  # use GPU acceleration
        devices=-1
        if dist.is_initialized()
        else 1,  # use all available GPUs if in distributed mode
        strategy="ddp"
        if dist.is_initialized()
        else "auto",  # use DDP if in distributed mode
        enable_progress_bar=True,  # enable progress bar
        callbacks=[early_stop, checkpoint_callback],  # add callbacks
        logger=TensorBoardLogger(
            save_dir=f"/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/{PROJECT_NAME}/tensorboard",
            name=f"optuna_trial_{trial.number}",
        ),  # TensorBoard logger
        # limit_train_batches=250, # used for debug
        # limit_val_batches=250,   # used for debug
        # use_distributed_sampler=False,
    )

    # Log hyperparameters to TensorBoard
    if RANK == 0:
        # ... for standard classification task
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
        # ... for domain boundary detection task
        else:
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

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Log final metrics as hp_metric
    best_val_loss = checkpoint_callback.best_model_score.item()

    # get val precision average from logged metrics and gather it
    val_precision = trainer.logged_metrics.get("val_prec_avg", 0.0)
    if hasattr(val_precision, "item"):
        val_precision = val_precision.item()

    # print trial results & log to tensorboard
    if RANK == 0:
        print(
            f"Trial {trial.number}: val_loss={best_val_loss:.4f}, val_precision={val_precision:.4f}"
        )
        trainer.logger.experiment.add_scalar("hp_metric_loss", best_val_loss, 0)
        trainer.logger.experiment.add_scalar("hp_metric_precision", val_precision, 0)

    # return best val loss and val precision for standard classification task
    if domain_task is False:
        return best_val_loss, val_precision
    else:
        return best_val_loss


# -------------------------
# 7.3 Class distribution checker and weights calculation
# -------------------------


def class_distribution():
    """
    Analyze class distribution in training and validation datasets from cached H5 file.
    Calculate class weights to handle class imbalance during training.
    Returns:
        - weights: torch tensor containing class weights for each class. Used during loss calculation for training.
    """
    # Load labels from H5 cache file split in train_labels_ and val_labels_ datasets
    with h5py.File(CACHE_PATH, "r") as f:
        train_labels = np.concatenate(
            [f[key][:] for key in f.keys() if key.startswith("train_labels_")]
        )
        val_labels = np.concatenate(
            [f[key][:] for key in f.keys() if key.startswith("val_labels_")]
        )

    # Calculate class counts for train and validation sets
    # i.e. how many samples are per class
    train_class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    val_class_counts = np.bincount(val_labels, minlength=NUM_CLASSES)

    # total class counts and samples
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

    # Print class distribution and weights split by train, val and combined sets to gather insights
    if RANK == 0:
        print("Training set label distribution:")
        for i in range(NUM_CLASSES):
            print(
                f"  Class {i}: {train_class_counts[i]} samples ({train_class_counts[i] / len(train_labels) * 100:.2f}%)"
            )
        print(f"  Total training samples: {len(train_labels)}")
        print("Validation set label distribution:")
        for i in range(NUM_CLASSES):
            print(
                f"  Class {i}: {val_class_counts[i]} samples ({val_class_counts[i] / len(val_labels) * 100:.2f}%)"
            )
        print(f"  Total validation samples: {len(val_labels)}")
        print("Combined dataset label distribution:")
        for i in range(NUM_CLASSES):
            print(
                f"  Class {i}: {total_class_counts[i]} samples ({total_class_counts[i] / total_samples * 100:.2f}%)"
            )
        print(f"  Total combined samples: {total_samples}")
        print("Class weights calculated:", weights.cpu().numpy())

    return weights


# -------------------------
# 7.4 Optuna study setup and execution
# -------------------------


def run_optuna_study(train_dataset, train_loader, val_loader, weights):
    """
    Setup and run Optuna hyperparameter optimization study.

    Creates or loads an existing Optuna study for multi-objective optimization,
    ensuring proper synchronization across distributed processes. Optimizes the
    objective function for the specified number of trials.

    Args:
        train_dataset: Training dataset used to infer input dimensions
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        weights: Class weights tensor for handling class imbalance

    Returns:
        study: Optuna study object containing all trial results
        input_dim: Input dimension for model building
    """
    # only RANK 0 creates the study, other ranks load it
    if RANK == 0:
        print("Starting Optuna hyperparameter search...")
        study = optuna.create_study(
            directions=[
                "minimize",
                "maximize",
            ],  # multi-objective: minimize val_loss, maximize val_precision
            storage=f"sqlite:////global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/{PROJECT_NAME}/optuna_study.db",  # SQLite storage path
            load_if_exists=True,  # load existing study if it exists
            study_name=PROJECT_NAME,  # name of the study
            # overwrite=True,               # overwrite existing study (to quickly wipe HP progress)
        )
    # synchronize all ranks before starting the study
    if dist.is_initialized():
        dist.barrier()

    # other ranks load the existing study, with the same name and storage
    if RANK != 0:
        study = optuna.create_study(
            directions=[
                "minimize",
                "maximize",
            ],  # multi-objective: minimize val_loss, maximize val_precision
            storage=f"sqlite:////global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/{PROJECT_NAME}/optuna_study.db",  # SQLite storage path
            load_if_exists=True,  # load existing study if it exists
            study_name=PROJECT_NAME,  # name of the study
            # overwrite=True,               # overwrite existing study (to quickly wipe HP progress)
        )

    # Gather input dimension from dataset subset for model building
    sample_embedding, _ = train_dataset[0]
    input_dim = sample_embedding.shape[0]

    def objective_wrapper(trial):
        """
        Wrapper for the objective function to include additional fixed arguments.
        """
        return objective(
            trial,
            input_dim,
            train_loader,
            val_loader,
            weights,
            domain_task=False,
            EPOCHS=EPOCHS,
        )

    # Start the optimization process with objective wrapper function
    study.optimize(
        objective_wrapper,
        n_trials=STUDY_N_TRIALS,
    )

    return study, input_dim


# -------------------------
# 7.5 Find and broadcast best trial from Optuna study
# -------------------------


def find_and_broadcast_best_trial(study):
    """
    Find the best trial from Optuna study using Pareto optimality and broadcast to all ranks.

    For multi-objective optimization, this function:
    1. Identifies Pareto optimal trials (non-dominated solutions)
    2. Selects the trial with highest precision among Pareto optimal trials
    3. Broadcasts the selected trial to all distributed processes

    Args:
        study: Optuna study object containing completed trials

    Returns:
        best_trial: The selected best trial (Optuna Trial object)

    Raises:
        RuntimeError: If no completed trials are found in the study
    """
    if RANK == 0:
        # print optimization summary
        print("Optimization complete!")
        print(f"Number of trials: {len(study.trials)}")

        # For multi-objective, there's no single "best" trial
        # Get Pareto optimal trials
        pareto_trials = []
        # loop through trials
        for trial in study.trials:
            # only consider completed trials
            if trial.state == optuna.trial.TrialState.COMPLETE:
                # bool flag
                is_dominated = False
                # check if trial is dominated by any other trial
                for other_trial in study.trials:
                    # only consider completed trials
                    if (
                        other_trial.state == optuna.trial.TrialState.COMPLETE
                        and other_trial != trial
                    ):
                        # check domination conditions
                        if (
                            other_trial.values[0] <= trial.values[0]  # loss
                            and other_trial.values[1] >= trial.values[1]  # precision
                            and (
                                other_trial.values[0] < trial.values[0]
                                or other_trial.values[1] > trial.values[1]
                            )
                        ):  # flag if dominated
                            is_dominated = True
                            break
                # add non-dominated trial to pareto list
                if not is_dominated:
                    pareto_trials.append(trial)

        # print pareto optimal trials
        print(f"Pareto optimal trials: {len(pareto_trials)}")
        for i, trial in enumerate(pareto_trials):
            print(
                f"  Trial {trial.number}: loss={trial.values[0]:.4f}, precision={trial.values[1]:.4f}"
            )

        # Choose the trial with best precision among Pareto optimal
        if pareto_trials:
            best_trial = max(pareto_trials, key=lambda t: t.values[1])
            # verify if best_trial has the expected params and is not empty
            if "num_neurons" in best_trial.params:
                print(
                    f"Selected trial {best_trial.number} (highest precision): loss={best_trial.values[0]:.4f}, precision={best_trial.values[1]:.4f}"
                )
            # if empty or missing params, search nearby trials that get created due to multi gpu usage
            else:
                for i in range(1, 10):
                    print("No best trial found, searching numbers close to it...")
                    # try next higher trial number
                    best_trial = study.trials[best_trial.number + i]
                    # check again for expected params
                    if "num_neurons" not in best_trial.params:
                        # else try next lower trial number
                        best_trial = study.trials[best_trial.number - i]
                    # if found, break
                    if "num_neurons" in best_trial.params:
                        print(
                            f"Selected trial {best_trial.number} (Found actual trial): loss={best_trial.values[0]:.4f}, precision={best_trial.values[1]:.4f}"
                        )
                        break
        # Fallback if no Pareto trials found
        else:
            # gather all completed trials
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            # select trial with best precision
            if completed_trials:
                best_trial = max(
                    completed_trials,
                    key=lambda t: t.values[1] if len(t.values) > 1 else 0,
                )
                print(f"Fallback: Selected trial {best_trial.number}")
            else:
                print("No completed trials found!")
                best_trial = None

    # RANK != 0 processes: best_trial is None for now
    else:
        best_trial = None

    # Broadcast the best_trial to all ranks
    if dist.is_initialized():
        # Create a list to broadcast the best trial information
        best_trial_data = [None]

        # Send the trial number which can be used to reconstruct the trial
        if RANK == 0 and best_trial is not None:
            best_trial_data[0] = best_trial.number

        # Broadcast the best trial number to all ranks
        dist.broadcast_object_list(best_trial_data, src=0)

        # Reconstruct the best_trial on non-rank 0 processes
        if RANK != 0 and best_trial_data[0] is not None:
            trial_number = best_trial_data[0]
            # Find the trial with the matching number
            for trial in study.trials:
                if trial.number == trial_number:
                    best_trial = trial
                    break
        # fallback
        elif best_trial_data[0] is None:
            best_trial = None

    # Final fallback if best_trial is still None
    if best_trial is None:
        if len(study.trials) > 0:
            # Use the last completed trial as fallback
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if completed_trials:
                best_trial = completed_trials[-1]
                if RANK == 0:
                    print(f"Using fallback trial {best_trial.number}")
            else:
                raise RuntimeError("No completed trials found in the study!")
        else:
            raise RuntimeError("No trials found in the study!")

    return best_trial


# -------------------------
# 7.6.  Load best model after HP search for final training
# -------------------------


def load_best_model(trial, input_dim, weights, domain_task=False):
    """
    Load the best model archetecture after hyperparameter optimization for final training or evaluation.
    Args:
        trial: Optuna trial object containing the best hyperparameters found during optimization.
        input_dim: dimension of input features automatically inferred based on ESM model selected.
        weights: class weights tensor for handling class imbalance in loss func, given from dataset analysis.
        domain_task: boolean flag indicating if the task is domain segmentation (uses Transformer model) or standard classification (uses FFNClassifier).
    Returns:
        - model: LitClassifier model instance initialized with the best hyperparameters.
    """
    # domain classification task
    if domain_task is False:
        # print which mode
        if RANK == 0:
            print("DOMAIN CLASSIFICATION TASK:", domain_task)
        # get hyperparameters from best trial
        p = trial.params

        # index each hyperparameter from trial params
        n_neurons = p["num_neurons"]
        n_layers = p["num_hidden_layers"]
        hidden_dims = [n_neurons] * n_layers

        # optimizer gathering
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

        # activation function & init gathering
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

        # dropout, lr, wd gathering
        drop = p["dropout"]
        lr = p["lr"]
        wd = p["weight_decay"]

        # build model with best hyperparameters
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

        # print final model settings for Final training
        if RANK == 0:
            print(
                f"\n=== FINAL TRAINING WITH {trial.number} ===\n"
                f"Hidden layers: {n_layers}, Neurons: {n_neurons}\n"
                f"Dropout: {drop:.3f}, LR: {lr:.6f}, WD: {wd:.6f}\n"
                f"Optimizer: {optimizer_class.__name__}, Activation: {activation_fn.__class__.__name__}\n"
                f"========================\n"
            )

    # domain boundary detection task
    else:
        # gather hyperparameters from best trial
        p = trial.params

        # index each hyperparameter from trial params
        d_model = p["d_model"]
        n_heads = p["n_heads"]
        n_layers = p["n_layers"]

        # optimizer gathering
        opt_name = p["optimizer"]
        if opt_name == "adam":
            optimizer_class = torch.optim.Adam
        elif opt_name == "adamw":
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.SGD

        # activation function & init gathering
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

        # dropout, lr, wd gathering
        drop = p["dropout"]
        lr = p["lr"]
        wd = p["weight_decay"]

        # build model with best hyperparameters
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

        # print final model settings for Final training
        if RANK == 0:
            print(
                f"\n=== FINAL TRAINING WITH {trial.number} ===\n"
                f"Hidden layers: {n_layers}, Hidden dim: {d_model}, n Heads: {n_heads}\n"
                f"Dropout: {drop:.3f}, LR: {lr:.6f}, WD: {wd:.6f}\n"
                f"Optimizer: {optimizer_class.__name__}, Activation: {activation_fn.__class__.__name__}\n"
                f"========================\n"
            )

    return model


# -------------------------
# 7.7 Final training with best hyperparameters
# -------------------------


def final_training(lit_model, train_loader, val_loader):
    """
    Perform final training with the best hyperparameters found during HP search.

    Sets up callbacks, trainer, and trains the model with the best configuration.
    Saves the final trained model to disk.

    Args:
        lit_model: LitClassifier model instance initialized with best hyperparameters
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data

    Returns:
        trainer: PyTorch Lightning Trainer object after training
        final_model_path: Path where the final model was saved
    """
    # Setup callbacks for early stopping and model checkpointing
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, mode="min", verbose=True, min_delta=0.01
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    if RANK == 0:
        print("Lit model created")

    # Setup PyTorch-Lightning Trainer for final training
    trainer = pl.Trainer(
        max_epochs=EPOCHS,  # set max epochs
        accelerator="gpu",  # use GPU acceleration
        devices=-1
        if dist.is_initialized()
        else 1,  # use all available GPUs if in distributed mode
        strategy="ddp"
        if dist.is_initialized()
        else "auto",  # use DDP if in distributed mode
        enable_progress_bar=True,  # enable progress bar
        callbacks=[early_stop, checkpoint_callback],  # add callbacks
        logger=TensorBoardLogger(  # TensorBoard logger for final training
            save_dir=f"/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/{PROJECT_NAME}/tensorboard",
            name=f"{PROJECT_NAME}_final",
        ),
        # limit_train_batches=250,  # used for debug
        # limit_val_batches=250,    # used for debug
        use_distributed_sampler=False,  # disable distributed sampler if already handled
    )

    if RANK == 0:
        print("Trainer created")

    # Train the model with best hyperparameters
    trainer.fit(lit_model, train_loader, val_loader)

    # Define path for saving the final model
    final_model_path = f"/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/{PROJECT_NAME}.pt"

    if RANK == 0:
        print(
            f"\nTraining complete, saving final model under: {final_model_path} ...\n"
        )

    # Save the final model (skip if using Transformer model)
    if (
        hasattr(lit_model, "model")
        and hasattr(lit_model.model, "__class__")
        and "Transformer" in lit_model.model.__class__.__name__
    ):
        # Don't save Transformer models (special handling needed)
        if RANK == 0:
            print("Transformer model detected, skipping save...")
    else:
        # Create directory if it doesn't exist and save the model
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(lit_model, final_model_path)

        if RANK == 0:
            print(f"Model saved successfully to {final_model_path}")

    return trainer, final_model_path


# --------------------------------------------------------------------------------------------
# 8. Helper Func for Main for usage
# --------------------------------------------------------------------------------------------

# -------------------------
# 8.1. Dataset and DataLoader for inference using saved embeddings
# -------------------------


class ClassifierDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading embeddings from H5 file for inference, i.e. usage mode.
    Labels are not returned, as they are only dummies in this case and not needed.
    Args:
        h5_file: path to H5 file containing saved embeddings.
    Returns:
        - embeddings: torch tensor of embeddings for given index.
    """

    def __init__(self, h5_file):
        self.h5_file = h5_file

        with h5py.File(self.h5_file, "r") as f:
            # gather all embedding keys in sorted order
            self.embedding_keys = sorted(
                [
                    k
                    for k in f.keys()
                    if k.startswith("embeddings_") or k.startswith("train_embeddings_")
                ]
            )
            # calculate chunk sizes and cumulative sizes for indexing
            self.chunk_sizes = [f[key].shape[0] for key in self.embedding_keys]
            # compute cumulative sizes for indexing
            self.cumulative_sizes = np.cumsum(self.chunk_sizes)
            # total length of dataset
            self.length = (
                self.cumulative_sizes[-1] if self.cumulative_sizes.size > 0 else 0
            )

    def __len__(self):
        # return length precomputed in __init__
        return self.length

    def __getitem__(self, idx):
        # open H5 file and retrieve embedding for given index
        with h5py.File(self.h5_file, "r") as f:
            # determine which chunk the index belongs to
            chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")
            # calculate local index within the chunk
            local_idx = (
                idx - self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else idx
            )
            # get the corresponding embedding key
            embedding_key = self.embedding_keys[chunk_idx]
            embeddings = torch.tensor(f[embedding_key][local_idx], dtype=torch.float32)
            
            # Only return embeddings, no labels
            return embeddings 


# -------------------------
# 8.2. Loader function to get model and dataloader for inference
# -------------------------


def loader(csv_path):
    """
    Load in embeddings for classifciation and the best model for inference. Usage mode func.
    Args:
        csv_path: path to CSV file containing sequences to be used.
    Returns:
        - model: trained LitClassifier model loaded from disk for inference.
        - classifier_loader: DataLoader for inference using ClassifierDataset.  
    """
    # Generate embeddings if not already present
    if not os.path.exists(
        "/global/scratch2/sapelt/tempTest/embeddings/embeddings_domain_classifier.h5"
    ):
        ESMDataset(
            esm_model=ESM_MODEL,                # the ESM model to use
            FSDP_used=False,                    # whether FSDP is used or not (unstable!)
            domain_boundary_detection=False,    # no domain boundary detection here
            training=True,                      # training mode true here, usage mode should turn on when finding missing columns
            num_classes=NUM_CLASSES,            # number of classes for classification task
            csv_path=csv_path,                  # path to CSV file with sequences to be tested
            category_col=CATEGORY_COL,          # name of the column with category labels
            sequence_col=SEQUENCE_COL,          # name of the column with sequences
        )

    # Create inference dataset
    classifier_dataset = ClassifierDataset(
        "/global/scratch2/sapelt/tempTest/embeddings/embeddings_domain_classifier.h5"
    )

    # Create DataLoader for inference
    classifier_loader = DataLoader(
        classifier_dataset,             # create DataLoader from dataset
        batch_size=BATCH_SIZE,          # batch size for inference from global
        shuffle=False,                  # no shuffling for inference, IMPORTANT TO KEEP INDEX!
        persistent_workers=True,        # keep workers alive for faster loading
        num_workers=NUM_WORKERS_EMB,    # number of workers for data loading
        pin_memory=True,                # pin memory for faster transfer to GPU
        prefetch_factor=4,              # prefetch factor for faster loading
    )

    # Load the best model with trained weights
    model = torch.load(
        "/global/research/students/sapelt/Masters/MasterThesis/models/Optuna_1000d_uncut_t33.pt",
        weights_only=False,
    )
    # set model to eval mode and move to device
    model.to(DEVICE).eval()

    # debug prints
    # if RANK == 0:
        # print("Model loaded and set to eval mode")
        # print(len(classifier_loader), "batches in classifier_loader")

    return model, classifier_loader


# -------------------------
# 8.3. Prediction function using model and dataloader
# -------------------------


def predictor(model, classifier_loader):
    """
    Predictor for inference using trained model and dataloader. Retrieves predictions for all samples.
    Args:
        model: trained LitClassifier model for inference.
        classifier_loader: DataLoader for inference using ClassifierDataset.
    Returns:
        - all_predictions: list of predicted class indices for all samples. Softmaxed to get class labels.
        - all_predictions_raw: list of raw prediction scores for all samples.
    """

    # initialize lists to store predictions
    all_predictions = []
    all_predictions_raw = []

    # only inference mode, no gradients needed
    with torch.inference_mode():
        # loop through batches in dataloader
        for batch in classifier_loader:
            # move inputs to device, put through model
            inputs = batch
            inputs = inputs.to(DEVICE)
            output = model(inputs)

            # Get predictions and raw scores
            preds_raw, preds = torch.max(output, dim=1)

            # append to lists
            all_predictions.extend(preds.cpu().numpy())
            all_predictions_raw.extend(preds_raw.cpu().numpy())

    return all_predictions, all_predictions_raw




# --------------------------------------------------------------------------------------------
# 9. Argument parser for command line usage
# --------------------------------------------------------------------------------------------


def parse_args():
    """
    Parse command line arguments. Used to switch between HP search and usage mode.
    Returns:
        args:
        - args.csv_path: path to input CSV file
        - args.HP_mode: boolean flag indicating if HP search mode should be used  
    """

    parser = argparse.ArgumentParser(description="ESM Embeddings HP Search")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/RemainingEntriesCompleteProteins.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--HP_mode", action="store_true", help="Use to run hyperparameter optimization"
    )

    return parser.parse_args()


# --------------------------------------------------------------------------------------------
# 10. Main for HP search and final training
# --------------------------------------------------------------------------------------------


def main_HP(Final_training=False):
    """
    Main function for HP search and final training.
    Sets up datasets, dataloaders, performs HP search with Optuna, and final training with best hyperparameters to save a final best performing model.
    Args:
        Final_training: boolean flag indicating if final training with best hyperparameters should be performed after HP search or not.
    """

    # Create necessary directories if they don't exist
    os.makedirs(
        "/global/research/students/sapelt/Masters/MasterThesis/pickle", exist_ok=True
    )
    os.makedirs(
        f"/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/{PROJECT_NAME}",
        exist_ok=True,
    )
    os.makedirs(
        "/global/research/students/sapelt/Masters/MasterThesis/models", exist_ok=True
    )
    if os.path.exists(
        f"/global/research/students/sapelt/Masters/MasterThesis/temp/progress_{NUM_CLASSES}.txt"
    ):
        
    # -------------------------
    # 10.1. Embedding generation
    # -------------------------

        # Generation of Embeddings
        # first check status file to see if all chunks were processed
        with open(
            f"/global/research/students/sapelt/Masters/MasterThesis/temp/progress_{NUM_CLASSES}.txt",
            "r",
        ) as status_file:
            # If not find final tag "All chunks processed. Exiting.", start embedding generation again
            if "All chunks processed. Exiting." not in status_file.read():
                ESMDataset(
                    esm_model=ESM_MODEL,  # the ESM model to use
                    FSDP_used=False,  # whether FSDP is used or not (unstable!)
                    domain_boundary_detection=False,  # whether domain boundary detection is used
                    training=True,  # whether in training mode (True) or eval mode (False)
                    num_classes=NUM_CLASSES,  # number of classes for classification task
                    csv_path=CSV_PATH,  # path to CSV file with sequences and labels
                    category_col=CATEGORY_COL,  # name of the column with category labels
                    sequence_col=SEQUENCE_COL,  # name of the column with sequences
                )
                pass
    # if status file does not exist, start embedding generation from scratch
    elif not os.path.exists(CACHE_PATH):
        ESMDataset(
            esm_model=ESM_MODEL,  # the ESM model to use
            FSDP_used=False,  # whether FSDP is used or not (unstable!)
            domain_boundary_detection=False,  # whether domain boundary detection is used
            training=True,  # whether in training mode (True) or eval mode (False)
            num_classes=NUM_CLASSES,  # number of classes for classification task
            csv_path=CSV_PATH,  # path to CSV file with sequences and labels
            category_col=CATEGORY_COL,  # name of the column with category labels
            sequence_col=SEQUENCE_COL,  # name of the column with sequences
        )

    # -------------------------
    # 10.2. Dataset and DataLoader setup
    # -------------------------

    # Create train and validation datasets
    train_dataset = TrainDataset(CACHE_PATH)
    val_dataset = ValDataset(CACHE_PATH)

    # # Test dataset loading with samples
    # sample_embedding, sample_label = train_dataset[0]
    # print("shapes:", sample_embedding.dim(), sample_label.dim())
    # print(sample_embedding, sample_label)

    # Create DataLoaders with optimized settings
    train_loader = DataLoader(
        train_dataset,  # train dataset
        batch_size=BATCH_SIZE,  # batch size
        shuffle=True,  # shuffle for training
        persistent_workers=True,  # keep workers alive between epochs
        num_workers=NUM_WORKERS_EMB,  # number of worker processes for data loading
        pin_memory=True,  # pin memory for faster transfer to GPU
        prefetch_factor=4,  # prefetch factor for each worker
        # drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,  # validation dataset
        batch_size=BATCH_SIZE,  # batch size
        shuffle=False,  # no shuffling for validation, recommended by PyTorch
        persistent_workers=True,  # keep workers alive between epochs
        num_workers=NUM_WORKERS_EMB,  # number of worker processes for data loading
        pin_memory=True,  # pin memory for faster transfer to GPU
        prefetch_factor=4,  # prefetch factor for each worker
        # drop_last=True,
    )

    # Print dataset sizes
    if RANK == 0:
        print(
            f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}"
        )

    # -------------------------
    # 10.3. Weight calculation for class imbalance handling
    # -------------------------

    # Calculate class weights for handling class imbalance with class_distribution function
    weights = class_distribution()

    # -------------------------
    # 10.4. HP search with Optuna
    # -------------------------

    # Setup and run Optuna study
    study, input_dim = run_optuna_study(
        train_dataset, train_loader, val_loader, weights
    )

    # -------------------------
    # 10.5. Load best model after HP search for final training
    # -------------------------

    # Find and broadcast the best trial to each rank
    best_trial = find_and_broadcast_best_trial(study)

    # Load the best model architecture with best hyperparameters
    lit_model = load_best_model(best_trial, input_dim, weights)

    if RANK == 0:
        print("Best model loaded and set to eval mode")

    # -------------------------
    # 10.6. Final training with best hyperparameters
    # -------------------------

    if Final_training is True:
        trainer, final_model_path = final_training(lit_model, train_loader, val_loader)

    # Cleanup distributed processes to end script cleanly
    if dist.is_initialized():
        # Ensure all processes are synchronized before exiting
        dist.barrier()
        if RANK == 0:
            print("Done! Exiting...")
        # destroy the process group
        dist.destroy_process_group()


# --------------------------------------------------------------------------------------------
# 11. Main usage function, calling loader and predicter
# --------------------------------------------------------------------------------------------


def main_usage(csv_path, output_dir=None):
    """
    Wrapper function for usage mode: loads model and dataloader, performs prediction, and saves results.
    
    This function orchestrates the entire inference pipeline:
    1. Loads the trained model and creates the dataloader
    2. Performs predictions on all samples
    3. Saves predictions to CSV file
    
    Args:
        csv_path: Path to CSV file containing sequences to be used for inference
        output_dir: Directory to save prediction results. If None, uses default tempTest directory
        
    Returns:
        predictions_df: DataFrame containing predictions and raw scores
        output_path: Path where predictions were saved
    """
    # Load model and dataloader
    model, classifier_loader = loader(csv_path)

    # Perform predictions
    all_predictions, all_predictions_raw = predictor(model, classifier_loader)

    # Create DataFrame with predictions
    predictions_df = pd.DataFrame(
        {"prediction": all_predictions, "raw_scores": all_predictions_raw}
    )

    # Determine output directory
    if output_dir is None:
        output_dir = "/global/research/students/sapelt/Masters/MasterThesis/tempTest"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    output_path = os.path.join(output_dir, "predictions.csv")
    
    # Save predictions to CSV
    predictions_df.to_csv(output_path, index=False)

    # Print summary (only on rank 0 to avoid duplicate output)
    if RANK == 0:
        print(f"\n{'='*60}")
        print("Prediction Summary:")
        print(f"{'='*60}")
        print(f"Total predictions: {len(all_predictions)}")
        print(f"{'='*60}\n")

    # Cleanup distributed processes to end script cleanly
    if dist.is_initialized():
        # Ensure all processes are synchronized before exiting
        dist.barrier()
        if RANK == 0:
            print("Done! Exiting...")
        # destroy the process group
        dist.destroy_process_group()

    return


# --------------------------------------------------------------------------------------------
# 12. Main entry point for usage and HP search
# --------------------------------------------------------------------------------------------


def main():
    """
    Main entry point for HP search and usage mode. Gets called on top-level execution.
    
    Parses command line arguments to determine execution mode:
    1. If HP_mode is True: Starts hyperparameter search and final training, then exits
    2. If HP_mode is False: Runs inference mode using trained model on provided CSV
    
    Command line usage:
        HP search mode: python ESM_Embeddings_HP_search.py --HP_mode
        Usage mode: python ESM_Embeddings_HP_search.py --csv_path path/to/data.csv
    """
    args = parse_args()

    # HP search mode: run hyperparameter optimization and final training
    if args.HP_mode is True:
        main_HP(Final_training=True)
        exit(0)

    # else: Usage mode: perform predictions on provided CSV
    csv_path = args.csv_path
    main_usage(csv_path)
    exit(0)

#############################################################################################################

if __name__ == "__main__":
    if NUM_CLASSES == "FULL":
        NUM_CLASSES = 24381  # for full pfam classification, based on pfam 37.3
    main()
