import os
import time
import pickle
import glob
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import esm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics import Precision, Recall

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True 
torch.backends.cuda.matmul.allow_tf32 = (
    True 
)
torch.backends.cudnn.allow_tf32 = True

# -------------------------
# 1. Global settings
# -------------------------
CSV_PATH = "./Dataframes/DataTrainSwissPro_esm_10d_Thiolase_shuffled.csv"
CATEGORY_COL = "categories"
SEQUENCE_COL = "Sequences"
CACHE_PATH = "./pickle/DataTrainSwissPro_esm_10d_Thiolase_shuffled_10000.pkl"
PROJECT_NAME = "Optuna_10d_test_Thio_35M"


NUM_CLASSES = 11
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15

EMB_BATCH = 64
NUM_WORKERS_EMB = max(16, os.cpu_count())
print(f"Using {NUM_WORKERS_EMB} workers for embedding generation")
BATCH_SIZE = 128
LR = 1e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 150
STUDY_N_TRIALS = 300

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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
                activation,  # ReLU activation
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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer_class = optimizer_class
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
            )
        else:
            return self.optimizer_class( 
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
    def __init__(self, skip_df=None):
        def map_label(cat):
            # Adaptive mapping based on NUM_CLASSES
            # Classes 0-9 get mapped to 1-10
            # Classes 10 and 11 get mapped to 0
            if cat <= NUM_CLASSES - 2:
                return cat + 1
            else:
                return 0

        if skip_df is None:
            df = pd.read_csv(CSV_PATH,nrows=10000)
            df["label"] = df[CATEGORY_COL].apply(map_label)
            df.drop(columns=[CATEGORY_COL], inplace=True)
            self.df = df
            print("Data loaded")
        else:
            self.df = skip_df

        self.model, self.alphabet, self.batch_converter = self.esm_loader()

        # Use sequences in original order
        sequences = self.df[SEQUENCE_COL].tolist()

        print(f"Starting embedding generation for {len(sequences)} sequences...")
        start_time = time.time()

        self.embeddings = self._embed(sequences).cpu()

        end_time = time.time()
        embedding_time = end_time - start_time

        # Print timing information
        print("Embedding generation completed!")
        print(
            f"Total time: {embedding_time:.2f} seconds ({embedding_time / 60:.2f} minutes)"
        )
        print(
            f"Time per sequence: {embedding_time / len(sequences):.4f} seconds"
        )
        print(f"Sequences per second: {len(sequences) / embedding_time:.2f}")

        # No need to restore order since we didn't sort
        self.labels = torch.tensor(self.df["label"].values, dtype=torch.long)

        # Add stratified train/val split using scikit-learn
        print("Creating stratified train/val split...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.embeddings,
            self.labels,
            test_size=TEST_FRAC,
            stratify=self.labels,
            random_state=42,
        )

        # Calculate validation size from remaining data
        val_size_adjusted = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=42,
        )

        self.train_embeddings = X_train
        self.train_labels = y_train
        self.val_embeddings = X_val
        self.val_labels = y_val
        self.test_embeddings = X_test
        self.test_labels = y_test

        print(f"Train set: {len(self.train_embeddings)} samples")
        print(f"Val set: {len(self.val_embeddings)} samples")
        print(f"Test set: {len(self.test_embeddings)} samples")
        print("Embeddings computed and split completed")

    def esm_loader(self):
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # Changed back to bigger model

        # Set the model to the primary device first
        model = model.to(DEVICE).eval()

        # Enable mixed precision for faster computation
        if torch.cuda.is_available():
            model = model.half()

        print(f"ESM model loaded on device: {DEVICE}")
        if torch.cuda.device_count() > 1:
            print(
                f"Model will be wrapped with DataParallel for {torch.cuda.device_count()} GPUs"
            )

        return model, alphabet, alphabet.get_batch_converter()
    

    def _embed(self, seqs):
        """
        Optimized batching with multi-GPU support for embedding generation.
        """
        embed_start_time = time.time()

        # 1) Enable multi-GPU processing with optimized batch sizing
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for embedding generation")
            if not hasattr(self.model, "module"):
                self.model = torch.nn.DataParallel(self.model)
            # Increase batch size more aggressively for multi-GPU
            effective_batch_size = EMB_BATCH * torch.cuda.device_count()
        else:
            effective_batch_size = EMB_BATCH * 2  # Increase single GPU batch size
            print("Using single GPU for embedding generation")

        # 2) Use sequences in original order (no sorting)
        sequences = seqs

        # 3) JIT compilation for single GPU
        if not hasattr(self, "_jit_compiled"):
            compile_start = time.time()
            try:
                if hasattr(torch, "compile") and torch.cuda.device_count() == 1:
                    self.model = torch.compile(self.model, mode="max-autotune")
                    print(f"Model compilation took: {time.time() - compile_start:.2f} seconds")
            except Exception as e:
                print(f"Torch compile failed: {e}")
            self._jit_compiled = True

        batch_size = effective_batch_size
        all_outputs = []
        total_batches = (len(sequences) + batch_size - 1) // batch_size

        print(f"Processing {len(sequences)} sequences with batch size {batch_size}")
        print(f"Total batches to process: {total_batches}")

        # 4) Pre-allocate CUDA streams for better GPU utilization
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()
        
        batch_times = []

        # 5) Optimized batching in original order
        for batch_idx, start in enumerate(range(0, len(sequences), batch_size)):
            batch_start_time = time.time()

            if batch_idx % 5 == 0:  # More frequent progress updates
                elapsed = time.time() - embed_start_time
                if batch_idx > 0:
                    avg_batch_time = sum(batch_times) / len(batch_times)
                    remaining_batches = total_batches - batch_idx
                    eta = remaining_batches * avg_batch_time
                    print(f"\rBatch {batch_idx + 1}/{total_batches} | "
                        f"Elapsed: {elapsed:.1f}s | ETA: {eta / 60:.1f}min", end="", flush=True)

            chunk = sequences[start : start + batch_size]
            labels = [f"seq{i}" for i in range(len(chunk))]  # Simplified labels

            try:
                with torch.cuda.stream(stream) if torch.cuda.is_available() else torch.no_grad():
                    # Batch conversion
                    batch_labels, batch_strs, batch_tokens = self.batch_converter(
                        list(zip(labels, chunk))
                    )
                    batch_tokens = batch_tokens.to(DEVICE, non_blocking=True)

                    with (
                        torch.no_grad(),
                        torch.amp.autocast("cuda", enabled=True, dtype=torch.float16),
                    ):
                        # Model inference
                        if hasattr(self.model, "module"):
                            out = self.model(
                                tokens=batch_tokens,
                                repr_layers=[33],  # Changed back to 33
                                return_contacts=False,
                            )["representations"][33]  # Changed back to 33
                        else:
                            out = self.model(
                                batch_tokens,
                                repr_layers=[33],  # Changed back to 33
                                return_contacts=False,
                            )["representations"][33]  # Changed back to 33

                    # Optimized pooling with vectorized operations
                    padding_mask = (batch_tokens == self.alphabet.padding_idx)
                    # Use efficient masking
                    out_masked = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                    seq_lengths = (~padding_mask).sum(dim=1, keepdim=True).float()
                    pooled = out_masked.sum(dim=1) / seq_lengths.clamp(min=1)

                    # Keep in float16 to save memory
                    all_outputs.append(pooled.half())

                    # Aggressive memory cleanup
                    del out, out_masked, padding_mask, seq_lengths, batch_tokens

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\rOOM in batch {batch_idx}, falling back to single sequence processing", end="", flush=True)
                    
                    # Process each sequence individually
                    for seq_idx, seq in enumerate(chunk):
                        try:
                            single_batch = self.batch_converter([(f"seq{seq_idx}", seq)])
                            _, _, single_tokens = single_batch
                            single_tokens = single_tokens.to(DEVICE)

                            with (
                                torch.no_grad(),
                                torch.amp.autocast("cuda", enabled=True, dtype=torch.float16),
                            ):
                                if hasattr(self.model, "module"):
                                    single_out = self.model(
                                        tokens=single_tokens,
                                        repr_layers=[33],  # Changed back to 33
                                        return_contacts=False,
                                    )["representations"][33]  # Changed back to 33
                                else:
                                    single_out = self.model(
                                        single_tokens,
                                        repr_layers=[33],  # Changed back to 33
                                        return_contacts=False,
                                    )["representations"][33]  # Changed back to 33

                            single_mask = (single_tokens == self.alphabet.padding_idx)
                            single_out_masked = single_out.masked_fill(single_mask.unsqueeze(-1), 0.0)
                            single_lengths = (~single_mask).sum(dim=1, keepdim=True).float()
                            single_pooled = single_out_masked.sum(dim=1) / single_lengths.clamp(min=1)

                            all_outputs.append(single_pooled.half())
                            
                            del single_out, single_out_masked, single_mask, single_lengths, single_tokens
                            
                        except Exception as single_e:
                            print(f"Failed single sequence {seq_idx}: {single_e}")
                            # Only use dummy as absolute last resort for individual sequences
                            dummy = torch.zeros((1, 1280), device=DEVICE, dtype=torch.float16)  # Changed back to 1280
                            all_outputs.append(dummy)
                            

                else:
                    print(f"Non-OOM error in batch {batch_idx}: {e}")
                    # For non-OOM errors, also try individual processing
                    print("Attempting individual sequence processing...")
                    for seq_idx, seq in enumerate(chunk):
                        try:
                            single_batch = self.batch_converter([(f"seq{seq_idx}", seq)])
                            _, _, single_tokens = single_batch
                            single_tokens = single_tokens.to(DEVICE)

                            with (
                                torch.no_grad(),
                                torch.amp.autocast("cuda", enabled=True, dtype=torch.float16),
                            ):
                                if hasattr(self.model, "module"):
                                    single_out = self.model(
                                        tokens=single_tokens,
                                        repr_layers=[33],  # Changed back to 33
                                        return_contacts=False,
                                    )["representations"][33]  # Changed back to 33
                                else:
                                    single_out = self.model(
                                        single_tokens,
                                        repr_layers=[33],  # Changed back to 33
                                        return_contacts=False,
                                    )["representations"][33]  # Changed back to 33

                            single_mask = (single_tokens == self.alphabet.padding_idx)
                            single_out_masked = single_out.masked_fill(single_mask.unsqueeze(-1), 0.0)
                            single_lengths = (~single_mask).sum(dim=1, keepdim=True).float()
                            single_pooled = single_out_masked.sum(dim=1) / single_lengths.clamp(min=1)

                            all_outputs.append(single_pooled.half())
                            
                            del single_out, single_out_masked, single_mask, single_lengths, single_tokens
                            
                        except Exception as single_e:
                            print(f"Failed single sequence {seq_idx}: {single_e}")
                            dummy = torch.zeros((1, 1280), device=DEVICE, dtype=torch.float16)  # Changed back to 1280
                            all_outputs.append(dummy)

            # Track batch timing
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)



        # 6) Simple concatenation (no reordering needed)
        if all_outputs:
            embeddings = torch.cat(all_outputs, dim=0).float().cpu()
        else:
            embeddings = torch.zeros((len(seqs), 1280))  # Changed back to 1280


        total_embed_time = time.time() - embed_start_time
        print(f"\n=== Embedding Generation Summary ===")
        print(f"Total time: {total_embed_time:.2f}s ({total_embed_time / 60:.2f}min)")
        print(f"Sequences per second: {len(seqs) / total_embed_time:.2f}")
        print("=====================================\n")

        return embeddings
    

# -------------------------
# 5. Main
# -------------------------
def main(Final_training=False):
    os.makedirs("pickle", exist_ok=True)
    os.makedirs(f"logs/{PROJECT_NAME}", exist_ok=True)  # Add this line
    os.makedirs("models", exist_ok=True)  # Add this line for model saving

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
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_EMB, pin_memory=True
    )
    print("Train loader created")

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_EMB, pin_memory=True
    )
    print("Val loader created")

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
            f"\n=== Trial {trial.number} ===\n"
            f"Hidden layers: {len(hidden_dims)}, Neurons: {n_neurons}\n"
            f"Dropout: {drop:.3f}, LR: {lr:.6f}, WD: {wd:.6f}\n"
            f"Optimizer: {optimizer}, Activation: {activation.__class__.__name__}\n"
            f"========================\n"
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
            monitor="val_loss", patience=10, mode="min", verbose=True
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
                save_dir=f"./logs/{PROJECT_NAME}", name=f"optuna_trial_{trial.number}"
            ),
        )
        # print(model)

        trainer.fit(model, train_loader, val_loader)
        return checkpoint_callback.best_model_score.item()

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///logs/{PROJECT_NAME}/optuna_study.db",
        load_if_exists=True,
        study_name="esm_10d_hp_search",
    )
    study.optimize(objective, n_trials=STUDY_N_TRIALS)

    print("Best trial number:", study.best_trial.number)
    print("Best trial:", study.best_trial)
    # print("Best trial params:", study.best_trial.params)

    best_trial = study.best_trial

    def load_best_model(trial):
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

        torch.save(model, f"./models/{PROJECT_NAME}_bestmodel10d.pt")
        model = model.to(DEVICE).eval()

        return model

    lit_model = load_best_model(best_trial)

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
            max_epochs=200,
            accelerator="gpu",
            devices=-1,
            # strategy="ddp",
            enable_progress_bar=True,
            callbacks=[early_stop, checkpoint_callback],
            logger=TensorBoardLogger(
                save_dir=f"./logs/{PROJECT_NAME}", name=PROJECT_NAME
            ),
        )

        print("Trainer created")

        trainer.fit(lit_model, train_loader, val_loader)


#############################################################################################################

if __name__ == "__main__":
    main(Final_training=True)
