import glob
import math
import os
import pickle
import time

import esm
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from fairscale.nn.wrap import enable_wrap, wrap
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics import Precision, Recall

os.environ["NCCL_P2P_DISABLE"] = "1"

torch.set_float32_matmul_precision("high")  # More stable than "high"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
if not dist.is_initialized():
    dist.init_process_group("nccl")
    if dist.get_rank() == 0:
        print("Initializing process group for DDP")
RANK = dist.get_rank()
print(f"Start running basic DDP example on rank {RANK}.")
# create model and move it to GPU with id rank
DEVICE_ID = RANK % torch.cuda.device_count()


pd.set_option("display.max_rows", None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

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

BATCH_SIZE = 512
LR = 1e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 150
STUDY_N_TRIALS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if RANK == 0:
    # print(f"Using {NUM_WORKERS_EMB} workers for embedding generation")
    print(f"Using device: {DEVICE} | count: {torch.cuda.device_count()}")

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
        if domain_task is False:
            if class_weights is not None:
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")           # for domain boundary detection

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
        loss = self.loss_fn(logits, y)
        # print(loss)
        assert not torch.isnan(loss), "Validation loss is NaN"
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
                nesterov=True,  # Nesterov momentum
            )
        else:
            return self.optimizer_class(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
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
        return self.seqs[idx]


class ESMDataset:
    def __init__(
        self,
        skip_df=None,
        FSDP_used=False,
        domain_boundary_detection=False,
        num_classes=NUM_CLASSES,
        esm_model=ESM_MODEL,
        csv_path=CSV_PATH,
        category_col=CATEGORY_COL,
        sequence_col=SEQUENCE_COL,
    ):
        self.esm_model = esm_model

        def map_label(cat):
            # Adaptive mapping based on NUM_CLASSES
            # Classes 0-9 get mapped to 1-10
            # Classes 10 and 11 get mapped to 0
            if cat <= num_classes - 2:
                return cat + 1
            else:
                return 0

        def minicutter(row):
            # reads in start, end values to cut the sequence
            sequence = row[sequence_col]
            start = int(row["start"])
            end = int(row["end"])
            return sequence[start:end]

        if skip_df is None:
            df = pd.read_csv(csv_path)
            if (
                "start" in df.columns
                and "end" in df.columns
                and domain_boundary_detection is False
            ):
                if RANK == 0:
                    print("Cutting sequences based on start and end positions")
                df[sequence_col] = df.apply(minicutter, axis=1)

                # Filter out sequences that are too short to ensure no NaN values
                initial_count = len(df)
                df = df[df[sequence_col].str.len() >= 10]    # 10 for biological reasonings
                final_count = len(df)
                if initial_count != final_count:
                    if RANK == 0:
                        print(
                            f"Removed {initial_count - final_count} sequences with length <= 10"
                        )
                        print(f"Remaining sequences: {final_count}")

            if category_col != "Pfam_id":
                df["label"] = df[category_col].apply(map_label)
            elif domain_boundary_detection is False:
                # Create mapping once outside the apply function
                unique_pfam_ids = df[category_col].unique()

                # Count occurrences of each Pfam ID
                pfam_counts = df[category_col].value_counts()

                # Filter to only IDs with more than 100 occurrences
                frequent_pfam_ids = pfam_counts[pfam_counts > 100].index.tolist()
                if RANK == 0:
                    print(
                        f"Found {len(frequent_pfam_ids)} Pfam IDs with more than 100 occurrences"
                    )

                # Define the specific 10 Pfam IDs you want to use first
                priority_pfam_ids = [
                    "PF00177",
                    "PF00210",
                    "PF00211",
                    "PF00215",
                    "PF00217",
                    "PF00406",
                    "PF00303",
                    "PF00246",
                    "PF00457",
                    "PF00502",
                ]

                # Filter priority IDs that actually exist in the dataset AND have >100 occurrences
                available_priority_ids = [
                    pid for pid in priority_pfam_ids if pid in frequent_pfam_ids
                ]

                # Get remaining frequent IDs (excluding the priority ones)
                remaining_ids = [
                    pid for pid in frequent_pfam_ids if pid not in priority_pfam_ids
                ]

                # Take up to additional IDs to reach NUM_CLASSES-1 total
                max_additional_ids = num_classes - 1 - len(available_priority_ids)
                selected_remaining_ids = remaining_ids[:max_additional_ids]

                # Combine priority IDs with selected remaining IDs
                selected_ids = available_priority_ids + selected_remaining_ids

                pfam_to_label = {}
                # Map the selected IDs to labels 1 through len(selected_ids)
                for i, pfam_id in enumerate(selected_ids):
                    pfam_to_label[pfam_id] = i + 1

                # Map all other IDs to label 0
                for pfam_id in unique_pfam_ids:
                    if pfam_id not in pfam_to_label:
                        pfam_to_label[pfam_id] = 0

                # Use map instead of apply for better performance
                df["label"] = df[category_col].map(pfam_to_label).fillna(0)

                # # Count samples per class
                # class_counts = df["label"].value_counts()

                # # Subsample class 0 to be similar size to other classes
                # class_0_samples = df[df["label"] == 0]
                # other_class_avg = class_counts[class_counts.index != 0].mean()
                # target_class_0_size = int(
                #     other_class_avg * 2
                # )  # Make class 0 times 5 the average

                # if len(class_0_samples) > target_class_0_size:
                #     print(
                #         f"Subsampling class 0 from {len(class_0_samples)} to {target_class_0_size}"
                #     )
                #     class_0_subsampled = class_0_samples.sample(
                #         n=target_class_0_size, random_state=42
                #     )

                #     # Combine subsampled class 0 with other classes
                #     other_classes = df[df["label"] != 0]
                #     df = pd.concat(
                #         [other_classes, class_0_subsampled], ignore_index=True
                #     )


                if RANK == 0:
                    print("Final class distribution:")
                    final_counts = df["label"].value_counts()
                    for i in range(0, len(final_counts)):
                        if i in final_counts:
                            print(f"Class {i}: {final_counts[i]} samples")

            if category_col != "Pfam_id":
                df.drop(columns=[category_col], inplace=True)
            else:
                if domain_boundary_detection is False:
                    df.drop(columns=["start", "end", "id", "Pfam_id"], inplace=True)
                else:
                    df.drop(columns=["id", "Pfam_id"], inplace=True)
            self.df = df
            if RANK == 0:
                print("Data loaded")
        else:
            self.df = skip_df

        # print(self.df.head(150))

        self.FSDP_used = FSDP_used

        self.model, self.batch_converter = self.esm_loader()

        # Use sequences in original order
        sequences = self.df[sequence_col].tolist()

        if domain_boundary_detection is False:
            self.labels = torch.tensor(self.df["label"].values, dtype=torch.long)

        else:
            # Switching labels for domain boundary detection
            print("Switching labels for domain boundary detection")
            labels_list = []
            for index, row in self.df.iterrows():
                seq_len = len(row[sequence_col])
                label = [0] * seq_len
                start = int(row["start"])
                end = int(row["end"])
                if start < seq_len:
                    label[start] = 1
                if end < seq_len:
                    label[end] = 1
                labels_list.append(torch.tensor(label, dtype=torch.long))

            self.labels = labels_list  # Store as list of tensors
            print(self.labels[:5])  # Print first 5 labels for verification

        start_time = time.time()

        self.embeddings = self._embed(sequences).cpu()

        end_time = time.time()
        embedding_time = end_time - start_time

        # Print timing information
        print("Embedding generation completed!")
        print(
            f"Total time: {embedding_time:.2f} seconds ({embedding_time / 60:.2f} minutes)"
        )
        print(f"Time per sequence: {embedding_time / len(sequences):.4f} seconds")
        print(f"Sequences per second: {len(sequences) / embedding_time:.2f}")

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
        # init the distributed world with world_size 1

        # download model data from the hub
        model_name = self.esm_model

        self.model_layers = int(model_name.split("_")[1][1:])
        if RANK == 0:
            print(f"Loading ESM model: {model_name} with {self.model_layers} layers")

        model_data, regression_data = (
            esm.pretrained._download_model_and_regression_data(model_name)
        )

        if self.FSDP_used is True:
            print("Using FSDP for model wrapping")
            # initialize the model with FSDP wrapper
            fsdp_params = dict(
                mixed_precision=True,
                flatten_parameters=True,
                state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
                cpu_offload=True,  # enable cpu offloading
            )
            with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
                model, vocab = esm.pretrained.load_model_and_alphabet_core(
                    model_name, model_data, regression_data
                )
                batch_converter = vocab.get_batch_converter()
                model.eval()

                # Wrap each layer in FSDP separately
                for name, child in model.named_children():
                    if name == "layers":
                        for layer_name, layer in child.named_children():
                            wrapped_layer = wrap(layer)
                            setattr(child, layer_name, wrapped_layer)
                model = wrap(model)

        else:
            model, alphabet = getattr(esm.pretrained, model_name)()
            batch_converter = alphabet.get_batch_converter()
            model.eval()  # disables dropout for deterministic results

            model = model.cuda()

            model = model.half()

            model = DDP(model, device_ids=[DEVICE_ID])

        return model, batch_converter

    def _embed(self, seqs):
        """
        Generate embeddings for a list of sequences using ESM model.
        """
        
        # Determine the correct embedding dimension from the model
        model_dims = {
            "esm2_t6_8M_UR50D": 320,
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
            "esm2_t48_15B_UR50D": 5120,
        }
        expected_dim = model_dims.get(self.esm_model, None)

        all_embeddings = []
        seqs = list(seqs)
        
        start_time = time.time()



        seq_dataset = SeqDataset(seqs)
        sampler = DistributedSampler(
            seq_dataset, 
            num_replicas=dist.get_world_size(), 
            rank=RANK, 
            shuffle=False, 
            drop_last=False
        )

        dataloader = DataLoader(
            seq_dataset, 
            batch_size=EMB_BATCH, 
            num_workers=0, 
            sampler=sampler,
            pin_memory=True
        )

        if RANK == 0:
            print(f"Expected embedding dimension: {expected_dim}")
            print(f"Total sequences to process: {len(seqs)} with batch size {EMB_BATCH}")
        print(f"Rank:{RANK} will process sequences {sampler.num_samples} out of {len(seqs)} total")

        for batch_num, batch_seqs in enumerate(dataloader):    
            try:
                # batch_seqs is already a list of sequences from the DataLoader
                batch = [(f"seq{i}", seq) for i, seq in enumerate(batch_seqs)]
                _, _, batch_tokens = self.batch_converter(batch)
                batch_tokens = batch_tokens.cuda(non_blocking=True)

                with torch.no_grad():
                    results = self.model(
                        batch_tokens,
                        repr_layers=[self.model_layers],
                        return_contacts=False,
                    )
                    embeddings = results["representations"][self.model_layers].float()

                    # Pool over sequence dimension (dim=1), ignoring padding (token > 1)
                    mask = batch_tokens != 1
                    mask = mask[:, 1:-1]
                    embeddings = embeddings[:, 1:-1, :]

                    lengths = mask.sum(dim=1, keepdim=True)
                    pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / lengths
                    pooled = pooled.cpu()

                    all_embeddings.append(pooled)

            # Progress reporting (only from rank 0)
                if (batch_num % 1 == 0 or batch_num == len(dataloader) - 1):
                    elapsed_time = time.time() - start_time
                    if batch_num > 0:
                        avg_time_per_batch = elapsed_time / (batch_num + 1)
                        remaining_batches = len(dataloader) - batch_num - 1
                        eta_seconds = remaining_batches * avg_time_per_batch
                        
                        eta_hours = int(eta_seconds // 3600)
                        eta_minutes = int((eta_seconds % 3600) // 60)
                        eta_seconds_remainder = int(eta_seconds % 60)
                        
                        print(f"Batch {batch_num + 1}/{len(dataloader)} | "
                            f"ETA: {eta_hours}h {eta_minutes}m {eta_seconds_remainder}s", 
                            end='\r', flush=True)

                del batch_tokens, results, embeddings, pooled, mask

            except Exception as e:
                print(f"\nRank {RANK}: Error processing batch {batch_num + 1}: {e}")
                # Handle individual sequences...
                for seq in batch_seqs:
                    try:
                        single_seq = [(f"seq0", seq)]
                        single_labels, single_strs, single_tokens = self.batch_converter(single_seq)
                        single_tokens = single_tokens.cuda()

                        with torch.no_grad():
                            results = self.model(
                                single_tokens,
                                repr_layers=[self.model_layers],
                                return_contacts=False,
                            )
                            embedding = results["representations"][self.model_layers].float()

                            mask = single_tokens != 1
                            mask = mask[:, 1:-1]
                            embedding = embedding[:, 1:-1, :]

                            length = mask.sum(dim=1, keepdim=True)
                            pooled = (embedding * mask.unsqueeze(-1)).sum(dim=1) / length
                            pooled = pooled.cpu()

                            all_embeddings.append(pooled)

                    except Exception as single_e:
                        print(f"Rank {RANK}: Error processing individual sequence: {single_e}")
                        zero_embedding = torch.zeros(1, expected_dim)
                        all_embeddings.append(zero_embedding)

        # Concatenate local embeddings
        local_embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, expected_dim)
        print(f"Rank {RANK}: Generated {local_embeddings.size(0)} local embeddings")

        # Synchronize all processes
        dist.barrier()

        # Gather embeddings from all processes
        if dist.get_world_size() > 1:
            # Get the size of embeddings from each rank
            local_size = torch.tensor([local_embeddings.size(0)], dtype=torch.int64, device='cuda')
            all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
            dist.all_gather(all_sizes, local_size)
            
            # Move local embeddings to GPU for gathering
            local_embeddings_gpu = local_embeddings.cuda()
            
            # Prepare tensors for gathering (pad to max size)
            max_size = max(size.item() for size in all_sizes)
            if local_embeddings_gpu.size(0) < max_size:
                padding = torch.zeros(max_size - local_embeddings_gpu.size(0), expected_dim, 
                                    device=local_embeddings_gpu.device, dtype=local_embeddings_gpu.dtype)
                local_embeddings_gpu = torch.cat([local_embeddings_gpu, padding], dim=0)
            
            # Gather all embeddings
            gathered_embeddings = [torch.zeros_like(local_embeddings_gpu) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_embeddings, local_embeddings_gpu)
            
            # Concatenate only the valid portions and move to CPU
            if RANK == 0:
                final_embeddings_list = []
                for i, (emb_tensor, size_tensor) in enumerate(zip(gathered_embeddings, all_sizes)):
                    valid_size = size_tensor.item()
                    if valid_size > 0:
                        final_embeddings_list.append(emb_tensor[:valid_size].cpu())
                
                final_embeddings = torch.cat(final_embeddings_list, dim=0) if final_embeddings_list else torch.empty(0, expected_dim)
                print(f"\nRank 0: Gathered {final_embeddings.size(0)} total embeddings from all ranks")
            else:
                final_embeddings = torch.empty(0, expected_dim)
            
            # Clean up GPU memory
            del local_embeddings_gpu, gathered_embeddings
            torch.cuda.empty_cache()
        else:
            final_embeddings = local_embeddings

        return final_embeddings
    
# -------------------------
# 5. Optuna objective func, load best model func
# -------------------------

def objective(
    trial, train_embeddings, train_loader, val_loader, weights, domain_task=False
):

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

    activation = trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu"])
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
        domain_task=False,  # Set to True if using domain boundary detection
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=150,
        accelerator="gpu",
        devices=1,
        # strategy="ddp",
        enable_progress_bar=True,
        callbacks=[early_stop, checkpoint_callback],
        logger=TensorBoardLogger(
            save_dir=f"./logs/{PROJECT_NAME}", name=f"optuna_trial_{trial.number}"
        ),
        gradient_clip_val=1.0,  # Add gradient clipping
        detect_anomaly=True,  # Enable anomaly detection for debugging
    )

    # print(model)

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
        esm_data = ESMDataset(FSDP_used=False)

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
            max_epochs=50,
            accelerator="gpu",
            devices=1,
            # strategy="ddp",
            enable_progress_bar=True,
            callbacks=[early_stop, checkpoint_callback],
            logger=TensorBoardLogger(
                save_dir=f"./logs/{PROJECT_NAME}", name=PROJECT_NAME
            ),
        )

        print("Trainer created")

        trainer.fit(lit_model, train_loader, val_loader)

        # save the final model
        final_model_path = f"./models/{PROJECT_NAME}.pt"
        torch.save(lit_model, final_model_path)

#############################################################################################################

if __name__ == "__main__":
    main(Final_training=True)
