import time
import h5py
import numpy as np
import esm
import gc
import pandas as pd
import torch
import os
import psutil
# from pympler import asizeof
# from pympler import summary,muppy
# from pympler.classtracker import ClassTracker
import torch.distributed as dist
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

world_size = int(os.environ.get("WORLD_SIZE", 1))
use_ddp = world_size > 1

# print(torch.__version__)


if use_ddp:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        if dist.get_rank() == 0:
            print("Initializing process group for DDP")
    RANK = dist.get_rank()
    if RANK == 0:
        print(f"Start running basic DDP example with worldsize {dist.get_world_size()}.")
    # create model and move it to GPU with id rank
    DEVICE_ID = RANK % torch.cuda.device_count()
else:
    # Single-GPU fallback for quick testing
    DEVICE_ID = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RANK = 0
    print("Running in single-GPU mode without DDP.")


TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2

EMB_BATCH = 1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if RANK == 0:
    print(f"Using device: {DEVICE} | count: {torch.cuda.device_count()}")

# -------------------------
# 4. Dataset & embedding
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


class SeqDatasetForEval(Dataset):
    def __init__(self, seqs, labels, starts, ends):
        self.seqs = seqs
        self.labels = labels
        self.starts = starts
        self.ends = ends

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx], self.starts[idx], self.ends[idx]


class ESMDataset:
    def __init__(
        self,
        num_classes,
        esm_model,
        csv_path,
        category_col,
        sequence_col,
        skip_df=None,
        FSDP_used=False,
        domain_boundary_detection=False,
        training=False,

    ):

        self.esm_model = esm_model
        self.training = training
        self.skip_df = skip_df
        self.domain_boundary_detection = domain_boundary_detection
        self.sequence_col = sequence_col
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
        
        def subsampler(df,target):

            if target is None:
                return df

            # Use map instead of apply for better performance
            df["label"] = df[category_col].map(pfam_to_label).fillna(0)

            # Count samples per class
            class_counts = df["label"].value_counts()
            
            # Subsample ALL classes to target samples each
            target_samples_per_class = target
            subsampled_dfs = []
            
            if RANK == 0:
                print("Subsampling all classes to 100 samples each:")
            
            for class_label in sorted(class_counts.index):
                class_samples = df[df["label"] == class_label]
                current_count = len(class_samples)
                
                if current_count > target_samples_per_class:
                    # Subsample to target size
                    class_subsampled = class_samples.sample(
                        n=target_samples_per_class, random_state=42
                    )
                    if RANK == 0:
                        print(f"  Class {class_label}: {current_count} -> {target_samples_per_class}")
                else:
                    # Keep all samples if less than target
                    class_subsampled = class_samples
                    if RANK == 0:
                        print(f"  Class {class_label}: {current_count} (kept all)")
                
                subsampled_dfs.append(class_subsampled)
            
            # Combine all subsampled classes
            df = pd.concat(subsampled_dfs, ignore_index=True)

            if RANK == 0:
                all_pfam = []
                # print("Final class distribution after subsampling:")
                final_counts = df["label"].value_counts()
                for i in sorted(final_counts.index):
                    if i in final_counts:
                        pfam_id = (
                            selected_ids[i - 1]
                            if i > 0 and i <= len(selected_ids)
                            else "other"
                        )
                        print(
                            f"Class {i} with ID {pfam_id}: {final_counts[i]} from samples | avg length {df[df['label'] == i][sequence_col].str.len().mean():.2f}"
                        )
                        
                        all_pfam.append(pfam_id)

                if self.training is True:
                    os.makedirs("./temp", exist_ok=True)
                    all_pfam=all_pfam[1:]  # Exclude "other" category
                    open(f"./temp/selected_pfam_ids_{num_classes}.txt", "w").write(
                        "\n".join(all_pfam)
                    )
            print("\n")
            return df


        if skip_df is None:
            df = pd.read_csv(csv_path,nrows=100)          ##################### TESTING PURPOSES
            if (
                "start" in df.columns
                and "end" in df.columns
                and domain_boundary_detection is False
            ):
                if training is True:
                    if RANK == 0:
                        print("Cutting sequences based on start and end positions")

                    df[sequence_col] = df.apply(minicutter, axis=1)

                # Filter out sequences that are too short to ensure no NaN values
                initial_count = len(df)

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

                if (
                    self.training is False
                ):  
               # to ensure same ids used during evaluation on prediction quality
                    if RANK == 0:
                        print("Using fixed priority Pfam IDs for evaluation")
                    with open(f"./temp/selected_pfam_ids_{num_classes}.txt", "r") as f:
                        priority_pfam_ids = [line.strip() for line in f.readlines()]

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

                df = subsampler(df,10000)

            if self.training is False:
                self.all_starts = df["start"].tolist()
                self.all_ends = df["end"].tolist()

            if category_col != "Pfam_id":
                df.drop(columns=[category_col], inplace=True)
            else:
                if domain_boundary_detection is False:
                    df.drop(columns=["start", "end", "id", "Pfam_id"], inplace=True)
                else:
                    df.drop(columns=["id", "Pfam_id"], inplace=True)
            

            df = df[
                df[sequence_col].str.len() >= 10
            ]                                           # 10 for biological reasonings
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
            if RANK == 0:
                print("Switching labels for domain boundary detection")
            labels_list = []
            for index, row in self.df.iterrows():
                seq_len = len(row[sequence_col])
                label = [0] * seq_len
                start = int(row["start"])
                end = int(row["end"])
                if start < seq_len and end < seq_len:
                    for pos in range(start, end + 1):
                        label[pos] = 1


                labels_list.append(torch.tensor(label, dtype=torch.long))

            self.labels = labels_list  # Store as list of tensors
            # print(self.labels[3:4])  # Print first 5 labels for verification
            # print("SHAPE OF LABELS:")
            # print(self.labels[0].shape)


        start_time = time.time()

        if self.training is False:
            self.embeddings, self.labels, self.starts, self.ends = self._embed(
                sequences
            )
        else:
            self.embeddings, self.labels = self._embed(sequences)


        if self.domain_boundary_detection is True:
            print("Closing ESM Embedder")
            return  # Return early if domain boundary detection is enabled


        if self.domain_boundary_detection is False:

            self.embeddings = self.embeddings.cpu()
            self.labels = self.labels.cpu()
            if self.training is False:
                self.starts = self.starts.cpu()
                self.ends = self.ends.cpu()

            if len(self.embeddings) != len(self.labels):
                if RANK == 0:
                    print(
                        f"WARNING: Number of embeddings does not match number of labels! {len(self.embeddings)} != {len(self.labels)}"
                    )
                if len(self.embeddings) > len(self.labels):
                    self.embeddings = self.embeddings[
                        : len(self.labels)
                    ]  # Discarding last embedding, due to being a duplicate
                    if self.training is False:
                        self.starts = self.starts[: len(self.labels)]
                        self.ends = self.ends[: len(self.labels)]
                    print(f"After fix length: {len(self.embeddings)} == {len(self.labels)}")
                else:
                    raise ValueError(
                        f"Number of embeddings is less than number of labels! {len(self.embeddings)} < {len(self.labels)}, check your data!"
                    )

            print(
                f"{RANK}: embedding 1, label 1: {self.embeddings[0].shape}, {self.labels[0]}"
            )


        end_time = time.time()
        embedding_time = end_time - start_time

        # Print timing information
        print("Embedding generation completed!")
        print(
            f"Total time: {embedding_time:.2f} seconds ({embedding_time / 60:.2f} minutes)"
        )
        print(f"Time per sequence: {embedding_time / len(sequences):.4f} seconds")
        print(f"Sequences per second: {len(sequences) / embedding_time:.2f}")


        if self.training is True:
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

        if self.domain_boundary_detection is True:

            # DATASET AND DATALOADER FOR DOMAIN BOUNDARY DETECTION NEEDED


            class DomainBoundaryDataset(Dataset):
                """Custom dataset for domain boundary detection with ESM embeddings that loads data from an H5 file."""

                def __init__(self, h5_file):
                    self.h5_file = h5_file
                    self.file = None  # Will be opened in __getitem__
                    
                    # Determine the total length by inspecting the H5 file
                    with h5py.File(self.h5_file, "r") as f:
                        self.embedding_keys = sorted([k for k in f.keys() if k.startswith("batch_")])
                        self.chunk_sizes = [f[key].shape[0] for key in self.embedding_keys]
                        self.cumulative_sizes = np.cumsum(self.chunk_sizes)
                        self.length = self.cumulative_sizes[-1] if self.cumulative_sizes.size > 0 else 0

                def __len__(self):
                    return self.length

                def __getitem__(self, idx):
                    if idx < 0 or idx >= self.length:
                        raise IndexError("Index out of range")

                    # Open the file here to be safe for multiprocessing
                    if self.file is None:
                        self.file = h5py.File(self.h5_file, 'r')

                    # Find which chunk the index belongs to
                    chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
                    if chunk_idx > 0:
                        local_idx = idx - self.cumulative_sizes[chunk_idx - 1]
                    else:
                        local_idx = idx

                    embedding_key = self.embedding_keys[chunk_idx]
                    # Construct corresponding keys for other data
                    key_suffix = embedding_key.replace("batch_", "")
                    labels_key = f"labels_batch_{key_suffix}"
                    starts_key = f"starts_batch_{key_suffix}"
                    ends_key = f"ends_batch_{key_suffix}"

                    embeddings = torch.tensor(self.file[embedding_key][local_idx], dtype=torch.float32)
                    labels = torch.tensor(self.file[labels_key][local_idx], dtype=torch.long)
                    
                    # Assuming starts and ends are also saved per chunk
                    starts = torch.tensor(self.file[starts_key][local_idx], dtype=torch.long)
                    ends = torch.tensor(self.file[ends_key][local_idx], dtype=torch.long)
                    
                    return embeddings, labels, starts, ends

                def __del__(self):
                    if self.file:
                        self.file.close()


                
            # Create the dataset and dataloader
            self.domain_boundary_dataset = DomainBoundaryDataset("./temp/embeddings_domain.h5")

            # Split indices for train and validation sets
            dataset_size = len(self.domain_boundary_dataset)
            indices = list(range(dataset_size))

            # Split indices into training and validation
            val_size = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
            train_indices, val_indices = train_test_split(
                indices,
                test_size=val_size,
                shuffle=True,
                random_state=42
            )

            # Create Subset datasets
            train_dataset = torch.utils.data.Subset(self.domain_boundary_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(self.domain_boundary_dataset, val_indices)

            # Create DataLoaders for each subset
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=EMB_BATCH,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=EMB_BATCH,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            print(f"Train set: {len(train_dataset)} samples")
            print(f"Val set: {len(val_dataset)} samples")
            print("Datasets and DataLoaders for domain boundary detection created.")



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
        def windower(stepsize=1000,dimension=1000):
            # cut sequences longer than 1000 characters into sliding windows, because of ESM2 limitations
            new_seqs = []
            new_labels = []
            count = 0
            if self.training is True:
                for seq, label in zip(seq_dataset.seqs, seq_dataset.labels):
                    if len(seq) > 1000:
                        if RANK == 0:
                            count += 1
                        slices = [
                            seq[i : i + dimension]
                            for i in range(0, len(seq) - dimension + 1, stepsize)
                        ]
                        if len(seq) % dimension != 0:
                            slices.append(seq[-dimension:])
                        new_seqs.extend(slices)
                        new_labels.extend([label] * len(slices))
                    else:
                        new_seqs.append(seq)
                        new_labels.append(label)
                self.end_window = [len(seq) for seq in new_seqs]
                print(
                    f"Warning: {count} sequences were longer than 1000 characters and slided into windows"
                )

            if self.training is False:
                new_starts = []
                new_ends = []
                for seq, label, start, end in zip(
                    seq_dataset.seqs,
                    seq_dataset.labels,
                    seq_dataset.starts,
                    seq_dataset.ends,
                ):
                    if len(seq) > 1000:
                        if RANK == 0:
                            count += 1
                        slices = []
                        slice_positions = []
                        for i in range(0, len(seq) - dimension + 1, stepsize):
                            slices.append(seq[i : i + dimension])
                            slice_positions.append((i, i + dimension))
                        
                        if len(seq) % dimension != 0:
                            slices.append(seq[-dimension:])
                            slice_positions.append((len(seq) - dimension, len(seq)))
                        
                        new_seqs.extend(slices)

                        for (slice_start, slice_end) in slice_positions:
                            # Always window the label to match the sequence window (dimension=1000)
                            windowed_label = label[slice_start:slice_end]  # Slice the label
                            
                            # Ensure windowed_label is exactly dimension length
                            if windowed_label.shape[0] < dimension:
                                windowed_label = torch.nn.functional.pad(
                                    windowed_label, (0, dimension - windowed_label.shape[0]), value=0
                                )
                            elif windowed_label.shape[0] > dimension:
                                windowed_label = windowed_label[:dimension]  # Truncate to dimension
                            
                            new_labels.append(windowed_label)
                        
                        new_starts.extend([start] * len(slices))
                        new_ends.extend([end] * len(slices))
                    else:
                        new_seqs.append(seq)
                        # Ensure label is padded/truncated to dimension=1000 for consistency
                        if label.shape[0] < dimension:
                            padded_label = torch.nn.functional.pad(
                                label, (0, dimension - label.shape[0]), value=0
                            )
                        elif label.shape[0] > dimension:
                            padded_label = label[:dimension]
                        else:
                            padded_label = label
                        
                        new_labels.append(padded_label)
                        new_starts.append(start)
                        new_ends.append(end)

                self.end_window = [len(seq) for seq in new_seqs]
                print(
                    f"Warning: {count} sequences were longer than 1000 characters and slided into windows"
                )

                # Update the dataframe with the new sequences and labels
                if len(new_seqs) != len(seq_dataset.seqs):
                    # Create new dataframe with updated sequences
                    new_df_data = []
                    for i, (seq, label) in enumerate(zip(new_seqs, new_labels)):
                        new_df_data.append({self.sequence_col: seq, "label": label})

                    # If we have starts and ends, add them too
                    if self.training is False:
                        for i, (start, end) in enumerate(zip(new_starts, new_ends)):
                            new_df_data[i]["start"] = start
                            new_df_data[i]["end"] = end

                    # Update the dataframe
                    self.df = pd.DataFrame(new_df_data)

                    if RANK == 0:
                        print(
                            f"Updated dataframe from {len(seq_dataset.seqs)} to {len(new_seqs)} sequences due to sliding window"
                        )


            if self.training is True:
                return new_seqs, new_labels
            else:
                return new_seqs, new_labels, new_starts, new_ends
        

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
        all_labels = []
        if self.domain_boundary_detection is True or self.training is False:
            all_starts = []
            all_ends = []

        seqs = list(seqs)

        start_time = time.time()

        # Use the new dataset that includes labels
        if self.training is True and self.skip_df is None:
            seq_dataset = SeqDataset(seqs, self.labels)
        else:
            seq_dataset = SeqDatasetForEval(
                seqs, self.labels, self.all_starts, self.all_ends
            )

        if self.training is True:
            new_seqs, new_labels = windower(stepsize=500, dimension=1000)
        else:
            new_seqs, new_labels, new_starts, new_ends = windower(
                stepsize=500, dimension=1000
            )

        seq_dataset.seqs = new_seqs
        seq_dataset.labels = new_labels
        if self.training is False:
            seq_dataset.starts = new_starts
            seq_dataset.ends = new_ends



        if self.domain_boundary_detection is True:
            # Convert seq_dataset.labels to a tensor if it's a list
            if isinstance(seq_dataset.labels, list):
                if isinstance(seq_dataset.labels[0], torch.Tensor):
                    seq_dataset.labels = torch.stack(seq_dataset.labels)
                else:
                    seq_dataset.labels = torch.tensor(seq_dataset.labels, dtype=torch.long)

            print(seq_dataset.labels.shape) 

        # if RANK == 0:
        #     print( f"Seq: {seq_dataset.seqs[0:5]}, Label: {seq_dataset.labels[0:5]}")

        # Create a DataLoader with DistributedSampler for distributed training
        sampler = DistributedSampler(
            seq_dataset,
            num_replicas=dist.get_world_size(),
            rank=RANK,
            shuffle=False,
            drop_last=False,
        )

        dataloader = DataLoader(
            seq_dataset,
            batch_size=EMB_BATCH,
            num_workers=0,
            sampler=sampler,
            pin_memory=True,
        )

        if RANK == 0:
            print(f"Expected embedding dimension: {expected_dim}")
            print(
                f"Total sequences to process: {len(seq_dataset.seqs)} with batch size {EMB_BATCH}"
            )
        print(f"Each Rank will process sequences {sampler.num_samples}")

        chunk = 1

        seq_dataset = None  # Clear the dataset to free memory
        for batch_num, batch_data in enumerate(dataloader):
            # if batch_num % 100 == 0 and RANK == 0:
                # all_objects = muppy.get_objects()
                # sum1 = summary.summarize(all_objects)
                # del all_objects  # Clear the list to free memory
            if self.training is True:
                batch_seqs, batch_labels = batch_data
            else:
                batch_seqs, batch_labels, batch_start, batch_end = batch_data
                
                if batch_num == 0 and RANK == 0:
                    print(batch_seqs, batch_labels, batch_start, batch_end)

            try:
                # batch_seqs and batch_labels are now properly paired
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

                mask = batch_tokens != 1
                mask = mask[:, 1:-1]
                embeddings = embeddings[:, 1:-1, :]         #remove <eos> and <cls> tokens, made by esm model (start and end tokens)

                if RANK == 0 and batch_num < 2:  # Print first 2 batches for debugging
                    print(
                        f"Batch {batch_num + 1}: {embeddings.shape} embeddings, {mask.shape} mask"
                    )
                    print(embeddings[0:2, 0:5, 0:5])

                if self.domain_boundary_detection is True:
                    # no pooling, just store the embeddings as they are

                    all_embeddings.append(embeddings)
                    all_labels.append(batch_labels)


                    # if RANK == 0:
                    #     print(type(all_embeddings), "type of embeddings before move")
                    #     print(type(all_labels), "type of labels before move")
                    
                    if batch_num % 1000 == 0 and batch_num > 0:
                        # # Move the previous chunk to CPU in batch to stop vram overflow, but yet keep speed high
                        
                        # start_idx = 0
                        # end_idx = min(1001, len(all_embeddings))

                        start_idx = (chunk-1) * 1000
                        end_idx = min((chunk * 1000)+1, len(all_embeddings))
                        
                        # print(start_idx, end_idx, len(all_embeddings), len(all_labels))

                        # Batch move to CPU (more efficient)
                        for i in range(start_idx, end_idx):
                            # if RANK == 0:
                            #     print(type(all_embeddings), "type of embeddings")
                                    
                            all_embeddings[i] = all_embeddings[i].cpu()
                        
                            # if RANK == 0:
                                # print(type(all_embeddings), "type of embeddings")
                                

                        for i in range(start_idx, end_idx):
                            all_labels[i] = all_labels[i].cpu()
                        
                        chunk += 1

                        # Optional: Check for any remaining CUDA tensors, sanity check
                        for i, emb in enumerate(all_embeddings):
                            if emb.is_cuda:
                                print(f"Warning: embedding {i} still on GPU")
        

                else:
                    # For ID classification, we pool the embeddings
                    lengths = mask.sum(dim=1, keepdim=True)
                    pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / lengths
                    pooled = pooled.cpu()  # Move to CPU immediately

                    all_embeddings.append(pooled)
                    all_labels.append(batch_labels)  # Store the corresponding labels
                if self.training is False or self.domain_boundary_detection is True:
                    all_starts.append(batch_start)
                    all_ends.append(batch_end)

                # Progress reporting, nothing special here
                if batch_num % 100 == 0 or batch_num == len(dataloader) - 1 and RANK == 0:   
                    elapsed_time = time.time() - start_time
                    if batch_num > 0:
                        avg_time_per_batch = elapsed_time / (batch_num + 1)
                        remaining_batches = len(dataloader) - batch_num - 1
                        eta_seconds = remaining_batches * avg_time_per_batch




                        eta_hours = int(eta_seconds // 3600)
                        eta_minutes = int((eta_seconds % 3600) // 60)
                        eta_seconds_remainder = int(eta_seconds % 60)
                        if RANK == 0:
                            # summary.print_(sum1)

                            # Add RAM monitoring
                            process = psutil.Process(os.getpid())
                            memory_info = process.memory_info()
                            memory_gb = memory_info.rss / 1024 / 1024 / 1024
                            
                            system_memory = psutil.virtual_memory()
                            total_gb = system_memory.total / 1024 / 1024 / 1024
                            available_gb = system_memory.available / 1024 / 1024 / 1024
                            used_percent = system_memory.percent
                            
                            print(f"\n=== Memory Status at Batch {batch_num} ===")
                            print(f"Process RAM: {memory_gb:.2f} GB")
                            print(f"System RAM: {used_percent:.1f}% used ({total_gb-available_gb:.1f}/{total_gb:.1f} GB)")
                            print(f"Available RAM: {available_gb:.1f} GB")

                        print(
                            f"Rank {RANK}: Batch {batch_num + 1}/{len(dataloader)} | "
                            f"ETA: {eta_hours}h {eta_minutes}m {eta_seconds_remainder}s",
                            end="\r",
                            flush=True,
                        )






            except Exception as e:                                                  
                # Handle exceptions during batch processing, if vram blows up, we will try to process each sequence individually
                # kind of unnecessary, cuz emb_batch is already 1 is the fastes (no padding the batch to same length needed, due to 1 sequence only), but was already implemented
                # can be ignored because it never triggers during ram oom
                print(f"\nRank {RANK}: Error processing batch {batch_num + 1}: {e}")
                # Handle individual sequences...
                for seq, label in zip(batch_seqs, batch_labels):
                    try:
                        single_seq = [("seq0", seq)]
                        single_labels, single_strs, single_tokens = (
                            self.batch_converter(single_seq)
                        )
                        single_tokens = single_tokens.cuda()

                        with torch.no_grad():
                            results = self.model(
                                single_tokens,
                                repr_layers=[self.model_layers],
                                return_contacts=False,
                            )
                            embedding = results["representations"][
                                self.model_layers
                            ].float()

                            mask = single_tokens != 1
                            mask = mask[:, 1:-1]
                            embedding = embedding[:, 1:-1, :]

                            length = mask.sum(dim=1, keepdim=True)

                            if self.domain_boundary_detection is True:
                                embedding = embedding.cpu()  # Move to CPU immediately
                                all_embeddings.append(embedding)
                                all_labels.append(batch_labels)  # Keep label paired

                            else:
                                pooled = (embedding * mask.unsqueeze(-1)).sum(
                                    dim=1
                                ) / length
                                pooled = pooled.cpu()

                                all_embeddings.append(pooled)
                                all_labels.append(batch_labels)  # Keep label paired
                            if self.training is False or self.domain_boundary_detection is True:
                                all_starts.append(torch.tensor([0], dtype=torch.long))
                                all_ends.append(
                                    torch.tensor([len(seq)], dtype=torch.long)
                                )

                    except Exception as single_e:
                        # last fallback, empty embedding, if something goes wrong
                        # never triggers, at least never seen it trigger
                        print(
                            f"Rank {RANK}: Error processing individual sequence: {single_e}"
                        )
                        zero_embedding = torch.zeros(1, expected_dim)
                        all_embeddings.append(zero_embedding)
                        all_labels.append(label.unsqueeze(0))  # Still keep the label
                        if self.training is False or self.domain_boundary_detection is True:
                            all_starts.append(torch.tensor([0], dtype=torch.long))
                            all_ends.append(torch.tensor([len(seq)], dtype=torch.long))


            if self.domain_boundary_detection is True:
                # Save embeddings and labels in chunks to avoid memory overflow
                # on palma i do only 100 
                if batch_num % 1000 == 0 and batch_num > 0 or batch_num == len(dataloader) - 1: 

                    max_len = 1000

                    # if RANK == 0:
                        # print(all_labels[0].shape, "labels shape before padding")


                    # Verify all embeddings and labels are already the correct size
                    for i, (emb, lbl) in enumerate(zip(all_embeddings, all_labels)):
                        if emb.shape[1] != max_len:
                            all_embeddings[i] = torch.nn.functional.pad(
                                emb, (0, 0, 0, max_len - emb.shape[1]), value=0
                            )

                    #     if lbl.shape[0] != max_len:
                    #         all_labels[i] = torch.nn.functional.pad(
                    #             lbl, (0, max_len - lbl.shape[0]), value=0
                    #         )

                    # if RANK == 0:
                    #     print(all_labels[0].shape, "labels shape after padding")


                    # Convert list of tensors to a single tensor, then to numpy
                    embeddings_tensor = torch.stack([emb.cpu() for emb in all_embeddings])
                    labels_tensor = torch.stack([lbl.cpu() for lbl in all_labels])


                    dist.barrier()  # Ensure all ranks are synchronized before saving

                    if RANK == 0:
                        print("\n\nWriting...")
                    for rank_id in range(dist.get_world_size()):
                        if RANK == rank_id:
                            with h5py.File("./temp/embeddings_domain.h5", "a") as f:
                                f[f"batch_{batch_num}_rank_{RANK}"] = embeddings_tensor.numpy()
                                f[f"labels_batch_{batch_num}_rank_{RANK}"] = labels_tensor.numpy()
                        dist.barrier()  # Ensure only one rank writes at a time
                    if RANK == 0:
                        print(
                            f"Wrote embeddings and labels for batch {batch_num} to file"
                        )


                    # Clear the lists to free memory, should do the job, but somewhere i leak memory

                    del embeddings_tensor, labels_tensor

                    # Clear references before deleting the lists
                    all_embeddings.clear()
                    all_labels.clear()

                    # Reinitialize the lists
                    all_embeddings = []
                    all_labels = []

                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache()
                    chunk = 1

            else:
                # basic way for ID classification, just store the embeddings and labels
                
                # Concatenate local embeddings and labels
                local_embeddings = (
                    torch.cat(all_embeddings, dim=0)
                    if all_embeddings
                    else torch.empty(0, expected_dim)
                )
                local_labels = (
                    torch.cat(all_labels, dim=0)
                    if all_labels
                    else torch.empty(0, dtype=torch.long)
                )
            if self.training is False:
                local_starts = (
                    torch.cat(all_starts, dim=0)
                    if all_starts
                    else torch.empty(0, dtype=torch.long)
                )
                local_ends = (
                    torch.cat(all_ends, dim=0)
                    if all_ends
                    else torch.empty(0, dtype=torch.long)
                )

        #     # Print the shapes of the gathered data
        #     print(
        #         f"Rank {RANK}: Generated {local_embeddings.size(0)} local embeddings with {local_labels.size(0)} labels"
        #     )
        #     print(
        #         f"Rank {RANK}: Generated {local_starts.size(0)} starts with {local_ends.size(0)} ends"
        #     )

        # print(
        #     f"Rank {RANK}: Generated {local_embeddings.size(0)} local embeddings with {local_labels.size(0)} labels"
        # )



        # Synchronize all processes, so all embeddings are done
        dist.barrier()

        print("embeddings DONE!")


        # ------------------------------------------------------
        #   BROADCASTING AND GATHERING EMBEDDINGS AND LABELS
        # ------------------------------------------------------
        # used in classification tasks, where we need to gather all embeddings and labels from all ranks,
        # tried my way here, it works, but some parts i dont understand fully
        # NOT NEEDED FOR DOMAIN BOUDNARY TASK


        if dist.get_world_size() > 1 and self.domain_boundary_detection is False:
            # Gather embeddings and labels from all processes
            local_size = torch.tensor([local_embeddings.size(0)], dtype=torch.int64, device="cuda")
            all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
            dist.all_gather(all_sizes, local_size)

            total_size = sum(size.item() for size in all_sizes)

            if RANK == 0:
                # Dynamically determine the maximum sequence length for embeddings
                max_seq_len = max(emb.shape[0] for emb in all_embeddings)  # Use the second dimension (seq_len)

                # Initialize final_embeddings with the correct dimensions
                final_embeddings = torch.empty(total_size, max_seq_len, expected_dim, dtype=torch.float32)

                # Initialize final_labels with the correct dimensions
                max_label_seq_len = max(lab.shape[0] for lab in all_labels)  # Use the second dimension (seq_len)
                final_labels = torch.empty(total_size, max_label_seq_len, dtype=torch.long)

                if self.training is False:
                    final_starts = torch.empty(total_size, dtype=torch.long)
                    final_ends = torch.empty(total_size, dtype=torch.long)

                current_idx = 0
                for rank in range(dist.get_world_size()):
                    rank_size = all_sizes[rank].item()
                    if rank_size > 0:
                        if rank == 0:
                            final_embeddings[current_idx:current_idx + rank_size] = local_embeddings
                            final_labels[current_idx:current_idx + rank_size] = local_labels
                            if self.training is False:
                                final_starts[current_idx:current_idx + rank_size] = local_starts
                                final_ends[current_idx:current_idx + rank_size] = local_ends
                        else:
                            rank_embeddings = torch.empty(rank_size, max_seq_len, expected_dim, dtype=torch.float32, device="cuda")
                            dist.recv(rank_embeddings, src=rank)
                            final_embeddings[current_idx:current_idx + rank_size] = rank_embeddings.cpu()

                            rank_labels = torch.empty(rank_size, max_label_seq_len, dtype=torch.long, device="cuda")
                            dist.recv(rank_labels, src=rank)
                            final_labels[current_idx:current_idx + rank_size] = rank_labels.cpu()

                            if self.training is False:
                                rank_starts = torch.empty(rank_size, dtype=torch.long, device="cuda")
                                dist.recv(rank_starts, src=rank)
                                final_starts[current_idx:current_idx + rank_size] = rank_starts.cpu()

                                rank_ends = torch.empty(rank_size, dtype=torch.long, device="cuda")
                                dist.recv(rank_ends, src=rank)
                                final_ends[current_idx:current_idx + rank_size] = rank_ends.cpu()

                            del rank_embeddings, rank_labels
                            if self.training is False:
                                del rank_starts, rank_ends
                            torch.cuda.empty_cache()
                        current_idx += rank_size
            else:
                if local_embeddings.size(0) > 0:
                    local_embeddings_gpu = local_embeddings.cuda()
                    dist.send(local_embeddings_gpu, dst=0)
                    del local_embeddings_gpu

                    local_labels_gpu = local_labels.cuda()
                    dist.send(local_labels_gpu, dst=0)
                    del local_labels_gpu

                    if self.training is False:
                        local_starts_gpu = local_starts.cuda()
                        dist.send(local_starts_gpu, dst=0)
                        del local_starts_gpu

                        local_ends_gpu = local_ends.cuda()
                        dist.send(local_ends_gpu, dst=0)
                        del local_ends_gpu

                    torch.cuda.empty_cache()

            # Broadcast final data from rank 0 to all ranks
            if RANK == 0:
                size_tensor = torch.tensor([final_embeddings.size(0)], device="cuda")
            else:
                size_tensor = torch.tensor([0], device="cuda")

            dist.broadcast(size_tensor, 0)

            total_size = size_tensor.item()
            chunk_size = 10000
            if self.domain_boundary_detection is True:
                chunk_size = 1000  # Smaller chunk size for domain boundary detection


            if RANK == 0:
                max_label_seq_len = max(lab.shape[0] for lab in all_labels)  # Calculate on rank 0
                max_seq_len = max(emb.shape[0] for emb in all_embeddings)

            else:
                max_label_seq_len = 0  # Placeholder on other ranks
                max_seq_len = 0


            # Synchronize max_label_seq_len across all ranks
            max_label_seq_len_tensor = torch.tensor([max_label_seq_len], device="cuda")
            dist.all_reduce(max_label_seq_len_tensor, op=dist.ReduceOp.MAX)
            max_label_seq_len = max_label_seq_len_tensor.item()
            max_seq_len_tensor = torch.tensor([max_seq_len], device="cuda")
            dist.all_reduce(max_seq_len_tensor, op=dist.ReduceOp.MAX)
            max_seq_len = max_seq_len_tensor.item()
            
            if RANK != 0:
                final_labels = torch.empty(total_size, max_label_seq_len, dtype=torch.long, device="cuda")
                final_embeddings = torch.empty(total_size, max_seq_len, expected_dim, dtype=torch.float32)
                
                if self.training is False:
                    final_starts = torch.empty(total_size, dtype=torch.long)
                    final_ends = torch.empty(total_size, dtype=torch.long)

            for start_idx in range(0, total_size, chunk_size):
                end_idx = min(start_idx + chunk_size, total_size)
                chunk_len = end_idx - start_idx

                # Broadcast embeddings chunk
                if RANK == 0:
                    chunk_gpu = final_embeddings[start_idx:end_idx].cuda()
                else:
                    chunk_gpu = torch.empty(chunk_len, max_seq_len, expected_dim, dtype=torch.float32, device="cuda")
                dist.broadcast(chunk_gpu, 0)
                if RANK != 0:
                    final_embeddings[start_idx:end_idx] = chunk_gpu.cpu()
                del chunk_gpu

                # Broadcast labels chunk
                if RANK == 0:
                    labels_chunk_gpu = final_labels[start_idx:end_idx].cuda()
                else:
                    labels_chunk_gpu = torch.empty(chunk_len, max_label_seq_len, dtype=torch.long, device="cuda")
                dist.broadcast(labels_chunk_gpu, 0)
                if RANK != 0:
                    final_labels[start_idx:end_idx] = labels_chunk_gpu.cpu()
                del labels_chunk_gpu

                if self.training is False:
                    # Broadcast starts chunk
                    if RANK == 0:
                        starts_chunk_gpu = final_starts[start_idx:end_idx].cuda()
                    else:
                        starts_chunk_gpu = torch.empty(chunk_len, dtype=torch.long, device="cuda")
                    dist.broadcast(starts_chunk_gpu, 0)
                    if RANK != 0:
                        final_starts[start_idx:end_idx] = starts_chunk_gpu.cpu()
                    del starts_chunk_gpu

                    # Broadcast ends chunk
                    if RANK == 0:
                        ends_chunk_gpu = final_ends[start_idx:end_idx].cuda()
                    else:
                        ends_chunk_gpu = torch.empty(chunk_len, dtype=torch.long, device="cuda")
                    dist.broadcast(ends_chunk_gpu, 0)
                    if RANK != 0:
                        final_ends[start_idx:end_idx] = ends_chunk_gpu.cpu()
                    del ends_chunk_gpu

                torch.cuda.empty_cache()




        elif self.domain_boundary_detection is True:
            # # Counterpart for domain boundary detection, where we gather all embeddings and labels from all ranks via their .h5 files            
            # # Open embeddings and labels from the single h5 file
            # all_embeddings = []
            # all_labels = []

            # if RANK == 0:
            #     print("Gathering embeddings and labels from single h5 file...")

            # try:
            #     with h5py.File(f"./temp/embeddings_doamain.h5", "r") as f:
            #         # Get all keys and sort them for consistent ordering
            #         embedding_keys = sorted([key for key in f.keys() if "batch" in key and not key.startswith("labels_")])
                    
            #         for key in embedding_keys:
            #             # Load embeddings
            #             embeddings = torch.tensor(f[key][:], dtype=torch.float32)
            #             all_embeddings.append(embeddings)
                        
            #             # Construct corresponding labels key
            #             labels_key = f"labels_{key}"
            #             if labels_key in f:
            #                 labels = torch.tensor(f[labels_key][:], dtype=torch.long)
            #                 all_labels.append(labels)
            #             else:
            #                 print(f"Warning: No labels found for key {key}")
            # except FileNotFoundError:
            #     print(f"Warning: File ./temp/embeddings_doamain.h5 not found")
            #     # Initialize empty lists as fallback
            #     all_embeddings = []
            #     all_labels = []

            # # Concatenate all embeddings and labels
            # final_embeddings = torch.cat(all_embeddings, dim=0)
            # final_labels = torch.cat(all_labels, dim=0)
            # if self.training is False:
            #     final_starts = torch.cat(all_starts, dim=0)
            #     final_ends = torch.cat(all_ends, dim=0)

            # # Safety check if the embeddings and labels have the expected dimensions
            # if final_embeddings.dim() == 4 and final_labels.dim() == 3:
            #     final_embeddings = final_embeddings.squeeze(1)
            #     final_labels = final_labels.squeeze(1)
            #     if RANK == 0:
            #         print("Squeezed embeddings and labels to correct dimensions.")

            # if RANK == 0 and len(all_embeddings) > 0:
            #     print(f"Loaded {final_embeddings.shape[0]} total samples")
            #     print(f"Final embeddings shape: {final_embeddings.shape}")
            #     print(f"Final labels shape: {final_labels.shape}")
            pass
            

        else:
            # if no ddp is used, just return the local embeddings and labels
            final_embeddings = local_embeddings
            final_labels = local_labels
            if self.training is False:
                final_starts = local_starts
                final_ends = local_ends

        # At the end of _embed function, return the final embeddings and labels
        if self.training is False and self.domain_boundary_detection is False:
            return final_embeddings, final_labels, final_starts, final_ends
        elif self.domain_boundary_detection is True:
            return None, None, None, None
        else:
            return final_embeddings, final_labels


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not intended to be run directly"
    )
