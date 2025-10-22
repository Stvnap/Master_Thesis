import datetime
import gc
import os
import subprocess
import time
import tqdm
import esm
import h5py
import pandas as pd
import psutil
import torch
import torch.distributed as dist
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
os.environ["NCCL_TIMEOUT"] = "36000000"  # 10 hours in milliseconds
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
world_size = int(os.environ.get("WORLD_SIZE", 1))
use_ddp = world_size > 1

# print(torch.__version__)


if not dist.is_initialized():
    # Set up environment variables if not already set
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        timeout=datetime.timedelta(seconds=36000),
    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if use_ddp and torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        if dist.get_rank() == 0:
            print("Initializing process group for DDP with device: CUDA")
    RANK = dist.get_rank()
    if RANK == 0:
        print(
            f"Start running basic DDP example with worldsize {dist.get_world_size()}."
        )
    # create model and move it to GPU with id rank
    DEVICE_ID = RANK % torch.cuda.device_count()
else:
    # Single-GPU fallback for quick testing
    DEVICE_ID = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RANK = 0
    print("Running in single-GPU mode without DDP.")


TRAIN_FRAC = 0.8
VAL_FRAC = 0.2


VRAM = psutil.virtual_memory().total // (1024 ** 3)  # in GB
EMB_BATCH = 32 if VRAM >= 24 else 16 if VRAM >= 16 else 8 if VRAM >= 8 else 4


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
        emb_batch=EMB_BATCH,
        skip_df=None,
        FSDP_used=False,
        domain_boundary_detection=False,
        training=False,
    ):
        self.esm_model = esm_model
        self.training = training
        # print(f"Training mode: {self.training}")
        self.skip_df = skip_df
        self.domain_boundary_detection = domain_boundary_detection
        self.sequence_col = sequence_col
        self.num_classes = num_classes
        self.FSDP_used = FSDP_used
        self.usage_mode = False
        self.model, self.batch_converter = self.esm_loader()
        self.all_starts = None
        self.all_ends = None
        self.emb_batch = emb_batch

        def map_label(cat):
            # Adaptive mapping based on NUM_CLASSES
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

        def subsampler(df, target, first_subsampler):
            if target is None:
                return df

            # Count samples per class
            class_counts = df["label"].value_counts()

            # Subsample class 0  to target samples each
            target_samples_per_class = target
            subsampled_dfs = []

            # if RANK == 0:
            #     print("Subsampling all classes to 100 samples each:")

            for class_label in sorted(class_counts.index):
                class_samples = df[df["label"] == class_label]
                current_count = len(class_samples)

                if class_label == 0 and current_count > target_samples_per_class:
                    # Subsample to target size
                    class_subsampled = class_samples.sample(
                        n=target_samples_per_class, random_state=42
                    )
                    # if RANK == 0:
                    #     print(f"  Class {class_label}: {current_count} -> {target_samples_per_class}")
                else:
                    # Keep all samples if less than target
                    class_subsampled = class_samples
                    # if RANK == 0:
                    #     print(f"  Class {class_label}: {current_count} (kept all)")

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
                        if first_subsampler is True:
                            print(
                                f"Class {i} with ID {pfam_id}: {final_counts[i]} from samples | avg length {df[df['label'] == i][sequence_col].str.len().mean():.2f}"
                            )

                        all_pfam.append(pfam_id)

                if self.training is True:
                    os.makedirs("/scratch/tmp/sapelt/Master_Thesis/temp", exist_ok=True)
                    all_pfam = all_pfam[1:]  # Exclude "other" category
                    open(f"/scratch/tmp/sapelt/Master_Thesis/temp/selected_pfam_ids_{num_classes - 1}.txt", "w").write(
                        "\n".join(all_pfam)
                    )
            return df

        if skip_df is None:
            if domain_boundary_detection is False:
                if RANK == 0:
                    print("Loading data...")

                # --- START: NEW PRE-COMPUTATION LOGIC ---
                # This block calculates global Pfam counts before the main processing.
                # It's memory-efficient because it only loads one column per chunk.
                selected_ids_file = f"/scratch/tmp/sapelt/Master_Thesis/temp/selected_pfam_ids_{num_classes - 1}.txt"
                
                if self.training is True and not os.path.exists(selected_ids_file):
                    if RANK == 0:
                        print("Performing initial pass to determine global class distribution...")
                    global_pfam_counts = pd.Series(dtype=int)
                    # Create a lightweight iterator that only reads the category column.
                    try:
                        count_iter = pd.read_csv(csv_path, usecols=[category_col], chunksize=10000000)
                        for chunk in count_iter:
                            global_pfam_counts = global_pfam_counts.add(chunk[category_col].value_counts(), fill_value=0)
                    except Exception as e:
                        print(f"Error during pre-computation pass: {e}")
                        # Handle error, maybe exit or fall back to old method with a warning.


                    print("Global counts calculated. Selecting top classes...")
                    # Now, use these global counts to create ONE definitive `selected_ids` list.
                    global_pfam_counts.sort_values(ascending=False, inplace=True)
                    frequent_pfam_ids = global_pfam_counts[global_pfam_counts > 10].index.tolist()
                    

                    
                    # Define the specific 10 Pfam IDs you want to use first | probably change here if thioset needed
                    # priority_pfam_ids = [
                    #     "PF00177",
                    #     "PF00210",
                    #     "PF00211",
                    #     "PF00215",
                    #     "PF00217",
                    #     "PF00406",
                    #     "PF00303",
                    #     "PF00246",
                    #     "PF00457",
                    #     "PF00502",
                    # ]

                    

                    # # FOR THE THIOLASESET ONLY
                    # priority_pfam_ids = [
                    #     "PF00108",
                    #     "PF00109",
                    #     "PF00195",
                    #     "PF01154",
                    #     "PF02797",
                    #     "PF02801",
                    #     "PF02803",
                    #     "PF07451",
                    #     "PF08392",
                    #     "PF08540",
                        # ]

                    # Your existing priority_pfam_ids list
                    priority_pfam_ids = [
                            "PF00177",
                            "PF00210",
                            "PF00211",
                            "PF00215",
                            "PF00406",
                            "PF00303",
                            "PF00246",
                            "PF00005",
                            "PF00072",
                            "PF00069",
                            "PF02518",
                            "PF07690",
                            "PF00528",
                            "PF00115",
                            "PF00271",
                            "PF00512",
                            "PF00078",
                            "PF00440",
                            "PF00106",
                            "PF03466",
                            "PF00270",
                            "PF00501",
                            "PF00126",
                            "PF13561",
                            "PF04055",
                            "PF00583",
                            "PF00535",
                            "PF04542",
                            "PF07992",
                            "PF12833",
                            "PF00004",
                            "PF00561",
                            "PF00155",
                            "PF00067",
                            "PF08240",
                            "PF07714",
                            "PF07715",
                            "PF00672",
                            "PF00171",
                            "PF00534",
                            "PF00702",
                            "PF13193",
                            "PF00107",
                            "PF00990",
                            "PF00009",
                            "PF00149",
                            "PF01370",
                            "PF00486",
                            "PF00392",
                            "PF00361",
                            "PF00196",
                            "PF00441",
                            "PF00001",
                            "PF02771",
                            "PF00664",
                            "PF00293",
                            "PF01381",
                            "PF02770",
                            "PF08281",
                            "PF00593",
                            "PF02653",
                            "PF00753",
                            "PF01979",
                            "PF12796",
                            "PF00077",
                            "PF00248",
                            "PF00076",
                            "PF00378",
                            "PF00589",
                            "PF00903",
                            "PF01266",
                            "PF01408",
                            "PF01926",
                            "PF00291",
                            "PF01546",
                            "PF13439",
                            "PF00226",
                            "PF08241",
                            "PF00122",
                            "PF00083",
                            "PF00169",
                            "PF01565",
                            "PF00665",
                            "PF00117",
                            "PF00202",
                            "PF00015",
                            "PF03144",
                            "PF07687",
                            "PF00563",
                            "PF00496",
                            "PF08245",
                            "PF00266",
                            "PF00027",
                            "PF13649",
                            "PF00651",
                            "PF00296",
                            "PF01494",
                            "PF00884",
                            "PF03372",
                            "PF00071",
                        ]

                    available_priority_ids = [pid for pid in priority_pfam_ids if pid in frequent_pfam_ids]
                    remaining_ids = [pid for pid in frequent_pfam_ids if pid not in priority_pfam_ids]
                    max_additional_ids = num_classes - 1 - len(available_priority_ids)
                    selected_ids = available_priority_ids + remaining_ids[:max_additional_ids]

                    # Save the globally determined list to a file.
                    with open(selected_ids_file, "w") as f:
                        for pfam_id in selected_ids:
                            f.write(f"{pfam_id}\n")
                    print(f"Saved {len(selected_ids)} globally selected Pfam IDs to file.")
            

                expected_chunks = 22

                if num_classes != 24381:
                    with open(selected_ids_file, "r") as f:
                        selected_ids = [line.strip() for line in f.readlines()]
                else: # For 24381 classes, use all Pfam IDs
                    with open("/scratch/tmp/sapelt/Master_Thesis/Dataframes/v3/all_pfamIDs.txt", "r") as f:
                        selected_ids = [line.strip() for line in f.readlines()]

                pfam_to_label = {pfam_id: i + 1 for i, pfam_id in enumerate(selected_ids)}


                first_subsampler = True


                # Check for existing progress and resume
                start_chunk = 0
                del_key = False
                if os.path.exists(f"/scratch/tmp/sapelt/Master_Thesis/temp/progress_{num_classes}.txt"):
                    with open(f"/scratch/tmp/sapelt/Master_Thesis/temp/progress_{num_classes}.txt", "r") as status_file:
                        lines = status_file.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if "Completed chunk" in last_line:  # Look for completed chunks
                                start_chunk = int(last_line.split(" ")[2]) + 1  # Start from NEXT chunk
                            else:
                                start_chunk = int(last_line.split(" ")[2])  # Start from last incomplete chunk
                                del_key = True
                            if RANK == 0:
                                print(f"Resuming from chunk {start_chunk}")

                # Create iterator once and skip processed chunks
                try:
                    chunk_iter = pd.read_csv(
                        csv_path,
                        usecols=["start", "end", "id", "Pfam_id", "Sequence"],
                        chunksize=10000000,
                    )
                except Exception as e:
                    if RANK == 0:
                        print(f"Warning: Could not read with specific columns, falling back to all columns. Error: {e}")
                    chunk_iter = pd.read_csv(csv_path, chunksize=10000000) #10000000
                    self.usage_mode = True

                # Skip processed chunks
                for _ in range(start_chunk):
                    try:
                        next(chunk_iter)
                    except StopIteration:
                        if RANK == 0:
                            print(f"All chunks processed. No more chunks to process.")
                            with open(f"/scratch/tmp/sapelt/Master_Thesis/temp/progress_{num_classes}.txt", "a") as status_file:
                                if RANK == 0:
                                    status_file.write(f"All chunks processed. Exiting.\n")
                        return

                # Process remaining chunks
                for chunk_num, chunk in enumerate(chunk_iter, start=start_chunk):
                    if RANK == 0:
                        print(f"Processing chunk {chunk_num}/{expected_chunks} with {len(chunk)} sequences")
                    
                    # Write processing status (before processing)
                    with open(f"/scratch/tmp/sapelt/Master_Thesis/temp/progress_{num_classes}.txt", "a") as status_file:
                        if RANK == 0:
                            status_file.write(f"Processing chunk {chunk_num}\n")
                    
                    if "start" in chunk.columns and "end" in chunk.columns:
                        if training is True:
                            if RANK == 0 and chunk_num == 0:
                                print(
                                    "Cutting sequences based on start and end positions"
                                )

                            chunk[sequence_col] = chunk.apply(minicutter, axis=1)

                        # Filter out sequences that are too short to ensure no NaN values
                        initial_count = len(chunk)

                        final_count = len(chunk)
                        if initial_count != final_count:
                            if RANK == 0:
                                print(
                                    f"Removed {initial_count - final_count} sequences with length <= 10"
                                )
                                print(f"Remaining sequences: {final_count}")

                    if category_col != "Pfam_id":
                        chunk["label"] = chunk[category_col].apply(map_label)

                    elif domain_boundary_detection is False and self.usage_mode is False:




                        chunk["label"] = (
                            chunk[category_col].map(pfam_to_label).fillna(0)
                        )

                        chunk["label"] = chunk["label"].astype(int)


                        chunk = subsampler(chunk, 100000, first_subsampler)

                    if self.training is False:
                        self.all_starts = chunk["start"].tolist()
                        self.all_ends = chunk["end"].tolist()

                    if category_col != "Pfam_id":
                        chunk.drop(columns=[category_col], inplace=True)
                    elif self.usage_mode is False:
                        chunk.drop(
                            columns=["start", "end", "id", "Pfam_id"], inplace=True
                        )

                    # print(chunk.columns)

                    chunk = chunk[
                        chunk[sequence_col].str.len() >= 10
                    ]  # 10 for biological reasonings
                    self.chunk = chunk
                    if RANK == 0:
                        print("Data loaded")

                    # Use sequences in original order
                    sequences = self.chunk[sequence_col].tolist()

                    if self.usage_mode is False:
                        self.labels = torch.tensor(
                            self.chunk["label"].values, dtype=torch.long
                        )

                    start_time = time.time()

                    if self.training is False:
                        (
                            embeddings,
                            labels,
                            starts,
                            ends,
                            idx_multiplied,
                        ) = self._embed(sequences)
                    else:
                        embeddings, labels, _, _, idx_multiplied = self._embed(
                            sequences
                        )

                    embeddings = embeddings.cpu()
                    labels = labels.cpu()
                    if self.training is False:
                        starts = starts.cpu()
                        ends = ends.cpu()

                    if len(embeddings) != len(labels):
                        if RANK == 0:
                            print(
                                f"WARNING: Number of embeddings does not match number of labels! {len(embeddings)} != {len(labels)}"
                            )
                        if len(embeddings) > len(labels):
                            embeddings = embeddings[
                                : len(labels)
                            ]  # Discarding last embedding, due to being a duplicate
                            if self.training is False:
                                starts = starts[: len(labels)]
                                ends = ends[: len(labels)]
                            print(
                                f"After fix length: {len(embeddings)} == {len(labels)}"
                            )
                        else:
                            raise ValueError(
                                f"Number of embeddings is less than number of labels! {len(embeddings)} < {len(labels)}, check your data!"
                            )

                    end_time = time.time()
                    embedding_time = end_time - start_time

                    if RANK == 0:
                        print("Embedding generation completed!")
                        print(
                            f"Total time: {embedding_time:.2f} seconds ({embedding_time / 60:.2f} minutes) | "
                            f"Time per sequence: {embedding_time / len(sequences):.4f} seconds | "
                            f"Sequences per second: {len(sequences) / embedding_time:.2f}"
                        )

                    if self.training is True and self.usage_mode is False:
                        # print("Creating stratified train/val split...")
                        if num_classes != 24381:
                            X_train, X_val, y_train, y_val = train_test_split(
                                embeddings,
                                labels,
                                test_size=VAL_FRAC,
                                stratify=labels,
                                random_state=42,
                            )
                        else:
                            X_train, X_val, y_train, y_val = train_test_split(
                                embeddings,
                                labels,
                                test_size=VAL_FRAC,
                                random_state=42,
                            )


                        train_embeddings = X_train
                        train_labels = y_train
                        val_embeddings = X_val
                        val_labels = y_val

                        # Save datasets to files
                        os.makedirs("/scratch/tmp/sapelt/Master_Thesis/temp", exist_ok=True)
                        if RANK == 0:
                            print("Writing embeddings to file...")
                        for rank_id in range(dist.get_world_size()):
                            if RANK == rank_id:
                                with h5py.File(
                                    f"/scratch/tmp/sapelt/Master_Thesis/temp/embeddings_classification_{self.num_classes - 1}d.h5",
                                    "a",
                                ) as f:
                                    if del_key is True:
                                        if f"train_embeddings_chunk{chunk_num}_rank{RANK}" in f:
                                            del f[f"train_embeddings_chunk{chunk_num}_rank{RANK}"]
                                        if f"train_labels_chunk{chunk_num}_rank{RANK}" in f:
                                            del f[f"train_labels_chunk{chunk_num}_rank{RANK}"]
                                        if f"val_embeddings_chunk{chunk_num}_rank{RANK}" in f:
                                            del f[f"val_embeddings_chunk{chunk_num}_rank{RANK}"]
                                        if f"val_labels_chunk{chunk_num}_rank{RANK}" in f:
                                            del f[f"val_labels_chunk{chunk_num}_rank{RANK}"]
                                        del_key = False
                                        if RANK == 0:
                                            print(f"Deleted existing datasets for chunk {chunk_num}")

                                    f.create_dataset(
                                        f"train_embeddings_chunk{chunk_num}_rank{RANK}",
                                        data=train_embeddings.cpu().numpy(),
                                    )
                                    f.create_dataset(
                                        f"train_labels_chunk{chunk_num}_rank{RANK}",
                                        data=train_labels.cpu().numpy(),
                                    )
                                    # exit(0)
                                    f.create_dataset(
                                        f"val_embeddings_chunk{chunk_num}_rank{RANK}",
                                        data=val_embeddings.cpu().numpy(),
                                    )
                                    f.create_dataset(
                                        f"val_labels_chunk{chunk_num}_rank{RANK}",
                                        data=val_labels.cpu().numpy(),
                                    )
                            dist.barrier()  # Ensure only one rank writes at a time

                        if RANK == 0:
                            print(
                                f"Wrote embeddings and labels for batch {chunk_num} to file"
                            )

                    elif self.usage_mode is True:
                        if self.usage_mode is True:
                            save_path = (
                                "/scratch/tmp/sapelt/Master_Thesis/tempTest/embeddings/embeddings_domain_classifier.h5"
                            )
                            os.makedirs("/scratch/tmp/sapelt/Master_Thesis/tempTest/embeddings", exist_ok=True)

                        else:
                            save_path = f"/scratch/tmp/sapelt/Master_Thesis/temp/embeddings_classification_{self.num_classes - 1}d.h5"
                            os.makedirs("/scratch/tmp/sapelt/Master_Thesis/temp", exist_ok=True)

                        for rank_id in range(dist.get_world_size()):
                            if RANK == rank_id:
                                with h5py.File(
                                    save_path,
                                    "a",
                                ) as f:
                                    f.create_dataset(
                                        f"embeddings_{chunk_num}_rank{RANK}",
                                        data=embeddings.cpu().numpy(),
                                    )
                                    f.create_dataset(
                                        f"idx_multiplied_{chunk_num}_rank{RANK}",
                                        data=idx_multiplied,
                                    )

                            dist.barrier()  # Ensure only one rank writes at a time

                    else:
                        # Save embeddings and labels to files
                        os.makedirs("/scratch/tmp/sapelt/Master_Thesis/temp", exist_ok=True)

                        if RANK == 0:
                            print("SHAPES")
                            print(
                                f"Embeddings: {embeddings.shape}, Labels: {labels.shape}"
                            )
                            # print(
                            #     f"Starts: {starts.shape}, Ends: {ends.shape}"
                            # )

                        for rank_id in range(dist.get_world_size()):
                            if RANK == rank_id:
                                with h5py.File(
                                    f"/scratch/tmp/sapelt/Master_Thesis/temp/embeddings_classification_{self.num_classes - 1}d_EVAL.h5",
                                    "a",
                                ) as f:
                                    f.create_dataset(
                                        f"embeddings_chunk{chunk_num}_rank{RANK}",
                                        data=embeddings.cpu().numpy(),
                                    )
                                    f.create_dataset(
                                        f"labels_chunk{chunk_num}_rank{RANK}",
                                        data=labels.cpu().numpy(),
                                    )
                                    if self.training is False:
                                        f.create_dataset(
                                            f"starts_chunk{chunk_num}_rank{RANK}",
                                            data=starts.cpu().numpy(),
                                        )
                                        f.create_dataset(
                                            f"ends_chunk{chunk_num}_rank{RANK}",
                                            data=ends.cpu().numpy(),
                                        )
                            dist.barrier()  # Ensure only one rank writes at a time

                        if RANK == 0:
                            print(
                                f"Wrote embeddings and labels for batch {chunk_num} to file"
                            )


                    with open(f"/scratch/tmp/sapelt/Master_Thesis/temp/progress_{num_classes}.txt", "a") as status_file:
                        if RANK == 0:
                            status_file.write(f"Completed chunk {chunk_num}\n")

                    # Add cleanup after each chunk
                    if self.training is True:
                        del embeddings, labels, idx_multiplied
                    else:
                        del embeddings, labels, starts, ends, idx_multiplied
                    del sequences, chunk
                    if hasattr(self, "chunk"):
                        del self.chunk

                    # Force garbage collection after each chunk
                    import gc

                    gc.collect()

                    first_subsampler = False    

                    
                    if RANK == 0:
                        print(f"Successfully completed chunk {chunk_num}")
                    # exit(0)

                print("Done Embedding! Closing Embedder")

                if self.domain_boundary_detection is True:
                    if RANK == 0:
                        print("Loading data...")
                    try:
                        df = pd.read_csv(
                            csv_path, usecols=["start", "end", "id", "Pfam_id", "Sequence"]
                        )
                    except Exception as e:
                        if RANK == 0:
                            print(
                                f"Warning: Could not read with specific columns, falling back to all columns. Error: {e}"
                            )
                        df = pd.read_csv(
                            csv_path,
                        )
                        self.usage_mode = True
                    if RANK == 0:
                        print(f"Data loaded with {len(df)} sequences")

                    if category_col != "Pfam_id":
                        df["label"] = df[category_col].apply(map_label)

                    if self.training is False and self.usage_mode is False:
                        self.all_starts = df["start"].tolist()
                        self.all_ends = df["end"].tolist()

                    if category_col != "Pfam_id":
                        df.drop(columns=[category_col], inplace=True)
                    elif self.usage_mode is False:
                        df.drop(columns=["id", "Pfam_id"], inplace=True)

                    df = df[
                        df[sequence_col].str.len() >= 10
                    ]  # 10 for biological reasonings
                    self.df = df
                    if RANK == 0:
                        print("Data loaded")

                    # Use sequences in original order
                    sequences = self.df[sequence_col].tolist()
                    if RANK == 0:
                        print("Switching labels for domain boundary detection")
                    labels_list = []
                    for index, row in self.df.iterrows():
                        seq_len = len(row[sequence_col])
                        label = [0] * seq_len
                        if self.usage_mode is False:
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
                        (
                            self.embeddings,
                            self.labels,
                            self.starts,
                            self.ends,
                            self.idx_multiplied,
                        ) = self._embed(sequences)
                    else:
                        self.embeddings, self.labels, _, _, idx_multiplied = self._embed(
                            sequences
                        )

                    print("Done Embedding! Closing Embedder")
                    return  # Return early if domain boundary detection is enabled

        else:
            self.df = skip_df

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
                state_dict_device=torch.device("cpu"),
                cpu_offload=True,)

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

        def windower(stepsize=1000, dimension=1000):
            # cut sequences longer than 1000 characters into sliding windows, because of ESM2 limitations
            new_seqs = []
            new_labels = []
            count = 0
            idx_multiplied = []  # Track which original indices were windowed

            if self.training is True:
                for idx, (seq, label) in enumerate(
                    zip(seq_dataset.seqs, seq_dataset.labels)
                ):
                    if len(seq) > 1000:
                        count += 1
                        slices = [
                            seq[i : i + dimension]
                            for i in range(0, len(seq) - dimension + 1, stepsize)
                        ]
                        if len(seq) % dimension != 0:
                            slices.append(seq[-dimension:])
                        new_seqs.extend(slices)
                        new_labels.extend([label] * len(slices))
                        # Track that this original index was windowed
                        idx_multiplied.extend([idx] * len(slices))
                    else:
                        new_seqs.append(seq)
                        new_labels.append(label)
                        idx_multiplied.append(idx)
                self.end_window = [len(seq) for seq in new_seqs]
                print(
                    f"Rank {RANK}: Warning: {count} sequences were longer than 1000 characters and slided into windows"
                )

            elif self.training is False and self.domain_boundary_detection is False:
                new_starts = []
                new_ends = []
                for idx, (seq, label, start, end) in enumerate(
                    zip(
                        seq_dataset.seqs,
                        seq_dataset.labels,
                        seq_dataset.starts,
                        seq_dataset.ends,
                    )
                ):
                    if len(seq) > 1000:
                        count += 1
                        slices = [
                            seq[i : i + dimension]
                            for i in range(0, len(seq) - dimension + 1, stepsize)
                        ]
                        if len(seq) % dimension != 0:
                            slices.append(seq[-dimension:])
                        new_seqs.extend(slices)
                        new_labels.extend([label] * len(slices))
                        new_starts.extend([start] * len(slices))
                        new_ends.extend([end] * len(slices))
                        # Track that this original index was windowed
                        idx_multiplied.extend([idx] * len(slices))
                    else:
                        new_seqs.append(seq)
                        new_labels.append(label)
                        new_starts.append(start)
                        new_ends.append(end)
                        idx_multiplied.append(idx)
                self.end_window = [len(seq) for seq in new_seqs]
                print(
                    f"Rank {RANK}: Warning: {count} sequences were longer than 1000 characters and slided into windows"
                )

            elif self.domain_boundary_detection is True:
                new_starts = []
                new_ends = []

                # For domain boundary detection with training=False but usage_mode=True,
                # we need dummy starts and ends since they weren't loaded from CSV
                if self.training is False and self.usage_mode is True:
                    # Create dummy start and end values for each sequence
                    starts = [0] * len(seq_dataset.seqs)
                    ends = [len(seq) for seq in seq_dataset.seqs]
                    new_starts = []
                    new_ends = []
                else:
                    starts = seq_dataset.starts
                    ends = seq_dataset.ends

                for idx, (seq, label, start, end) in enumerate(
                    zip(
                        seq_dataset.seqs,
                        seq_dataset.labels,
                        starts,
                        ends,
                    )
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

                        for slice_start, slice_end in slice_positions:
                            # Always window the label to match the sequence window (dimension=1000)
                            windowed_label = label[
                                slice_start:slice_end
                            ]  # Slice the label

                            # Ensure windowed_label is exactly dimension length
                            if windowed_label.shape[0] < dimension:
                                windowed_label = torch.nn.functional.pad(
                                    windowed_label,
                                    (0, dimension - windowed_label.shape[0]),
                                    value=0,
                                )
                            elif windowed_label.shape[0] > dimension:
                                windowed_label = windowed_label[
                                    :dimension
                                ]  # Truncate to dimension

                            new_labels.append(windowed_label)

                        new_starts.extend([start] * len(slices))
                        new_ends.extend([end] * len(slices))
                        # Track that this original index was windowed
                        idx_multiplied.extend([idx] * len(slices))
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
                        idx_multiplied.append(idx)

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
                return new_seqs, new_labels, idx_multiplied
            else:
                return new_seqs, new_labels, new_starts, new_ends, idx_multiplied

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
        # print(len(seqs))

        start_time = time.time()

        if self.usage_mode is True:
            # dummy labels for usage mode - make them all the same size (1000) for consistency
            self.labels = [torch.zeros(1000, dtype=torch.long) for seq in seqs]

        # Use the new dataset that includes labels
        if self.training is True and self.skip_df is None:
            seq_dataset = SeqDataset(seqs, self.labels)
        elif self.domain_boundary_detection is True:
            seq_dataset = SeqDatasetForEval(
                seqs, self.labels, self.all_starts, self.all_ends
            )
        elif (
            self.training is False
            and hasattr(self, "all_starts")
            and hasattr(self, "all_ends")
        ):
            seq_dataset = SeqDatasetForEval(
                seqs, self.labels, self.all_starts, self.all_ends
            )
        else:
            # Fallback to basic dataset
            seq_dataset = SeqDataset(seqs, self.labels)

        # if RANK == 0:
        print(len(seq_dataset))

        if self.training is True:
            new_seqs, new_labels, idx_multiplied = windower(
                stepsize=500, dimension=1000
            )
        else:
            new_seqs, new_labels, new_starts, new_ends, idx_multiplied = windower(
                stepsize=500, dimension=1000
            )


        seq_dataset.seqs = new_seqs
        seq_dataset.labels = new_labels

        # print("AFTER",len(seq_dataset))


        if self.training is False:
            seq_dataset.starts = new_starts
            seq_dataset.ends = new_ends

        if self.domain_boundary_detection is True:
            # Convert seq_dataset.labels to a tensor if it's a list
            if isinstance(seq_dataset.labels, list):
                if isinstance(seq_dataset.labels[0], torch.Tensor):
                    seq_dataset.labels = torch.stack(seq_dataset.labels)
                else:
                    seq_dataset.labels = torch.tensor(
                        seq_dataset.labels, dtype=torch.long
                    )

            print(seq_dataset.labels.shape)

        # if RANK == 0:
        #     print( f"Seq: {seq_dataset.seqs[0:5]}, Label: {seq_dataset.labels[0:5]}")

        # Create a DataLoader with DistributedSampler for distributed training
        sampler = DistributedSampler(
            seq_dataset,
            num_replicas=dist.get_world_size(),
            rank=RANK,
            shuffle=False,
            drop_last=True,
        )

        # print(len(seq_dataset))

        if self.usage_mode is True:
            # For usage mode or domain boundary detection, we need to ensure all sequences are processed
            # without padding, so we create a custom sampler that does not pad sequences
            class CustomDistributedSampler(DistributedSampler):
                def __init__(
                    self, dataset, num_replicas=None, rank=None, shuffle=False
                ):
                    super().__init__(
                        dataset, num_replicas, rank, shuffle, drop_last=False
                    )
                    # Recalculate num_samples to avoid padding
                    self.total_size = len(self.dataset)
                    samples_per_rank = self.total_size // self.num_replicas
                    remainder = self.total_size % self.num_replicas
                    
                    if self.rank < remainder:
                        self.num_samples = samples_per_rank + 1
                    else:
                        self.num_samples = samples_per_rank

            sampler = CustomDistributedSampler(
                seq_dataset,
                num_replicas=dist.get_world_size(),
                rank=RANK,
                shuffle=False,
            )
            print("SAMPLER CUSTOM USED")

        dataloader = DataLoader(
            seq_dataset,
            batch_size=self.emb_batch,
            num_workers=8,
            sampler=sampler,
            pin_memory=True,
        )

        if RANK == 0:
            # print(f"Expected embedding dimension: {expected_dim}")
            print(
                f"Total sequences to process: {len(seq_dataset.seqs)} with batch size {self.emb_batch}"
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

                # if batch_num == 0 and RANK == 0:
                #     print(batch_seqs, batch_labels, batch_start, batch_end)

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
                embeddings = embeddings[
                    :, 1:-1, :
                ]  # remove <eos> and <cls> tokens, made by esm model (start and end tokens)

                # if RANK == 0 and batch_num < 2:  # Print first 2 batches for debugging
                #     print(
                #         f"Batch {batch_num + 1}: {embeddings.shape} embeddings, {mask.shape} mask"
                #     )
                #     print(embeddings[0:2, 0:5, 0:5])

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

                        start_idx = (chunk - 1) * 1000
                        end_idx = min((chunk * 1000) + 1, len(all_embeddings))

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

                from tqdm import tqdm

                # Then replace your print statement with this tqdm implementation:
                if batch_num == 0 and RANK == 0:
                    # Initialize progress bar only once at the beginning
                    progress_bar = tqdm(
                        total=len(dataloader),
                        desc="Embedding Data",
                        position=0,
                        leave=True,
                        ncols=150  # Set a wider display for more info
                    )

                # Inside your loop, replace the current print statement with:
                if (batch_num % 1 == 0 or batch_num == len(dataloader) - 1 or batch_num == 0) and RANK == 0:
                    # Get memory stats
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    memory_gb = memory_info.rss / 1024 / 1024 / 1024
                    
                    system_memory = psutil.virtual_memory()
                    total_gb = system_memory.total / 1024 / 1024 / 1024
                    available_gb = system_memory.available / 1024 / 1024 / 1024
                    used_percent = system_memory.percent
                                        
                    # Update progress bar with all information
                    progress_bar.update(1 if batch_num == 0 else min(100, batch_num - progress_bar.n))
                    progress_bar.set_postfix({
                        'RAM': f"{memory_gb:.1f}GB ({used_percent:.0f}%, {available_gb:.1f}GB free)",
                        'Total': f"{total_gb:.1f}GB"
                    })

            except Exception as e:
                # Handle exceptions during batch processing, if vram blows up, we will try to process each sequence individually
                # kind of unnecessary, cuz emb_batch is already 1 is the fastes (no padding the batch to same length needed, due to 1 sequence only), but was already implemented
                # can be ignored because it never triggers during ram oom
                print(f"\nRank {RANK}: Error processing batch {batch_num + 1}: {e}\n")
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
                            if (
                                self.training is False
                                or self.domain_boundary_detection is True
                            ):
                                all_starts.append(torch.tensor([0], dtype=torch.long))
                                all_ends.append(
                                    torch.tensor([len(seq)], dtype=torch.long)
                                )

                    except Exception as single_e:
                        # last fallback, empty embedding, if something goes wrong
                        # never triggers, at least never seen it trigger
                        print(
                            f"\nRank {RANK}: Error processing individual sequence: {single_e}\n"
                        )
                        zero_embedding = torch.zeros(1, expected_dim)
                        all_embeddings.append(zero_embedding)
                        all_labels.append(label.unsqueeze(0))  # Still keep the label
                        if (
                            self.training is False
                            or self.domain_boundary_detection is True
                        ):
                            all_starts.append(torch.tensor([0], dtype=torch.long))
                            all_ends.append(torch.tensor([len(seq)], dtype=torch.long))

            if self.domain_boundary_detection is True:
                # Save embeddings and labels in chunks to avoid memory overflow
                # on palma do only 100

                # if batch_num % 1000:
                #     for emb in all_embeddings:
                #         emb = emb.cpu()
                #     for lbl in all_labels:
                #         lbl = lbl.cpu()

                if (
                    batch_num % 3000 == 0
                    and batch_num > 0
                    or batch_num == len(dataloader) - 1
                ):
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
                    embeddings_tensor = torch.stack(
                        [emb.cpu() for emb in all_embeddings]
                    )
                    labels_tensor = torch.stack([lbl.cpu() for lbl in all_labels])

                    dist.barrier()  # Ensure all ranks are synchronized before saving

                    if self.usage_mode is True:
                        save_path = "/scratch/tmp/sapelt/Master_Thesis/tempTest/embeddings/embeddings_domain.h5"
                    else:
                        save_path = "/scratch/tmp/sapelt/Master_Thesis/temp/embeddings_domain.h5"

                    if RANK == 0:
                        print("\n\nWriting...")
                    for rank_id in range(dist.get_world_size()):
                        if RANK == rank_id:
                            with h5py.File(save_path, "a") as f:
                                f[f"batch_{batch_num}_rank_{RANK}"] = (
                                    embeddings_tensor.numpy()
                                )
                                f[f"labels_batch_{batch_num}_rank_{RANK}"] = (
                                    labels_tensor.numpy()
                                )
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
                    dist.barrier()  # Synchronize all ranks before continuing

            else:
                # basic way for ID classification, just store the embeddings and labels

                # Concatenate local embeddings and labels
                local_embeddings = torch.cat(all_embeddings, dim=0)
                local_labels = torch.cat(all_labels, dim=0)

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

        if RANK == 0:
            print("\nEmbeddings DONE!\n")

        # ------------------------------------------------------
        #   BROADCASTING AND GATHERING EMBEDDINGS AND LABELS
        # ------------------------------------------------------
        # used in classification tasks, where we need to gather all embeddings and labels from all ranks,
        # tried my way here, it works, but some parts i dont understand fully
        # NOT NEEDED FOR DOMAIN BOUDNARY TASK

        if dist.get_world_size() > 1 and self.domain_boundary_detection is False:
            all_embeddings.clear()
            all_labels.clear()
            print(len(local_embeddings), "local embeddings size")
            final_embeddings = local_embeddings
            final_labels = local_labels
            if self.training is False:
                final_starts = local_starts
                final_ends = local_ends
        #     # Gather embeddings and labels from all processes
        #     local_size = torch.tensor(
        #         [local_embeddings.size(0)], dtype=torch.int64, device="cuda"
        #     )
        #     all_sizes = [
        #         torch.zeros_like(local_size) for _ in range(dist.get_world_size())
        #     ]
        #     dist.all_gather(all_sizes, local_size)

        #     total_size = sum(size.item() for size in all_sizes)

        #     if RANK == 0:
        #         # Initialize final_embeddings with the correct dimensions
        #         final_embeddings = torch.empty(
        #             total_size, expected_dim, dtype=torch.float32
        #         )

        #         # Initialize final_labels with the correct dimensions
        #         final_labels = torch.empty(total_size, dtype=torch.long)

        #         if self.training is False:
        #             final_starts = torch.empty(total_size, dtype=torch.long)
        #             final_ends = torch.empty(total_size, dtype=torch.long)

        #         current_idx = 0
        #         for rank in range(dist.get_world_size()):
        #             rank_size = all_sizes[rank].item()
        #             if rank_size > 0:
        #                 if rank == 0:
        #                     final_embeddings[current_idx : current_idx + rank_size] = (
        #                         local_embeddings
        #                     )
        #                     final_labels[current_idx : current_idx + rank_size] = (
        #                         local_labels
        #                     )
        #                     if self.training is False:
        #                         final_starts[current_idx : current_idx + rank_size] = (
        #                             local_starts
        #                         )
        #                         final_ends[current_idx : current_idx + rank_size] = (
        #                             local_ends
        #                         )
        #                 else:
        #                     rank_embeddings = torch.empty(
        #                         rank_size,
        #                         expected_dim,
        #                         dtype=torch.float32,
        #                         device="cuda",
        #                     )
        #                     dist.recv(rank_embeddings, src=rank)
        #                     final_embeddings[current_idx : current_idx + rank_size] = (
        #                         rank_embeddings.cpu()
        #                     )

        #                     rank_labels = torch.empty(
        #                         rank_size, dtype=torch.long, device="cuda"
        #                     )
        #                     dist.recv(rank_labels, src=rank)
        #                     final_labels[current_idx : current_idx + rank_size] = (
        #                         rank_labels.cpu()
        #                     )

        #                     if self.training is False:
        #                         rank_starts = torch.empty(
        #                             rank_size, dtype=torch.long, device="cuda"
        #                         )
        #                         dist.recv(rank_starts, src=rank)
        #                         final_starts[current_idx : current_idx + rank_size] = (
        #                             rank_starts.cpu()
        #                         )

        #                         rank_ends = torch.empty(
        #                             rank_size, dtype=torch.long, device="cuda"
        #                         )
        #                         dist.recv(rank_ends, src=rank)
        #                         final_ends[current_idx : current_idx + rank_size] = (
        #                             rank_ends.cpu()
        #                         )

        #                     del rank_embeddings, rank_labels
        #                     if self.training is False:
        #                         del rank_starts, rank_ends
        #                     torch.cuda.empty_cache()
        #                 current_idx += rank_size
        #     else:
        #         if local_embeddings.size(0) > 0:
        #             local_embeddings_gpu = local_embeddings.cuda()
        #             dist.send(local_embeddings_gpu, dst=0)
        #             del local_embeddings_gpu

        #             local_labels_gpu = local_labels.cuda()
        #             dist.send(local_labels_gpu, dst=0)
        #             del local_labels_gpu

        #             if self.training is False:
        #                 local_starts_gpu = local_starts.cuda()
        #                 dist.send(local_starts_gpu, dst=0)
        #                 del local_starts_gpu

        #                 local_ends_gpu = local_ends.cuda()
        #                 dist.send(local_ends_gpu, dst=0)
        #                 del local_ends_gpu

        #             torch.cuda.empty_cache()

        #     # Broadcast final data from rank 0 to all ranks
        #     if RANK == 0:
        #         size_tensor = torch.tensor([final_embeddings.size(0)], device="cuda")
        #     else:
        #         size_tensor = torch.tensor([0], device="cuda")

        #     dist.broadcast(size_tensor, 0)

        #     total_size = size_tensor.item()
        #     chunk_size = 10000
        #     if self.domain_boundary_detection is True:
        #         chunk_size = 1000  # Smaller chunk size for domain boundary detection

        #     if RANK != 0:
        #         final_labels = torch.empty(total_size, dtype=torch.long, device="cuda")
        #         final_embeddings = torch.empty(
        #             total_size, expected_dim, dtype=torch.float32
        #         )

        #         if self.training is False:
        #             final_starts = torch.empty(total_size, dtype=torch.long)
        #             final_ends = torch.empty(total_size, dtype=torch.long)

        #     for start_idx in range(0, total_size, chunk_size):
        #         end_idx = min(start_idx + chunk_size, total_size)
        #         chunk_len = end_idx - start_idx

        #         # Broadcast embeddings chunk
        #         if RANK == 0:
        #             chunk_gpu = final_embeddings[start_idx:end_idx].cuda()
        #         else:
        #             chunk_gpu = torch.empty(
        #                 chunk_len, expected_dim, dtype=torch.float32, device="cuda"
        #             )
        #         dist.broadcast(chunk_gpu, 0)
        #         if RANK != 0:
        #             final_embeddings[start_idx:end_idx] = chunk_gpu.cpu()
        #         del chunk_gpu

        #         # Broadcast labels chunk
        #         if RANK == 0:
        #             labels_chunk_gpu = final_labels[start_idx:end_idx].cuda()
        #         else:
        #             labels_chunk_gpu = torch.empty(
        #                 chunk_len, dtype=torch.long, device="cuda"
        #             )
        #         dist.broadcast(labels_chunk_gpu, 0)
        #         if RANK != 0:
        #             final_labels[start_idx:end_idx] = labels_chunk_gpu.cpu()
        #         del labels_chunk_gpu

        #         if self.training is False:
        #             # Broadcast starts chunk
        #             if RANK == 0:
        #                 starts_chunk_gpu = final_starts[start_idx:end_idx].cuda()
        #             else:
        #                 starts_chunk_gpu = torch.empty(
        #                     chunk_len, dtype=torch.long, device="cuda"
        #                 )
        #             dist.broadcast(starts_chunk_gpu, 0)
        #             if RANK != 0:
        #                 final_starts[start_idx:end_idx] = starts_chunk_gpu.cpu()
        #             del starts_chunk_gpu

        #             # Broadcast ends chunk
        #             if RANK == 0:
        #                 ends_chunk_gpu = final_ends[start_idx:end_idx].cuda()
        #             else:
        #                 ends_chunk_gpu = torch.empty(
        #                     chunk_len, dtype=torch.long, device="cuda"
        #                 )
        #             dist.broadcast(ends_chunk_gpu, 0)
        #             if RANK != 0:
        #                 final_ends[start_idx:end_idx] = ends_chunk_gpu.cpu()
        #             del ends_chunk_gpu

        #         torch.cuda.empty_cache()

        elif self.domain_boundary_detection is True and self.usage_mode is True:
            # print(len(local_embeddings), "local embeddings size")
            # final_embeddings = local_embeddings
            # final_labels = local_labels
            # # Counterpart for domain boundary detection, where we gather all embeddings and labels from all ranks via their .h5 files
            # # Open embeddings and labels from the single h5 file
            # all_embeddings = []
            # all_labels = []

            # if RANK == 0:
            #     print("Gathering embeddings and labels from single h5 file...")

            # try:
            #     with h5py.File(f"/scratch/tmp/sapelt/Master_Thesis/temp/embeddings_doamain.h5", "r") as f:
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
            #     print(f"Warning: File /scratch/tmp/sapelt/Master_Thesis/temp/embeddings_doamain.h5 not found")
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
            return (
                final_embeddings,
                final_labels,
                final_starts,
                final_ends,
                idx_multiplied
                ) 
        elif self.domain_boundary_detection is True:
            return None, None, None, None, idx_multiplied
        else:
            return final_embeddings, final_labels, None, None, idx_multiplied


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly")
