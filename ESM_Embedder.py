import datetime
import gc
import os
import time

import esm
import h5py
import pandas as pd
import psutil
import torch
import torch.distributed as dist
import tqdm
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

os.environ["NCCL_TIMEOUT"] = (
    "36000000"  # 10 hours in milliseconds, to prevent NCCL timeout issues that sometimes arise during long training runs
)
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = (
    "1"  # Enable blocking wait for NCCL operations to improve stability, prevent deadlock
)
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = (
    "1"  # Enable async error handling for NCCL to catch errors more effectively, prevent deadlocks
)
world_size = int(
    os.environ.get("WORLD_SIZE", 1)
)  # World size = number of processes based on number of GPUs
use_ddp = world_size > 1  # Determine if DDP should be used based on WORLD_SIZE

# print(torch.__version__)


if not dist.is_initialized():  # Set up environment variables if not already set
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"  # Default to rank 0 if not set
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"  # Default to world size 1 if not set
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"  # Default to localhost if not set
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"  # Default port if not set

    # Initialize ddp process group
    dist.init_process_group(
        backend="nccl"
        if torch.cuda.is_available()
        else "gloo",  # priorize gpu if available, else cpu backend
        rank=int(os.environ["RANK"]),  # get rank from environment variable
        world_size=int(
            os.environ["WORLD_SIZE"]
        ),  # get world size from environment variable
        timeout=datetime.timedelta(
            seconds=36000
        ),  # set timeout to 10 hours to prevent timeout during long runs, huge datasets
    )

    if dist.get_rank() == 0:
        print("Initializing process group for DDP with device: CUDA")


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # set device to cuda if available, else cpu. Important as Import for other scripts

if use_ddp and torch.cuda.is_available():
    torch.cuda.set_device(
        int(os.environ["LOCAL_RANK"])
    )  # set cuda devices based on rank
    RANK = (
        dist.get_rank()
    )  # Global rank to allow single usage of one process for logging
    if RANK == 0:
        print(
            f"Start running basic DDP example with worldsize {dist.get_world_size()}."
        )
    DEVICE_ID = (
        RANK % torch.cuda.device_count()
    )  # set device id based on rank and available gpus
else:
    # Single-GPU fallback for quick testing
    DEVICE_ID = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # single gpu use or cpu fallback
    RANK = 0
    print("Running in single-GPU mode without DDP.")


TRAIN_FRAC = 0.8  # fraction of data used for training
VAL_FRAC = 0.2  # fraction of data used for validation


VRAM = psutil.virtual_memory().total // (1024**3)  # get available vram in gb
EMB_BATCH = (
    32 if VRAM >= 24 else 16 if VRAM >= 16 else 8 if VRAM >= 8 else 4
)  # set embedding batch size based on available vram


# -------------------------
# 4. Dataset & embedding
# -------------------------
class SeqDataset(Dataset):
    """
    basic Dataset class to load seqs and corresponding labels
    """

    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


class SeqDatasetForEval(Dataset):
    """
    Dataset class to load seqs and corresponding labels for evaluation.
    For Eval we also need start and end positions.
    """

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
    """
    MAIN EMBEDDER CLASS with 3 main functions:
    - __init__: initializes the embedder, loads data from csv, preprocesses it, and starts embeddings
                Structurally differencitated by DomainBoundaryDetection and normal classification mode

                map_label: maps pfam ids to labels based on num_classes
                minicutter: cuts sequences based on start and end positions to single out only domains
                subsampler: subsamples class 0 to a target number of samples per class to reduce class imbalance (used for smaller num_classes settings)


    - esm_loader:   loads the esm model and batch converter, all other settings like ddp or fsdp are also set here

    - _embed:   embeds a list of sequences and returns embeddings and labels
                This function handles the actual embedding process, including batching and padding of sequences.
                Embeddings are done in batches to prevent oom errors.
                windower: if sequence is longer than model max length, it is split into overlapping windows to be embedded separately, adding corresponding start and end positions
                Then the embeddings are pooled (classification mode) or kept per residue (domain boundary detection mode)
                Finally embeddings and labels are saved to an h5 file for later use. Each loops appends to the same file, while the file is open for writing.
                Each rank writes its own part singlely to prevent conflicts or locked files (for ddp)

    Flags:
    - num_classes: number of classes for classification task (24381 for all pfam domains, else smaller subsets)
    - esm_model: the esm model to use for embedding (standard: esm2_t33_650M_UR50D)
    - csv_path: path to the csv file containing the data
    - category_col: name of the column in the csv file that contains the category labels (pfam ids)
    - sequence_col: name of the column in the csv file that contains the sequences (embedding data)
    - emb_batch: batch size for embedding, automatically set based on available vram
    - skip_df: if provided, skips data loading from csv and uses this dataframe directly. Used for re-embedding technique to determine domain boundaries (old)
    - FSDP_used: if True, uses Fully Sharded Data Parallel for model wrapping, else uses DDP or if single gpu no wrapping
    - domain_boundary_detection: if True, the embedder is used for domain boundary detection, else for classification
    - training: if True, the embedder is used for training data preparation, else for evaluation data preparation

    - usage_mode:   if detected True during data loading, it means the csv does not have standard columns and is used for domain boundary detection with different column names
                    Used as an intended exception to start usage mode in embedding (no train,val split)

    A checkpoint system is implemented with a temp file "progress_{num_classes}.txt" to allow resuming from last processed chunk in case of interruptions.
    Systems are given to prevent duplicate entries when resuming or skipping one chunk if the chunk got not completed.

    Final return is when all embeddings are generated and saved to file, when no more chunks are left to process.
    Embeddings are then loaded from the h5 file after the ESMEmbedder is initialized.
    """

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
        self.model, self.batch_converter = (
            self.esm_loader()
        )  # load esm model and batch converter
        self.all_starts = None
        self.all_ends = None
        self.emb_batch = emb_batch

        def map_label(cat):
            """Adaptive mapping based on NUM_CLASSES
            0 = Negativ/Unkown class
            1 - N = Pfam classes
            """
            if cat <= num_classes - 2:
                return cat + 1
            else:
                return 0

        def minicutter(row):
            """
            reads in start, end values to cut the input sequences to only use domain sequences for embedding. Used for train mode
            """
            sequence = row[sequence_col]
            start = int(row["start"])
            end = int(row["end"])
            return sequence[start:end]

        def subsampler(df, target, first_subsampler):
            """
            Subsamples class 0 to target samples to reduce class imbalance. Mainly used for smaller num_classes settings.
            """
            if target is None:  # early exit if no subsampling set
                return df

            # Count samples per class
            class_counts = df["label"].value_counts()

            # Subsample class 0  to target samples each
            subsampled_dfs = []

            for class_label in sorted(class_counts.index):
                class_samples = df[
                    df["label"] == class_label
                ]  # loop over classes to get samples
                current_count = len(
                    class_samples
                )  # get current count of samples for each class

                if (
                    class_label == 0 and current_count > target
                ):  # subsample only class 0 if above target
                    class_subsampled = class_samples.sample(
                        n=target,
                        random_state=42,  # reproducible subsampling
                    )
                    # if RANK == 0:
                    #     print(f"  Class {class_label}: {current_count} -> {target}")
                else:
                    class_subsampled = class_samples  # keep all other classes as is
                    # if RANK == 0:
                    #     print(f"  Class {class_label}: {current_count} (kept all)")

                subsampled_dfs.append(class_subsampled)  # store subsampled class dfs

            # Combine all subsampled classes
            df = pd.concat(subsampled_dfs, ignore_index=True)

            if RANK == 0:
                # print final class distribution
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
                    os.makedirs(
                        "/global/research/students/sapelt/Masters/MasterThesis/temp",
                        exist_ok=True,
                    )
                    all_pfam = all_pfam[1:]  # Exclude "other" category
                    open(
                        f"/global/research/students/sapelt/Masters/MasterThesis/temp/selected_pfam_ids_{num_classes - 1}.txt",
                        "w",
                    ).write("\n".join(all_pfam))

            del class_samples, class_subsampled

            return df

        if skip_df is None:
            # domain boundary detection mode
            if domain_boundary_detection is False:
                if RANK == 0:
                    print("Loading data...")

                # !!! Chunk size, adjust based on available memory !!!
                chunksize = 20000000

                # path to selected IDs file, based on num_classes set
                selected_ids_file = f"/global/research/students/sapelt/Masters/MasterThesis/temp/selected_pfam_ids_{num_classes - 1}.txt"

                if num_classes < 24381:
                    if self.training is True and not os.path.exists(selected_ids_file):
                        # If training mode and selected IDs file does not exist, create it
                        if RANK == 0:
                            print(
                                "Performing initial pass to determine global class distribution..."
                            )

                        # calculate global class distribution to pick top classes if num_classes < 24381 (all pfam ids)
                        global_pfam_counts = pd.Series(dtype=int)
                        try:
                            count_iter = pd.read_csv(
                                csv_path, usecols=[category_col], chunksize=chunksize
                            )
                            for chunk in count_iter:
                                global_pfam_counts = global_pfam_counts.add(
                                    chunk[category_col].value_counts(), fill_value=0
                                )
                        except Exception as e:
                            # Error handling
                            print(f"Error during pre-computation pass: {e}")
                            exit(1)

                        print("Global counts calculated. Selecting top classes...")
                        # Sort by class counts
                        global_pfam_counts.sort_values(ascending=False, inplace=True)

                        # Select Pfam IDs with more than 10 samples, to ensure enough data for training, preventive step should always be true for larger datasets
                        frequent_pfam_ids = global_pfam_counts[
                            global_pfam_counts > 10
                        ].index.tolist()

                        # 10d List
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

                        # 10d THIO list
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

                        # 100d list
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

                        # check which priority ids are in frequent ids and gather them
                        available_priority_ids = [
                            pid for pid in priority_pfam_ids if pid in frequent_pfam_ids
                        ]

                        # select remaining ids
                        remaining_ids = [
                            pid
                            for pid in frequent_pfam_ids
                            if pid not in priority_pfam_ids
                        ]
                        # number of additional ids that can be selected
                        max_additional_ids = (
                            num_classes - 1 - len(available_priority_ids)
                        )

                        # FINAL selected ids, remaining ids + remaining spots filled with most frequent remaining ids
                        selected_ids = (
                            available_priority_ids + remaining_ids[:max_additional_ids]
                        )

                        # Save the globally determined list to a file.
                        with open(selected_ids_file, "w") as f:
                            for pfam_id in selected_ids:
                                f.write(f"{pfam_id}\n")
                        print(
                            f"Saved {len(selected_ids)} globally selected Pfam IDs to file."
                        )

                    elif (
                        num_classes != 24381 and os.path.exists(selected_ids_file)
                    ):  # For smaller num_classes, load selected Pfam IDs from file if it exists
                        with open(selected_ids_file, "r") as f:
                            selected_ids = [line.strip() for line in f.readlines()]
                else:  # For 24381 classes, use all Pfam IDs stored in a made file
                    with open(
                        "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/all_pfamIDs.txt",
                        "r",
                    ) as f:
                        selected_ids = [line.strip() for line in f.readlines()]

                # Create mapping from Pfam ID to label index
                pfam_to_label = {
                    pfam_id: i + 1 for i, pfam_id in enumerate(selected_ids)
                }

                # Relic from previous boudnary detection via re-embedding
                first_subsampler = True

                # start chunk set and creating del_flag for interrupted chunks
                start_chunk = 0
                del_key = False

                # check progress file to resume from last processed chunk if available
                if os.path.exists(
                    f"/global/research/students/sapelt/Masters/MasterThesis/temp/progress_{num_classes}.txt"
                ):
                    with open(
                        f"/global/research/students/sapelt/Masters/MasterThesis/temp/progress_{num_classes}.txt",
                        "r",
                    ) as status_file:
                        lines = status_file.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if (
                                "Completed chunk" in last_line
                            ):  # Look for completed chunks
                                start_chunk = (
                                    int(last_line.split(" ")[2]) + 1
                                )  # Start from NEXT chunk
                            else:
                                start_chunk = int(
                                    last_line.split(" ")[2]
                                )  # Start from last incomplete chunk
                                del_key = True  # del_key to delete last incomplete chunk from h5 file if found in h5 file

                            if RANK == 0:
                                print(f"Resuming from chunk {start_chunk}")

                # Create chunk iterators for different use cases
                try:
                    # base iter for train mode
                    chunk_iter = pd.read_csv(
                        csv_path,
                        usecols=["start", "end", "id", "Pfam_id", "Sequence"],
                        chunksize=chunksize,
                    )
                except Exception as e:
                    # exception handling to enter usage mode based on the missing columns earlier described
                    if RANK == 0:
                        print(
                            f"Warning: Could not read with specific columns, falling back to all columns. Error: {e}"
                        )
                    chunk_iter = pd.read_csv(csv_path, chunksize=chunksize)
                    self.usage_mode = True

                # Skip processed chunks
                for _ in range(start_chunk):
                    try:
                        next(chunk_iter)
                    except StopIteration:
                        if RANK == 0:
                            print("All chunks processed. No more chunks to process.")
                            with open(
                                f"/global/research/students/sapelt/Masters/MasterThesis/temp/progress_{num_classes}.txt",
                                "a",
                            ) as status_file:
                                if RANK == 0:
                                    status_file.write(
                                        "All chunks processed. Exiting.\n"
                                    )
                        return  # exit if all chunks are already processed

                expected_chunks = int(
                    22 * (10000000 / chunksize)
                )  # Approximate number of chunks for progress reporting, based on the 22 chunks for chunksize 10000000 based of earlier calculation on the full uniprot set

                # Loop to process remaining chunks with the chosen chunk_iter from the set start_chunk
                for chunk_num, chunk in enumerate(chunk_iter, start=start_chunk):
                    if RANK == 0:
                        print(
                            f"Processing chunk {chunk_num}/{int(expected_chunks)} with {len(chunk)} sequences"
                        )

                    # Write processing status (before processing): "Processing chunk {chunk_num}"
                    with open(
                        f"/global/research/students/sapelt/Masters/MasterThesis/temp/progress_{num_classes}.txt",
                        "a",
                    ) as status_file:
                        if RANK == 0:
                            status_file.write(f"Processing chunk {chunk_num}\n")

                    # if domain start and end columns are present, cut sequences accordingly to only embed domains
                    if "start" in chunk.columns and "end" in chunk.columns:
                        if training is True:
                            if RANK == 0 and chunk_num == 0:
                                print(
                                    "Cutting sequences based on start and end positions"
                                )

                            # calling minicutter to cut sequences to domain sequences only
                            chunk[sequence_col] = chunk.apply(minicutter, axis=1)

                    # Map labels to integers, final label change if category column is not Pfam_id
                    if category_col != "Pfam_id":
                        chunk["label"] = chunk[category_col].apply(map_label)

                    elif (
                        domain_boundary_detection is False and self.usage_mode is False
                    ):
                        # Map Pfam IDs to labels using pfam_to_label mapping
                        chunk["label"] = (
                            chunk[category_col].map(pfam_to_label).fillna(0)
                        )

                        # set labels to int
                        chunk["label"] = chunk["label"].astype(int)

                        # Subsample class 0 to target samples to reduce class imbalance, when not in usage mode
                        chunk = subsampler(chunk, 100000, first_subsampler)

                    # store start and end positions for EVAL mode
                    if self.training is False:
                        self.all_starts = chunk["start"].tolist()
                        self.all_ends = chunk["end"].tolist()

                    # Drop unnecessary columns to save memory and prevent issues
                    if category_col != "Pfam_id":
                        chunk.drop(columns=[category_col], inplace=True)
                    elif self.usage_mode is False:
                        chunk.drop(
                            columns=["start", "end", "id", "Pfam_id"], inplace=True
                        )

                    # print(chunk.columns)

                    # Filter out sequences shorter than 10 residues, for biological reasonings
                    chunk = chunk[chunk[sequence_col].str.len() >= 10]

                    # add reference to current chunk
                    self.chunk = chunk
                    if RANK == 0:
                        print("Data loaded")

                    # Use sequences in original order for embedding
                    sequences = self.chunk[sequence_col].tolist()

                    # gather labels if not in usage mode
                    if self.usage_mode is False:
                        self.labels = torch.tensor(
                            self.chunk["label"].values, dtype=torch.long
                        )

                    # start time stamp
                    start_time = time.time()

                    # Different embedding calls for training and evaluation mode due to different return values.
                    # Evaluation mode needs start and end positions for later boundary detection
                    # Training mode and usage mode dont need / dont have start and end positions
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

                    # Check for mismatched lengths between embeddings and labels based on the windowing technique for uneven n of sequences
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

                    # embedding time calculation
                    end_time = time.time()
                    embedding_time = end_time - start_time

                    # prints
                    if RANK == 0:
                        print("Embedding generation completed!")
                        print(
                            f"Total time: {embedding_time:.2f} seconds ({embedding_time / 60:.2f} minutes) | "
                            f"Time per sequence: {embedding_time / len(sequences):.4f} seconds | "
                            f"Sequences per second: {len(sequences) / embedding_time:.2f}"
                        )

                    # train/val split and saving to h5 file for training mode
                    if self.training is True and self.usage_mode is False:
                        unique_labels, counts = torch.unique(labels, return_counts=True)
                        single_sample_classes = unique_labels[counts == 1]

                        # mask to remove single sample classes due to stratified splitting requirement
                        if len(single_sample_classes) > 0:
                            if RANK == 0:
                                print(
                                    f"Warning: Found {len(single_sample_classes)} classes with only one sample. Removing them before splitting."
                                )
                            mask = ~torch.isin(labels, single_sample_classes)

                            embeddings = embeddings[mask]
                            labels = labels[mask]

                        # Stratified train/val split, either stratified if num_classes != 24381 else normal split
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

                        # Save datasets to h5 file, each rank writes its own part singlely to prevent conflicts due to locking flagged with chunk_num and rank
                        os.makedirs(
                            "/global/research/students/sapelt/Masters/MasterThesis/temp",
                            exist_ok=True,
                        )
                        if RANK == 0:
                            print("Writing embeddings to file...")
                        for rank_id in range(dist.get_world_size()):
                            if RANK == rank_id:
                                # filename based on num_classes
                                with h5py.File(
                                    f"/global/research/students/sapelt/Masters/MasterThesis/temp/embeddings_classification_{self.num_classes - 1}d.h5",
                                    "a",
                                ) as f:
                                    # if del_key got turned true due to interrupted chunk, delete existing datasets for this chunk and rank to prevent duplicates
                                    if del_key is True:
                                        if (
                                            f"train_embeddings_chunk{chunk_num}_rank{RANK}"
                                            in f
                                        ):
                                            del f[
                                                f"train_embeddings_chunk{chunk_num}_rank{RANK}"
                                            ]
                                        if (
                                            f"train_labels_chunk{chunk_num}_rank{RANK}"
                                            in f
                                        ):
                                            del f[
                                                f"train_labels_chunk{chunk_num}_rank{RANK}"
                                            ]
                                        if (
                                            f"val_embeddings_chunk{chunk_num}_rank{RANK}"
                                            in f
                                        ):
                                            del f[
                                                f"val_embeddings_chunk{chunk_num}_rank{RANK}"
                                            ]
                                        if (
                                            f"val_labels_chunk{chunk_num}_rank{RANK}"
                                            in f
                                        ):
                                            del f[
                                                f"val_labels_chunk{chunk_num}_rank{RANK}"
                                            ]
                                        # reset del_key for next chunk
                                        del_key = False
                                        if RANK == 0:
                                            print(
                                                f"Deleted existing datasets for chunk {chunk_num}"
                                            )

                                    # Actual dataset creation in h5 file per chunk per rank for val and train set (x and y separated)
                                    f.create_dataset(
                                        f"train_embeddings_chunk{chunk_num}_rank{RANK}",
                                        data=X_train.cpu().numpy(),
                                    )
                                    f.create_dataset(
                                        f"train_labels_chunk{chunk_num}_rank{RANK}",
                                        data=y_train.cpu().numpy(),
                                    )
                                    # exit(0)
                                    f.create_dataset(
                                        f"val_embeddings_chunk{chunk_num}_rank{RANK}",
                                        data=X_val.cpu().numpy(),
                                    )
                                    f.create_dataset(
                                        f"val_labels_chunk{chunk_num}_rank{RANK}",
                                        data=y_val.cpu().numpy(),
                                    )
                            # Ensure only one rank writes at a time
                            dist.barrier()

                    # usage mode save option - only embeddings and idx_multiplied saved for later usage
                    # idx_multiplied to track which original sequences were slided into windows
                    elif self.usage_mode is True:
                        # fix save_path for usage mode, no need to make it variable
                        save_path = "/global/research/students/sapelt/Masters/MasterThesis/tempTest/embeddings/embeddings_domain_classifier.h5"
                        # create tempTest/embeddings directory if not exists to load in embeddings
                        os.makedirs(
                            "/global/research/students/sapelt/Masters/MasterThesis/tempTest/embeddings",
                            exist_ok=True,
                        )

                        # save loop for each rank singlely to prevent conflicts
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

                            # Ensure only one rank writes at a time
                            dist.barrier()

                    # EVAL mode save option - embeddings, labels, starts and ends saved for later Evaluation with "Predicter_for_ESM.py"
                    else:
                        # create temp directory if not exists
                        os.makedirs(
                            "/global/research/students/sapelt/Masters/MasterThesis/temp",
                            exist_ok=True,
                        )

                        # debug prints for shapes
                        # if RANK == 0:
                        #     print("SHAPES")
                        #     print(
                        #         f"Embeddings: {embeddings.shape}, Labels: {labels.shape}"
                        #     )
                        # print(
                        #     f"Starts: {starts.shape}, Ends: {ends.shape}"
                        # )

                        # save loop for each rank singlely to prevent conflicts, with embeddings, labels, starts and ends of domains
                        for rank_id in range(dist.get_world_size()):
                            if RANK == rank_id:
                                # filename based on num_classes with _EVAL.h5 suffix
                                with h5py.File(
                                    f"/global/research/students/sapelt/Masters/MasterThesis/temp/embeddings_classification_{self.num_classes - 1}d_EVAL.h5",
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

                            # Ensure only one rank writes at a time
                            dist.barrier()

                    if RANK == 0:
                        print(
                            f"Wrote embeddings and labels for batch {chunk_num} to file"
                        )

                    # Final status write after completing chunk. Now its ensured that chunk got fully processed
                    with open(
                        f"/global/research/students/sapelt/Masters/MasterThesis/temp/progress_{num_classes}.txt",
                        "a",
                    ) as status_file:
                        if RANK == 0:
                            status_file.write(f"Completed chunk {chunk_num}\n")

                    # Add cleanup after each chunk based on use case to
                    if self.training is True:
                        del embeddings, labels, idx_multiplied
                    else:
                        del embeddings, labels, starts, ends, idx_multiplied
                    del sequences, chunk
                    if hasattr(self, "chunk"):
                        del self.chunk
                    gc.collect()

                    # disable first_subsampler after first chunk, Relic from previous re-embedding for boundary detection technique
                    first_subsampler = False

                    if RANK == 0:
                        print(f"Successfully completed chunk {chunk_num}")

            # Domain boundary detection embedding procedure
            if self.domain_boundary_detection is True:
                if RANK == 0:
                    print("Loading data...")
                # load data in for basic training mode
                try:
                    df = pd.read_csv(
                        csv_path, usecols=["start", "end", "id", "Pfam_id", "Sequence"]
                    )
                # use exception handling to enter usage mode based on the missing columns earlier described
                except Exception as e:
                    if RANK == 0:
                        print(
                            f"Warning: Could not read with specific columns, falling back to all columns. Error: {e}"
                        )
                    df = pd.read_csv(
                        csv_path,
                    )
                    # set usage mode to true
                    self.usage_mode = True

                if RANK == 0:
                    print(f"Data loaded with {len(df)} sequences")

                # Map labels to integers, final label change if category column is not Pfam_id
                if category_col != "Pfam_id":
                    df["label"] = df[category_col].apply(map_label)

                # store start and end positions for EVAL mode
                if self.training is False and self.usage_mode is False:
                    self.all_starts = df["start"].tolist()
                    self.all_ends = df["end"].tolist()

                # Drop unnecessary columns to save memory and prevent data leakage, based on use case
                if category_col != "Pfam_id":
                    df.drop(columns=[category_col], inplace=True)
                elif self.usage_mode is False:
                    df.drop(columns=["id", "Pfam_id"], inplace=True)

                # Filter out sequences shorter than 10 residues for biological reasonings
                df = df[df[sequence_col].str.len() >= 10]
                # reference to dataframe
                self.df = df
                if RANK == 0:
                    print("Data loaded")

                # Use sequences in original order
                sequences = self.df[sequence_col].tolist()

                # generate labels per residue for domain boundary detection
                if RANK == 0:
                    print("Switching labels for domain boundary detection")
                labels_list = []
                # loop through dataframe rows
                for _, row in self.df.iterrows():
                    # gather sequence length
                    seq_len = len(row[sequence_col])
                    # Init label list with zeros
                    label = [0] * seq_len
                    # if not in usage mode, set domain regions to 1 in label
                    if self.usage_mode is False:
                        # gather start and end positions
                        start = int(row["start"])
                        end = int(row["end"])
                        # set label positions to 1 for domain region if start and end are within sequence length
                        if start < seq_len and end < seq_len:
                            # set for each residue label within start and end boundary to 1
                            for pos in range(start, end + 1):
                                label[pos] = 1

                    # append label tensor to full labels list
                    labels_list.append(torch.tensor(label, dtype=torch.long))

                # assign to self.labels
                self.labels = labels_list

                # # debug prints for labels
                # print(self.labels[3:4])  # Print first 5 labels for verification
                # print("SHAPE OF LABELS:")
                # print(self.labels[0].shape)

                # start time stamp
                start_time = time.time()

                # Embed sequences with _embed function
                self._embed(sequences)


            print("Done Embedding! Closing Embedder")
            # Return early if domain boundary detection is enabled, due to saved embeddings in _embed function
            return

        # Relic from previous re-embedding for boundary detection technique, for second embedding round
        else:
            self.df = skip_df

    def esm_loader(self):
        """
        Fetch function to load the specified ESM model and batch converter used for embedding.
        Chosen model is based on given self.esm_model parameter.
        FSDP or DDP wrapping is applied based on self.FSDP_used parameter.
        FSDP is still experimental and not stress tested, but should allow application on larger models.
        Returns:
            model: The loaded ESM model wrapped in DDP or FSDP for distributed training.
            batch_converter: The batch converter associated with the ESM model for preparing input data.
        """

        # Load the specified ESM model and prepare it for embedding.
        model_name = self.esm_model

        # determine number of layers from model name with user print
        self.model_layers = int(model_name.split("_")[1][1:])
        if RANK == 0:
            print(f"Loading ESM model: {model_name} with {self.model_layers} layers")

        # download model and regression data
        model_data, regression_data = (
            esm.pretrained._download_model_and_regression_data(model_name)
        )

        # possible FSDP usage for large models, not stress tested but should work (based on ESM FSDP example on github)
        # facebookresearch/esm · examples/esm2_infer_fairscale_fsdp_cpu_offloading.py
        if self.FSDP_used is True:
            print("Using FSDP for model wrapping")
            # initialize the model with FSDP wrapper
            fsdp_params = dict(
                mixed_precision=True,
                flatten_parameters=True,
                state_dict_device=torch.device("cpu"),
                cpu_offload=True,
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

        # Use DDP for distributed training instead
        else:
            # get model and batch_converter from esm fetcher
            model, alphabet = getattr(esm.pretrained, model_name)()
            batch_converter = alphabet.get_batch_converter()

            # disables dropout for deterministic results
            model.eval()
            # move model to GPU
            model = model.cuda()
            # convert to half precision for memory and time savings
            model = model.half()
            # wrap model in DDP based on DEVICE_ID
            model = DDP(model, device_ids=[DEVICE_ID])

        # return model and batch converter needed for _embed function
        return model, batch_converter

    def _embed(self, seqs):
        """
        Generate embeddings for a list of sequences using the preloaded ESM model from esm_loader func.
        Handles sequences longer than 1000 characters by cutting them into sliding windows. See windower func.
        Args:
            seqs (list): A list of input sequences to generate embeddings for.
        Returns:
            all_embeddings (list): A list of generated embeddings for the input sequences.
            all_labels (list): A list of labels corresponding to the input sequences.
            all_starts (list, optional): A list of start positions for each sequence (if EVAL mode).
            all_ends (list, optional): A list of end positions for each sequence (if EVAL mode).
            idx_multiplied (list): A list tracking original indices for windowed sequences.
        """

        def windower(stepsize=500, dimension=1000):
            """
            Cut sequences longer than 1000 characters into sliding windows to comply with ESM2 limitations.
            It creates windows of specified dimension with a given step size.
            It is covered for the shift in labels, meaning each new windowed sequence gets the corresponding label from the original sequence.
            Args:
                stepsize (int): The step size for sliding windows. Default is 500. Meaning we get overlapping windows by 50%.
                dimension (int): The window size for cutting sequences. Default is 1000, based on ESM2 limitations.
            Returns:
                new_seqs (list): A list of sequences after applying sliding window.
                new_labels (list): A list of labels corresponding to the new sequences.
                idx_multiplied (list): A list tracking original indices for windowed sequences.
                if self.training is False (EVAL mode):
                    new_starts (list): A list of start positions for each new sequence.
                    new_ends (list): A list of end positions for each new sequence.
            """

            # init variables needed
            new_seqs = []
            new_labels = []
            # n of sequences cut
            count = 0
            # Track which original indices were windowed
            idx_multiplied = []

            # ------------- Train mode windowing BLOCK -------------
            if self.training is True:
                # loop through sequences and labels as a zip between seqs and labels with index
                for idx, (seq, label) in enumerate(
                    zip(seq_dataset.seqs, seq_dataset.labels)
                ):
                    # if sequence longer than 1000, cut into windows
                    if len(seq) > 1000:
                        # +1 window count
                        count += 1
                        # Create slices based on Args for range sequence length
                        slices = [
                            seq[i : i + dimension]
                            for i in range(0, len(seq) - dimension + 1, stepsize)
                        ]
                        # Add last slice if not evenly divisible, dont care about overlap here
                        if len(seq) % dimension != 0:
                            slices.append(seq[-dimension:])
                        # Extend new sequences and labels
                        new_seqs.extend(slices)
                        new_labels.extend([label] * len(slices))
                        # Track that this original index was windowed
                        idx_multiplied.extend([idx] * len(slices))
                    # else just add original sequence and label
                    else:
                        new_seqs.append(seq)
                        new_labels.append(label)
                        idx_multiplied.append(idx)

            # ------------- Eval mode Classifiation windowing -------------
            elif self.training is False and self.domain_boundary_detection is False:
                # Additional init of new_starts and new_ends lists
                new_starts = []
                new_ends = []
                # loop through zipped sequences, labels, starts and ends with index
                for idx, (seq, label, start, end) in enumerate(
                    zip(
                        seq_dataset.seqs,
                        seq_dataset.labels,
                        seq_dataset.starts,
                        seq_dataset.ends,
                    )
                ):
                    # if sequence longer than 1000, cut into windows
                    if len(seq) > 1000:
                        count += 1
                        # Create slices based on Args for range sequence length
                        slices = [
                            seq[i : i + dimension]
                            for i in range(0, len(seq) - dimension + 1, stepsize)
                        ]
                        # Add last slice if not evenly divisible, dont care about overlap here
                        if len(seq) % dimension != 0:
                            slices.append(seq[-dimension:])
                        # Extend new sequences and labels with starts and ends
                        new_seqs.extend(slices)
                        new_labels.extend([label] * len(slices))
                        new_starts.extend([start] * len(slices))
                        new_ends.extend([end] * len(slices))
                        # Track that this original index was windowed
                        idx_multiplied.extend([idx] * len(slices))
                    # else just add original sequence, label, start and end
                    else:
                        new_seqs.append(seq)
                        new_labels.append(label)
                        new_starts.append(start)
                        new_ends.append(end)
                        idx_multiplied.append(idx)

            # ------------- Domain boundary detection mode windowing -------------
            elif self.domain_boundary_detection is True:
                # Additional init of new_starts and new_ends lists
                new_starts = []
                new_ends = []

                # usage mode handling with dummy starts and ends. Placeholder, not needed because we gather those data by ourselves later
                if self.training is False and self.usage_mode is True:
                    # Create dummy start and end values for each sequence
                    starts = [0] * len(seq_dataset.seqs)
                    ends = [len(seq) for seq in seq_dataset.seqs]
                    new_starts = []
                    new_ends = []
                # real start and end handling for EVAL mode boundary
                else:
                    starts = seq_dataset.starts
                    ends = seq_dataset.ends

                # loop through zipped sequences, labels, starts and ends with index
                for idx, (seq, label, start, end) in enumerate(
                    zip(
                        seq_dataset.seqs,
                        seq_dataset.labels,
                        starts,
                        ends,
                    )
                ):
                    # if sequence longer than 1000, cut into windows
                    if len(seq) > 1000:
                        if RANK == 0:
                            count += 1
                        slices = []
                        slice_positions = []
                        # Create slices based on Args for range sequence length with tracking positions for label windowing
                        for i in range(0, len(seq) - dimension + 1, stepsize):
                            slices.append(seq[i : i + dimension])
                            slice_positions.append((i, i + dimension))
                        # Add last slice if not evenly divisible, dont care about overlap here
                        if len(seq) % dimension != 0:
                            slices.append(seq[-dimension:])
                            slice_positions.append((len(seq) - dimension, len(seq)))

                        # Extend new sequences
                        new_seqs.extend(slices)

                        # Window labels according to slice positions
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
                            # Truncate if longer than dimension
                            elif windowed_label.shape[0] > dimension:
                                windowed_label = windowed_label[:dimension]

                            # Append the windowed label to the new labels
                            new_labels.append(windowed_label)

                        # Extend starts and ends
                        new_starts.extend([start] * len(slices))
                        new_ends.extend([end] * len(slices))
                        # Track that this original index was windowed
                        idx_multiplied.extend([idx] * len(slices))

                    # If we are in EVAL mode, we need to keep track of the original sequence boundaries
                    else:
                        new_seqs.append(seq)
                        # Ensure label is padded/truncated to dimension=1000 for consistency in dimensions to concat later
                        if label.shape[0] < dimension:
                            padded_label = torch.nn.functional.pad(
                                label, (0, dimension - label.shape[0]), value=0
                            )
                        elif label.shape[0] > dimension:
                            padded_label = label[:dimension]
                        else:
                            padded_label = label

                        # Append to all lists: label, start and end and idx_multiplied
                        new_labels.append(padded_label)
                        new_starts.append(start)
                        new_ends.append(end)
                        idx_multiplied.append(idx)

                # Update the dataframe with the new sequences and labels if windowing occurred
                if len(new_seqs) != len(seq_dataset.seqs):
                    # Update dataframe directly without intermediate list
                    self.df = pd.DataFrame(
                        {self.sequence_col: new_seqs, "label": new_labels}
                    )

                    # Add start/end columns if needed
                    if self.training is False:
                        self.df["start"] = new_starts
                        self.df["end"] = new_ends

                    if RANK == 0:
                        print(
                            f"Updated dataframe from {len(seq_dataset.seqs)} to {len(new_seqs)} sequences due to sliding window"
                        )

            # Print warning if any sequences were windowed per rank
            for rank in range(dist.get_world_size()):
                print(
                    f"Rank {rank}: Warning: {count} sequences were longer than 1000 characters and slided into windows"
                )

            # Returns training mode
            if self.training is True:
                return new_seqs, new_labels, idx_multiplied
            # Returns EVAL/usage mode
            else:
                return new_seqs, new_labels, new_starts, new_ends, idx_multiplied

        # -------------------------------- Main _embed function logic ---------------------------------------

        # Determine the correct embedding dimension from the model name
        # create a mapping of model names to their embedding dimensions
        model_dims = {
            "esm2_t6_8M_UR50D": 320,
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
            "esm2_t48_15B_UR50D": 5120,
        }

        # get dims
        expected_dim = model_dims.get(self.esm_model, None)

        # init output lists
        all_embeddings = []
        all_labels = []
        if self.domain_boundary_detection is True or self.training is False:
            all_starts = []
            all_ends = []

        # Convert seqs to a list if it's not already
        seqs = list(seqs)

        if self.usage_mode is True:
            # dummy labels for usage mode - make them all the same size (1000) for consistency to allow tensor concatenation later
            self.labels = [torch.zeros(1000, dtype=torch.long) for seq in seqs]

        # Determine the appropriate Dataset class based on the mode
        if (
            self.domain_boundary_detection is True
            or (
                self.training is False
                and hasattr(self, "all_starts")
                and hasattr(self, "all_ends")
                and self.all_starts is not None
                and self.all_ends is not None
            )
        ):
            # Domain boundary detection OR EVAL mode with positions
            seq_dataset = SeqDatasetForEval(
                seqs, self.labels, self.all_starts, self.all_ends
            )
        else:
            # Training mode or fallback (when positions aren't available)
            seq_dataset = SeqDataset(seqs, self.labels)
        
        # ESM2 limitation
        max_dimension = 1000

        # Apply windower func to cut sequences longer than 1000 into windows
        if self.training is True:
            new_seqs, new_labels, idx_multiplied = windower(stepsize=500, dimension=max_dimension)
            seq_dataset.seqs = new_seqs
            seq_dataset.labels = new_labels
        else:
            new_seqs, new_labels, new_starts, new_ends, idx_multiplied = windower(stepsize=500, dimension=max_dimension)
            seq_dataset.seqs = new_seqs
            seq_dataset.labels = new_labels
            seq_dataset.starts = new_starts
            seq_dataset.ends = new_ends


        if self.domain_boundary_detection is True:
            # Convert seq_dataset.labels to a tensor if it's a list facilitate overview
            if isinstance(seq_dataset.labels, list):
                if isinstance(seq_dataset.labels[0], torch.Tensor):
                    seq_dataset.labels = torch.stack(seq_dataset.labels)
                else:
                    seq_dataset.labels = torch.tensor(
                        seq_dataset.labels, dtype=torch.long
                    )
        # Debug prints for verification
        # if RANK == 0:
        #     print(seq_dataset.labels.shape)
        #     print(len(seq_dataset))
        #     print( f"Seq: {seq_dataset.seqs[0:5]}, Label: {seq_dataset.labels[0:5]}")


        # Create a DataLoader with DistributedSampler for distributed training, drop last to ensure equal batch sizes and no deadlocks
        sampler = DistributedSampler(
            seq_dataset,
            num_replicas=dist.get_world_size(),
            rank=RANK,
            shuffle=False,
            drop_last=True,
        )

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

                    # Distribute the remainder among the first 'remainder' ranks
                    if self.rank < remainder:
                        self.num_samples = samples_per_rank + 1
                    # For ranks >= remainder, assign the standard number of samples
                    else:
                        self.num_samples = samples_per_rank

            # init custom sampler for ddp
            sampler = CustomDistributedSampler(
                seq_dataset,
                num_replicas=dist.get_world_size(),
                rank=RANK,
                shuffle=False,
            )
            if RANK == 0:
                print("SAMPLER FOR USAGE MODE ACTIVE")

        # init Dataloader
        dataloader = DataLoader(
            seq_dataset,
            batch_size=self.emb_batch,
            num_workers=8,
            sampler=sampler,
            pin_memory=True,
            prefetch_factor=2,
        )

        # prints
        if RANK == 0:
            # print(f"Expected embedding dimension: {expected_dim}")
            print(
                f"Total sequences to process: {len(seq_dataset.seqs)} with batch size {self.emb_batch}"
            )
            print(f"Each Rank will process sequences {sampler.num_samples}")

        # Pre-allocate tensors on CPU to save GPU memory and improve speed
        total_samples = len(seq_dataset)
        if self.domain_boundary_detection:
            pass  # This logic will be handled inside the loop for domain boundary detection
        else:
            all_embeddings = torch.empty(
                (total_samples, expected_dim), dtype=torch.float32, device="cpu"
            )
            all_labels = torch.empty(total_samples, dtype=torch.long, device="cpu")
            # for EVAL mode, pre-allocate starts and ends
            if not self.training:
                all_starts = torch.empty(total_samples, dtype=torch.long, device="cpu")
                all_ends = torch.empty(total_samples, dtype=torch.long, device="cpu")

        # Flags for processed_samples and chunk number
        processed_samples = 0
        # Clear the dataset to free memory
        seq_dataset = None  

        # Embedding loop over dataloader with batch_num for progress tracking and actual batch_data
        for batch_num, batch_data in enumerate(dataloader):

            # Different unpacking based on training or EVAL mode
            if self.training is True:
                batch_seqs, batch_labels = batch_data
            else:
                batch_seqs, batch_labels, batch_start, batch_end = batch_data

                # debug prints for first batch
                # if batch_num == 0 and RANK == 0:
                #     print(batch_seqs, batch_labels, batch_start, batch_end)

            # try multi sample embedding in batches first
            try:
                # batch_seqs and batch_labels are now properly paired
                batch = [(f"seq{i}", seq) for i, seq in enumerate(batch_seqs)]

                # Convert sequences to tokens and move to GPU
                _, _, batch_tokens = self.batch_converter(batch)
                batch_tokens = batch_tokens.cuda(non_blocking=True)

                # Forward pass through the model to get embeddings
                with torch.inference_mode():
                    results = self.model(
                        batch_tokens,
                        repr_layers=[self.model_layers],    # last layer as representation
                        return_contacts=False,
                    )

                # Extract the embeddings from the last layer
                embeddings = results["representations"][self.model_layers].float()

                # Create mask to identify sequence tokens (excluding padding, where padding token ID = 1)
                mask = batch_tokens != 1
                # Remove <cls> (first token) and <eos> (last token) positions from mask
                mask = mask[:, 1:-1]
                # remove <eos> and <cls> tokens, made by esm model (start and end tokens)
                embeddings = embeddings[
                    :, 1:-1, :
                ]  

                # Print first 2 batches for debugging
                # if RANK == 0 and batch_num < 2:
                #     print(
                #         f"Batch {batch_num + 1}: {embeddings.shape} embeddings, {mask.shape} mask"
                #     )
                #     print(embeddings[0:2, 0:5, 0:5])

                
                # no pooling as we need per residue embeddings for domain boundary detection
                if self.domain_boundary_detection is True:
                    all_embeddings.append(embeddings.cpu())
                    all_labels.append(batch_labels.cpu())
                    # if EVAL mode store starts and ends
                    if self.training is False:
                        all_starts.append(batch_start)
                        all_ends.append(batch_end)

                # Pooling for Classification as we decide per sequence
                else:
                    lengths = mask.sum(dim=1, keepdim=True)
                    pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / lengths
                    # Move to CPU immediately, free vram
                    pooled = pooled.cpu()

                    # Store pooled embeddings and labels in pre-allocated tensors
                    start_idx = processed_samples
                    end_idx = start_idx + pooled.size(0)

                    # assign to preallocated tensors
                    all_embeddings[start_idx:end_idx] = pooled.cpu()
                    all_labels[start_idx:end_idx] = batch_labels.cpu()
                    if not self.training:
                        all_starts[start_idx:end_idx] = batch_start.cpu()
                        all_ends[start_idx:end_idx] = batch_end.cpu()

                    processed_samples = end_idx

                # Progress bar initialization
                if batch_num == 0 and RANK == 0:
                    progress_bar = tqdm(
                        total=len(dataloader),
                        desc="Embedding Data",
                        position=0,
                        leave=True,
                        ncols=150,
                    )

                # Update progress bar and memory stats every 1 batch or on last batch
                if (
                    batch_num % 1 == 0
                    or batch_num == len(dataloader) - 1
                    or batch_num == 0
                ) and RANK == 0:
                    # Get memory stats
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    memory_gb = memory_info.rss / 1024 / 1024 / 1024
                    system_memory = psutil.virtual_memory()
                    total_gb = system_memory.total / 1024 / 1024 / 1024
                    available_gb = system_memory.available / 1024 / 1024 / 1024
                    used_percent = system_memory.percent

                    # Update progress bar with all information
                    progress_bar.update(
                        1 if batch_num == 0 else min(100, batch_num - progress_bar.n)
                    )
                    # postfix with memory info
                    progress_bar.set_postfix(
                        {
                            "RAM": f"{memory_gb:.1f}GB ({used_percent:.0f}%, {available_gb:.1f}GB free)",
                            "Total": f"{total_gb:.1f}GB",
                        }
                    )

            except Exception as e:
                # Handle exceptions during batch processing, if vram blows up, we will try to process each sequence individually
                # can be ignored because it never triggers during ram oom
                print(f"\nRank {RANK}: Error processing batch {batch_num + 1}: {e}\n")
                # Handle individual sequences...
                for seq, label in zip(batch_seqs, batch_labels):
                    try:
                        # gather single seq
                        single_seq = [("seq0", seq)]
                        # convert to tokens and move to gpu
                        _, _, single_tokens = (
                            self.batch_converter(single_seq)
                        )
                        single_tokens = single_tokens.cuda()

                        # forward pass
                        with torch.inference_mode():
                            results = self.model(
                                single_tokens,
                                repr_layers=[self.model_layers], # last layer = representative layer
                                return_contacts=False,
                            )
                            # extract embeddings
                            embedding = results["representations"][
                                self.model_layers
                            ].float()

                            # masking padding and removing <cls> and <eos> tokens
                            mask = single_tokens != 1
                            mask = mask[:, 1:-1]
                            embedding = embedding[:, 1:-1, :]

                            # no pooling for domain boundary detection, keep per residue embeddings
                            if self.domain_boundary_detection is True:
                                # Move to CPU immediately, free vram
                                embedding = embedding.cpu() 
                                all_embeddings.append(embedding)
                                all_labels.append(batch_labels)

                            # Pooling for classification tasks
                            else:
                                length = mask.sum(dim=1, keepdim=True)
                                pooled = (embedding * mask.unsqueeze(-1)).sum(
                                    dim=1
                                ) / length
                                # Move to CPU immediately, free vram
                                pooled = pooled.cpu()

                                # Store pooled embeddings and labels in pre-allocated tensors
                                all_embeddings.append(pooled)
                                all_labels.append(batch_labels)
                            # if EVAL mode or Domain boundary detection is active, store starts and ends
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
                        # done to keep alignment between embeddings and labels
                        # never triggers, at least never seen it trigger
                        print(
                            f"\nRank {RANK}: Error processing individual sequence: {single_e}\n"
                        )
                        # append zero embedding in classification format
                        if not self.domain_boundary_detection:
                            start_idx = processed_samples
                            end_idx = start_idx + 1
                            all_embeddings[start_idx:end_idx] = torch.zeros(
                                1, expected_dim
                            )
                            all_labels[start_idx:end_idx] = label.unsqueeze(0)
                            processed_samples = end_idx
                        # append all zero embedding in domain boundary detection format
                        else:
                            zero_embedding = torch.zeros(1, expected_dim)
                            all_embeddings.append(zero_embedding)
                            all_labels.append(
                                label.unsqueeze(0)
                            )
                        # and if in EVAL mode or domain boundary detection, placeholder starts and ends
                        if (
                            self.training is False
                            or self.domain_boundary_detection is True
                        ):
                            all_starts.append(torch.tensor([0], dtype=torch.long))
                            all_ends.append(torch.tensor([len(seq)], dtype=torch.long))


            # Save part of the embed loop for domain boundary detection
            if self.domain_boundary_detection is True:
                # every 3000 batches or last batch, save to h5py file to avoid memory issues
                if (
                    batch_num % 3000 == 0
                    and batch_num > 0
                    or batch_num == len(dataloader) - 1
                ):
                    
                    # debug prints before padding
                    # if RANK == 0:
                    # print(all_labels[0].shape, "labels shape before padding")

                    # Verify all embeddings and labels are already the correct size
                    for i, (emb, lbl) in enumerate(zip(all_embeddings, all_labels)):
                        # if not: padding to max_dimension
                        if emb.shape[1] != max_dimension:
                            all_embeddings[i] = torch.nn.functional.pad(
                                emb, (0, 0, 0, max_dimension - emb.shape[1]), value=0
                            )

                    # debug prints after padding
                    # if RANK == 0:
                    #     print(all_labels[0].shape, "labels shape after padding")

                    # start with current rank concat to torch tensor
                    embeddings_tensor = torch.cat(all_embeddings, dim=0)
                    labels_tensor = torch.cat(all_labels, dim=0)

                    # Gather data to RANK 0
                    gathered_embeddings = [None] * dist.get_world_size()
                    gathered_labels = [None] * dist.get_world_size()
                    # use dist.gather to collect tensors from all ranks to rank 0 for embeddings and labels
                    dist.gather(
                        embeddings_tensor,
                        gather_list=gathered_embeddings if RANK == 0 else None,
                        dst=0,
                    )
                    dist.gather(
                        labels_tensor,
                        gather_list=gathered_labels if RANK == 0 else None,
                        dst=0,
                    )
                    # Ensure all ranks are synchronized before saving
                    dist.barrier()

                    # save path handling, different for usage mode and normal mode
                    if self.usage_mode is True:
                        save_path = "/global/research/students/sapelt/Masters/MasterThesis/tempTest/embeddings/embeddings_domain.h5"
                    else:
                        save_path = "/global/research/students/sapelt/Masters/MasterThesis/temp/embeddings_domain.h5"

                    # Only RANK 0 writes to the h5 file, others wait
                    if RANK == 0:
                        print("\n\nWriting...")
                        with h5py.File(save_path, "a") as f:
                            # Concatenate tensors from all ranks
                            final_embeddings = torch.cat(
                                gathered_embeddings, dim=0
                            ).numpy()
                            final_labels = torch.cat(gathered_labels, dim=0).numpy()

                            f[f"embeddings_chunk_{batch_num}"] = final_embeddings
                            f[f"labels_chunk_{batch_num}"] = final_labels
                        print(f"Wrote chunk {batch_num} to file")

                    # Clear the lists to free memory, should do the job, but somewhere i leak memory
                    del embeddings_tensor, labels_tensor

                    # Reinitialize the deleted lists
                    all_embeddings = []
                    all_labels = []

                    # Force garbage collection & synchronize
                    gc.collect()
                    torch.cuda.empty_cache()
                    dist.barrier()
        
        # Close progress bar if it exists
        if RANK == 0 and 'progress_bar' in locals():
            progress_bar.close()

        # Synchronize before finalizing
        dist.barrier()
        
        if RANK == 0:
            print("\nEmbeddings DONE!\n")

        # Return appropriate values based on mode
        if self.domain_boundary_detection:
            # Data already saved to h5 in chunks, just return idx_multiplied
            return 
        elif self.training is False:
            # EVAL mode: return embeddings, labels, starts, ends
            return all_embeddings, all_labels, all_starts, all_ends, idx_multiplied
        else:
            # Training mode: return embeddings and labels only
            return all_embeddings, all_labels, None, None, idx_multiplied
    
# not intended to be run directly, raise error
if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be run directly")
