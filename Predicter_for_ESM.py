"""
Predicter_for_ESM.py

Table of Contents:

1. Imports & basic setup
2. GLOBALS
3. Dataset Class for Evaluation
4. Opener Function
5. Predicter
6. Cutter / Re-embedder
7. Drop checker
8. Boundary gatherer
9. Cutter-Loop
10. Result List Creation & Concatenation
11. Plotting Functions
12. Main

Classes:
- EvalDataset

Functions:
- opener
- predict
- cut_inputs_embedding
- check_dropping_logits_across_cuts
- boundary_gatherer
- cutter_loop
- list_creater
- list_concatenater
- plotter
- boxplotter
- main
"""
# -------------------------
# 1. Imports & basic setup
# -------------------------
import os
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tqdm
from sklearn.metrics import (
    accuracy_score,
    # confusion_matrix,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from ESM_Embedder import DEVICE, RANK
from ESM_Embeddings_HP_search import ESMDataset, LitClassifier, FFNClassifier

torch.set_float32_matmul_precision("high")

os.environ["NCCL_P2P_DISABLE"] = "1"

# -------------------------
# 2. GLOBALS
# -------------------------

GLOBAL_RUN = 0  # keep 0
NUM_CLASSES = 3 # including 0 class

CSV_PATH = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/Evalsets/DataEvalSwissProt2d_esm_shuffled.csv"
CATEGORY_COL = "Pfam_id"
SEQUENCE_COL = "Sequences"
MODEL_PATH = "/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/t33_ALL_2d.pt"
CACHE_PATH = f"/global/research/students/sapelt/Masters/MasterThesis/temp/embeddings_classification_{NUM_CLASSES - 1}d_EVAL.h5"
TENSBORBOARD_LOG_DIR = (
    f"/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/{NUM_CLASSES - 1}d_uncut_ALL_cut_loop"
)

ESM_MODEL = "esm2_t33_650M_UR50D"

BATCH_SIZE = 512
EMB_BATCH = 64
NUM_WORKERS = min(16, os.cpu_count())
NUM_WORKERS_EMB = min(16, os.cpu_count())

THRESHOLD = 5  # Number of consecutive drops to trigger exclusion, relic from old approach
if RANK == 0:
    print("Treshold:", THRESHOLD) 


# -------------------------
# 3. Dataset Class for Evaluation
# -------------------------
class EvalDataset(Dataset):
    """
    Dataset class for loading ESM embeddings and labels from an HDF5 file for evaluation. Suited for Eval mode of ESMDataset.
    Args:
        h5_file: Path to the HDF5 file containing embeddings and labels.
    """

    def __init__(self, h5_file):
        # h5 file path
        self.h5_file = h5_file

        # open h5 file to get dataset info
        with h5py.File(self.h5_file, "r") as f:
            # Get all embedding keys
            self.embedding_keys = sorted(
                [k for k in f.keys() if k.startswith("embeddings_")]
            )

            # Debug: Print all keys and their shapes, if no keys found
            # if len(self.embedding_keys) == 0:
            #     print("No embedding keys found. Available keys:")
            #     for key in f.keys():
            #         print(f"  {key}: shape {f[key].shape}")

            # get sizes of each chunk and cumulative sizes for indexing
            self.chunk_sizes = [f[key].shape[0] for key in self.embedding_keys]
            self.cumulative_sizes = np.cumsum(self.chunk_sizes)
            # total length based on cumulative sizes
            self.length = (
                self.cumulative_sizes[-1] if self.cumulative_sizes.size > 0 else 0
            )

    def __len__(self):
        # return precomputed length
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, "r") as f:
            # chunk and local index calculation
            chunk_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")
            local_idx = (
                idx - self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else idx
            )

            # init keys for this chunk
            embedding_key = self.embedding_keys[chunk_idx]
            labels_key = embedding_key.replace("embeddings_", "labels_")
            starts_key = embedding_key.replace("embeddings_", "starts_")
            ends_key = embedding_key.replace("embeddings_", "ends_")

            # Get embeddings (2D array -> 1D vector)
            embeddings = torch.tensor(f[embedding_key][local_idx], dtype=torch.float32)

            # Get labels, starts, ends (1D arrays -> scalars)
            labels = torch.tensor(f[labels_key][local_idx], dtype=torch.long)
            starts = torch.tensor(f[starts_key][local_idx], dtype=torch.long)
            ends = torch.tensor(f[ends_key][local_idx], dtype=torch.long)

            # Debug print for first few items only
            # if RANK == 0:
            #     if idx < 10:
            #         print(f"Item {idx}: embeddings.shape={embeddings.shape}, labels={labels.item()}, starts={starts.item()}, ends={ends.item()}")

            # return all
            return embeddings, labels, starts, ends


# -------------------------
# 4. Opener Function
# -------------------------


def opener():
    """
    Opens the ESM model, loads the embeddings and labels from cache or computes them if not cached, via ESMDataset class.
    Returns:
        DataLoader for evaluation later.
    """
    # start print
    if RANK == 0:
        print("Opening data...")

    # check if cache exists, if not, create embeddings via ESMDataset
    if os.path.exists(
        f"/global/research/students/sapelt/Masters/MasterThesis/temp/progress_{NUM_CLASSES}.txt"
    ):
        with open(
            f"/global/research/students/sapelt/Masters/MasterThesis/temp/progress_{NUM_CLASSES}.txt", "r"
        ) as status_file:
            # continue only if not finished
            if "All chunks processed. Exiting." not in status_file.read():
                ESMDataset(
                    esm_model=ESM_MODEL,
                    FSDP_used=False,
                    domain_boundary_detection=False,
                    training=False,
                    num_classes=NUM_CLASSES,
                    csv_path=CSV_PATH,
                    category_col=CATEGORY_COL,
                    sequence_col=SEQUENCE_COL,
                )
                # pass
    elif not os.path.exists(CACHE_PATH):
        ESMDataset(
            esm_model=ESM_MODEL,
            FSDP_used=False,
            domain_boundary_detection=False,
            training=False,
            num_classes=NUM_CLASSES,
            csv_path=CSV_PATH,
            category_col=CATEGORY_COL,
            sequence_col=SEQUENCE_COL,
        )

    # Load the dataset from the HDF5 file
    evaldataset = EvalDataset(CACHE_PATH)

    # Shuffle the dataset by creating random indices
    indices = torch.randperm(len(evaldataset))
    evaldataset = torch.utils.data.Subset(evaldataset, indices.tolist())

    # optional subset for testing
    # evaldataset = torch.utils.data.Subset(evaldataset, list(range(100 * 5005)))

    sampler = torch.utils.data.distributed.DistributedSampler(
        evaldataset,
        num_replicas=dist.get_world_size(),
        rank=RANK,
        shuffle=True,
        drop_last=False
    ) 

    # init DataLoader
    eval_loader = DataLoader(
        evaldataset,  # Eval dataset
        sampler=sampler,
        batch_size=BATCH_SIZE,  # Global batch size
        # shuffle=True,  # Shuffle for eval
        persistent_workers=True,  # Keep workers alive, faster
        num_workers=NUM_WORKERS_EMB,  # Number of workers for loading
        pin_memory=True,  # Pin memory for faster transfer to GPU
        prefetch_factor=4,  # Prefetch factor for workers, faster loading
    )

    # end print
    if RANK == 0:
        print(f"Loaded {len(evaldataset)} sequences from {CACHE_PATH}")

    return eval_loader


# -------------------------
# 5. Predicter
# -------------------------
def predict(modelpath, loader, firstrun=False):
    """
    Predicts the classes for the given Eval dataloader using the specified pretrained model.
    Additionally logs metrics to TensorBoard if firstrun is True.
    Args:
        modelpath: Path to the pretrained model or the model object itself.
        loader: DataLoader containing the data to predict on.
        firstrun: Boolean indicating if this is the first run (for logging purposes).
    Returns:
        all_predictions: List of predicted classes.
        all_predictions_raw: List of raw prediction scores.
    """
    # Load model
    if isinstance(modelpath, str):
        model = torch.load(modelpath, map_location=DEVICE, weights_only=False)
        model = model.to(DEVICE)
    else:
        model = modelpath.to(DEVICE)
    
    # Wrap model with DDP if distributed training is enabled
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[RANK] if torch.cuda.is_available() else None,
            output_device=RANK if torch.cuda.is_available() else None
        )
    
    model.eval()
    # if RANK == 0:
    #     print(model)

    # Initialize lists to collect predictions and labels for this process
    all_predictions = []
    all_predictions_raw = []
    all_true_labels = []

    # Prediction loop
    with torch.inference_mode():
        for batch in tqdm.tqdm(loader, desc=f"Predicting batches (Rank {RANK})", disable=(RANK != 0)):
            if firstrun is True:
                inputs, true_labels, starts, ends = batch
            else:
                inputs, true_labels = batch


            # Store true labels for this batch
            all_true_labels.extend(true_labels.cpu().numpy())

            inputs = inputs.to(DEVICE)
            output = model(inputs)
            preds_raw, preds = torch.max(output, dim=1)

            # Accumulate predictions
            all_predictions.extend(preds.cpu().numpy())
            all_predictions_raw.extend(preds_raw.cpu().numpy())

    # Gather predictions from all processes if DDP is enabled
    if dist.is_initialized():
        # Move to CPU immediately to save GPU memory
        local_preds = torch.tensor(all_predictions, dtype=torch.long).cpu()
        local_preds_raw = torch.tensor(all_predictions_raw, dtype=torch.float32).cpu()
        local_labels = torch.tensor(all_true_labels, dtype=torch.long).cpu()
        
        if RANK == 0:
            # Only rank 0 gathers everything
            # Get sizes from all processes
            size_list = [None] * dist.get_world_size()
            dist.gather_object(local_preds.size(0), size_list if RANK == 0 else None, dst=0)
            
            # Gather actual data
            gathered_preds = [None] * dist.get_world_size()
            gathered_preds_raw = [None] * dist.get_world_size()
            gathered_labels = [None] * dist.get_world_size()
            
            dist.gather_object(local_preds, gathered_preds, dst=0)
            dist.gather_object(local_preds_raw, gathered_preds_raw, dst=0)
            dist.gather_object(local_labels, gathered_labels, dst=0)
            
            # Concatenate on CPU (rank 0 only)
            all_predictions = []
            all_predictions_raw = []
            all_true_labels = []
            
            for preds, preds_raw, labels in zip(gathered_preds, gathered_preds_raw, gathered_labels):
                all_predictions.extend(preds.numpy())
                all_predictions_raw.extend(preds_raw.numpy())
                all_true_labels.extend(labels.numpy())
            
            # Clean up gathered tensors
            del gathered_preds, gathered_preds_raw, gathered_labels
        else:
            # Other ranks just send their data
            dist.gather_object(local_preds.size(0), None, dst=0)
            dist.gather_object(local_preds, None, dst=0)
            dist.gather_object(local_preds_raw, None, dst=0)
            dist.gather_object(local_labels, None, dst=0)
        
        # Clean up local tensors
        del local_preds, local_preds_raw, local_labels

    # Calculate metrics ONCE after all batches are processed (only on rank 0)
    if firstrun is True and RANK == 0:
        os.makedirs(TENSBORBOARD_LOG_DIR, exist_ok=True)
        writer = SummaryWriter(TENSBORBOARD_LOG_DIR)

        # Overall metrics using ALL predictions and labels
        accuracy = accuracy_score(all_true_labels, all_predictions)
        weighted_precision = precision_score(
            all_true_labels, all_predictions, average="weighted", zero_division=0
        )

        # Class-specific precision & recall
        prec_per_class = precision_score(
            all_true_labels,
            all_predictions,
            labels=list(range(1, NUM_CLASSES)),
            average=None,
            zero_division=0,
        )
        rec_per_class = recall_score(
            all_true_labels,
            all_predictions,
            labels=list(range(1, NUM_CLASSES)),
            average=None,
            zero_division=0,
        )

        # Log overall metrics to TensorBoard
        writer.add_scalar("Metrics/Accuracy", accuracy, 0)
        writer.add_scalar("Metrics/Weighted_Precision", weighted_precision, 0)
        writer.add_scalar("Metrics/AvgPrecision", prec_per_class.mean(), 0)
        writer.add_scalar("Metrics/AvgRecall", rec_per_class.mean(), 0)

        # Create histograms
        prec_tensor = torch.tensor(prec_per_class)
        writer.add_histogram("Precision Distribution", prec_tensor, 0, bins=100)
        rec_tensor = torch.tensor(rec_per_class)
        writer.add_histogram("Recall Distribution", rec_tensor, 0, bins=100)

        # Log class-specific metrics
        for i in range(1, NUM_CLASSES):
            writer.add_scalar(f"Precision/Class_{i}", prec_per_class[i - 1], 0)
            writer.add_scalar(f"Recall/Class_{i}", rec_per_class[i - 1], 0)

        # Print summary
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print(f"Overall Weighted Precision: {weighted_precision:.4f}")
        
        total_precision = 0.0
        for i in range(1, NUM_CLASSES):
            mean_prec = prec_per_class[i - 1]
            mean_rec = rec_per_class[i - 1]
            print(f"Mean precision for class {i}: {mean_prec:.4f}, Mean recall: {mean_rec:.4f}")
            total_precision += mean_prec
        print(f"Mean precision over all classes: {total_precision / (NUM_CLASSES - 1):.4f}")

        writer.close()

    return all_predictions, all_predictions_raw


# -------------------------
# 6. Cutter / Re-embedder
# -------------------------


def cut_inputs_embedding(predictions, true_labels, df, cut_size, cut_front=True):
    """
    Main cut function used for the old, abandoned approach of cutting sequences and re-embedding them. Not used anymore, but kept for reference.
    Args:
        predictions: List of predicted classes.
        true_labels: List of true labels.
        df: DataFrame containing the sequences.
        cut_size: Size of the cut to apply.
        cut_front: Boolean indicating whether to cut from the front (True) or back (False).
    Returns:
        DataLoader with cut and re-embedded sequences, and their corresponding labels.
    """
    # init lists
    positive_embeddings = []
    positive_labels = []

    # loop through predictions and cut sequences accordingly
    for i in range(len(predictions)):
        # only reembed positive predictions
        if predictions[i] != 0:
            # cut on the set side
            if cut_front is True:
                # cut down from front
                temp_seq = df[SEQUENCE_COL][i][cut_size:]
            else:
                # cut down from back
                temp_seq = df[SEQUENCE_COL][i][:-cut_size]

            # append cut sequence and label
            positive_embeddings.append(temp_seq)
            # Handle both tensor and non-tensor labels
            label_value = true_labels[i].item() if isinstance(true_labels[i], torch.Tensor) else true_labels[i]
            positive_labels.append(label_value)

    # create dataframe for ESMDataset
    df = pd.DataFrame(
        {
            SEQUENCE_COL: positive_embeddings,
            "label": positive_labels,
        }
    )

    # test subset
    # df = df.sample(frac=0.01, random_state=42)  # Example: take a 1% random subset

    # create ESMDataset for re-embedding
    esm_data = ESMDataset(
        esm_model=ESM_MODEL,
        skip_df=df,
        FSDP_used=False,
        training=False,
        domain_boundary_detection=False,
        num_classes=NUM_CLASSES,
        csv_path=CSV_PATH,
        category_col=CATEGORY_COL,
        sequence_col=SEQUENCE_COL,
    )

    # relic of old approach, when functions did return and not write to h5 file
    # Each rank has its own portion of embeddings and labels
    local_embeddings = esm_data.embeddings
    local_labels = esm_data.labels
    local_df = esm_data.df

    # for rank in range(dist.get_world_size()):
    #     if RANK == rank:
    #         print(f"Rank {RANK} has {len(local_embeddings)} embeddings after cutting.")

    # Gather embeddings and labels from all ranks to all ranks
    if dist.is_initialized():
        # Gather from all ranks to rank 0 first
        if RANK == 0:
            gathered_embeddings = [None] * dist.get_world_size()
            gathered_labels = [None] * dist.get_world_size()
            gathered_dfs = [None] * dist.get_world_size()
        else:
            gathered_embeddings = None
            gathered_labels = None
            gathered_dfs = None
        
        # Gather to rank 0
        dist.gather_object(local_embeddings, gathered_embeddings, dst=0)
        dist.gather_object(local_labels, gathered_labels, dst=0)
        dist.gather_object(local_df, gathered_dfs, dst=0)
        
        # Concatenate on rank 0
        if RANK == 0:
            # Concatenate embeddings (assuming they're tensors)
            df_embeddings = torch.cat([e for e in gathered_embeddings if e is not None], dim=0)
            # Concatenate labels
            df_labels = torch.cat([torch.tensor(l) if not isinstance(l, torch.Tensor) else l 
                                  for l in gathered_labels if l is not None], dim=0)
            # Concatenate dataframes
            df = pd.concat([d for d in gathered_dfs if d is not None], ignore_index=True)
        else:
            df_embeddings = None
            df_labels = None
            df = None
        
        # Broadcast from rank 0 to all other ranks
        data_to_broadcast = [df_embeddings, df_labels, df]
        dist.broadcast_object_list(data_to_broadcast, src=0)
        
        # All ranks now have the complete data
        df_embeddings, df_labels, df = data_to_broadcast
        
        # Ensure all ranks are synchronized
        dist.barrier()

    # If not using distributed training, just use local data
    else:
        df_embeddings = local_embeddings
        df_labels = local_labels
        df = local_df


    # for rank in range(dist.get_world_size()):
    #     if RANK == rank:
    #         print(f"Rank {RANK} has {len(df_embeddings)} embeddings after sharing.")

    # basic Dataset
    df_embeddings = TensorDataset(df_embeddings, df_labels)

    # basic Dataloader
    df_embeddings = DataLoader(
        df_embeddings,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return df_embeddings, df_labels


# -------------------------
# 7. Drop checker
# -------------------------


def check_dropping_logits_across_cuts(logits_so_far, threshold=2):
    """
    Check if logits are dropping consecutively across different cut sizes.

    Args:
        logits_so_far: List of logit values collected so far for a sequence
        threshold: Number of consecutive drops to trigger exclusion (default: 2)

    Returns:
        bool: True if sequence should be dropped (has consecutive drops >= threshold)
    """

    # if logits length less than threshold + 1, cannot have enough drops. Return early
    if len(logits_so_far) < threshold + 1:
        return False

    # init drop counter
    consecutive_drops = 0

    # loop through logits and count consecutive drops
    for i in range(1, len(logits_so_far)):
        # check if current logit is less than or equal to previous
        if logits_so_far[i] <= logits_so_far[i - 1]:
            # add to drop counter
            consecutive_drops += 1
            # check if threshold reached, return True
            if consecutive_drops >= threshold:
                return True
        else:
            # reset drop counter if not met
            consecutive_drops = 0

    # if done loop, threshold not met, return False
    return False


# -------------------------
# 8. Boundary gatherer
# -------------------------


def boundary_gatherer(predictions, loader):
    """
    Gather the boundaries, true classes, true pos of predicted domains from the predictions list.
    For later evaluation of predicted boundaries.

    Args:
        predictions: List of predicted domain starts and ends.
        loader: DataLoader with sequence data

    Returns:
        pos_indices, pos_true_start, pos_true_end, true_class
    """

    # Initialize lists to collect the true results
    pos_indices = []
    pos_true_start = []
    pos_true_end = []
    true_class = []

    # Batch loop
    for batch_idx, batch in enumerate(loader):
        inputs, labels, df_starts, df_ends = batch

        # Determine the global indices for this batch
        batch_start_idx = batch_idx * BATCH_SIZE
        batch_end_idx = min(batch_start_idx + inputs.size(0), len(predictions))
        batch_predictions = predictions[batch_start_idx:batch_end_idx]

        # Find positive indices within this batch (local indices)
        batch_pos_indices = [i for i, p in enumerate(batch_predictions) if p != 0]

        # Convert to global indices
        global_pos_indices = [batch_start_idx + i for i in batch_pos_indices]

        # Extend the global pos_indices list
        pos_indices.extend(global_pos_indices)

        # Get true starts, ends, and labels for positive predictions in this batch
        if batch_pos_indices:  # Only process if there are positive predictions
            batch_pos_true_start = [df_starts[i].item() for i in batch_pos_indices]
            batch_pos_true_end = [df_ends[i].item() for i in batch_pos_indices]
            batch_true_class = [labels[i].item() for i in batch_pos_indices]

            # Extend the global lists
            pos_true_start.extend(batch_pos_true_start)
            pos_true_end.extend(batch_pos_true_end)
            true_class.extend(batch_true_class)

    # No positives found check
    if not pos_indices:
        if RANK == 0:
            print("No positives found.")
        return None, None, None, None

    # prints
    if RANK == 0:
        print(f"Total positives found: {len(pos_indices)}")
        print(f"pos_indices sample: {pos_indices[:5]}...")
        print(f"max pos_index: {max(pos_indices)}")

    return pos_indices, pos_true_start, pos_true_end, true_class


# -------------------------
# 9. Cutter-Loop
# -------------------------


def cutter_loop(
    predictions, true_labels, df, pos_indices, pos_true, raw_scores_nocut, name="start"
):
    """
    Main cutter loop function that iteratively cuts sequences, re-embeds them, predicts again, and tracks logits to determine boundaries.
    Still a relic from the old approach, not used anymore, but kept for reference. Might be bugged/not working anymore.
    Matrix-based implementation for efficiency with structure.
    Raw matrix structure: [sequence at question] [cut sizes]
    sequence_logits structure: [sequence at question] [domain count in the sequence] [logits per cut]
    Args:
        predictions: List of initial predicted classes.
        true_labels: List of true labels.
        df: DataFrame containing the sequences.
        pos_indices: List of indices of positive predictions.
        pos_true: List of true boundary positions for positive predictions.
        raw_scores_nocut: List of raw prediction scores without any cuts.
        name: String indicating whether cutting "start" or "end" boundaries.
    Returns:
        all_residues: List of predicted boundary positions for each positive sequence.
        errors: List of absolute errors between predicted and true boundaries.
    """
    # Determine how many cuts to perform based on sequence lengths
    # Allows to save time and keep a realistic biological sequence length for domains

    # Gather all sequence lengths
    seq_lengths = [len(seq) for seq in df[SEQUENCE_COL]]
    # get max, mean, min lengths
    max_len = max(seq_lengths)
    mean_len = np.mean(seq_lengths)
    min_len = min(seq_lengths)
    # determine cut sizes starting with 5 up to 80% of mean length or max_len/4, whichever is smaller
    cut_sizes = list(range(5, min(int(mean_len * 0.8), max_len // 4), 5))
    # if max cut size smaller than min length, extend cut sizes further
    cut_sizes = list(
        range(5, min(int(mean_len * 0.8), max_len // 4), 5)
        if max(cut_sizes) < min_len
        else range(5, int(max_len / 3), 5)
    )
    print(f"\nMax cut size: {max(cut_sizes)} residues")

    # BUILD UP OF MATRICES AND TRACKING VARIABLES, ADDING THE FIRST COLUMN WITH NO-CUT SCORES
    # number of positive sequences, number of cuts
    Npos, K = len(pos_indices), len(cut_sizes)
    # Initialize matrices and tracking variables
    raw_matrix = np.zeros((Npos, K), dtype=float)
    # Fill first column with no-cut raw scores
    for row_idx in range(Npos):
        raw_matrix[row_idx, 0] = raw_scores_nocut[pos_indices[row_idx]]
    # Maximum number of domains to track within a sequence
    max_domains = 1
    # matrix to track logits per sequence and domain
    sequence_logits = [
        [[] for _ in range(max_domains)] for _ in range(Npos)
    ]  # [sequence at question] [domain count in the sequence] [logits per cut]
    # put the raw_scores_nocut into the first column of the matrix
    for row_idx in range(Npos):
        sequence_logits[row_idx][0].append(raw_scores_nocut[pos_indices[row_idx]])
    # Track which domain each sequence is currently on
    sequence_domain_status = [0 for _ in range(Npos)]
    # init set with all sequences that dont need to be cut further
    stopped_sequences = set()

    # Loop through each cut size
    for j, cut in enumerate(cut_sizes):
        print(f"Processing cut size {cut} ({j + 1}/{K})...")

        # Determine active sequences that are not stopped
        active_sequences = [
            row_idx
            for row_idx in range(Npos)
            if row_idx not in stopped_sequences
        ]

        # If empty, break and quit early
        if not active_sequences:
            print("No active sequences left to process.")
            break

        # filter predictions and true_labels for active sequences & create filtered df
        # Use pos_indices to get global indices for filtering df
        global_indices = [pos_indices[row_idx] for row_idx in active_sequences]
        filtered_predictions = [predictions[i] for i in global_indices]
        filtered_true_labels = [true_labels[row_idx] for row_idx in active_sequences]
        filtered_df = df.iloc[global_indices].reset_index(drop=True)

        # print how many sequences are excluded due to stopping, more or less a progress report
        decrease_percentage = ((Npos - len(filtered_predictions)) / Npos) * 100
        print(
            f"Filtered predictions: {len(filtered_predictions)} / {Npos} ({decrease_percentage:.1f}% excluded)"
        )

        # Start the actual cutting, re-embedding, and predicting, depending on whether cutting start or end
        if name == "start":
            sub_loader, sub_labels = cut_inputs_embedding(
                filtered_predictions,
                filtered_true_labels,
                filtered_df,
                cut_size=cut,
                cut_front=True,
            )
        else:
            sub_loader, sub_labels = cut_inputs_embedding(
                filtered_predictions,
                filtered_true_labels,
                filtered_df,
                cut_size=cut,
                cut_front=False,
            )
        # predict on the cut and re-embedded sequences
        _, raw_scores = predict(MODEL_PATH, sub_loader, firstrun=False)

        # Update logits and check for stopping sequences
        # init idx to track position in raw_scores
        active_idx = 0
        # loop through only the active sequences, not all Npos
        for row_idx in active_sequences:  # Changed from: for row_idx in range(Npos)
            # get current domain
            current_domain = sequence_domain_status[row_idx]
            # update raw_matrix and sequence_logits
            raw_matrix[row_idx, j] = raw_scores[active_idx]
            # append current raw score to logits list
            sequence_logits[row_idx][current_domain].append(raw_scores[active_idx])

            # Check if this sequence should stop further prediction, if yes func returns True
            if check_dropping_logits_across_cuts(
                sequence_logits[row_idx][current_domain], threshold=THRESHOLD
            ):
                # move to next domain counter within sequence
                current_domain += 1
                # update domain status in matrix
                sequence_domain_status[row_idx] = current_domain

                # If we have not exceeded max_domains, add new domain list.
                if current_domain < max_domains:
                    # Check if previous domain has logits
                    prev_domain_logits = sequence_logits[row_idx][
                        current_domain - 1
                    ]
                    # if current score is better than previous domain max, append to current domain
                    if raw_scores[active_idx] > max(prev_domain_logits):
                        sequence_logits[row_idx][current_domain].append(
                            raw_scores[active_idx]
                        )
                # If exceeded, mark sequence as stopped
                else:
                    stopped_sequences.add(row_idx)

            # increment active idx to move to next active sequence
            active_idx += 1

        # Debug print for first few sequences
        # print(f"Sample sequence logits: {sequence_logits[0]}")

    # end of cutting loop
    print(
        f"Stopped early prediction for {len(stopped_sequences)} out of {Npos} sequences"
    )

    # Processing & Evaluating results. Init lists
    all_residues = []
    errors = []
    # loop through each positive sequence
    for row_idx in range(Npos):
        # get the logits curve for this sequence
        curve = raw_matrix[row_idx]
        # init lists for best cuts and residues
        all_j = []
        residues = []

        # Find the best cut among the cuts that were actually computed for this sequence
        for domain_idx in range(max_domains):
            # If this domain has logit
            if sequence_logits[row_idx][domain_idx]:
                # For stopped sequences, only consider cuts up to where they were stopped
                if row_idx in stopped_sequences:
                    non_zero_indices = np.nonzero(curve)[0]
                    # get best cut among non-zero indices
                    if len(non_zero_indices) > 0:
                        best_j = non_zero_indices[np.argmax(curve[non_zero_indices])]
                    # Fallback if no non-zero indices found to 0
                    else:
                        best_j = 0
                # For sequences that completed all cuts, consider all cuts
                else:
                    best_j = int(np.argmax(curve))
                # append best cut index to list
                all_j.append(best_j)

        # Convert cut indices to residue positions
        for j in all_j:
            residues.append(cut_sizes[j])

        # Calculate absolute error against true positions
        pred = residues
        true = pos_true[row_idx]

        # if no true positions, set error to None
        if not true:
            err = None
        else:
            all_errors = []
            # loop through predicted domains and find closest true domain
            for predicted_domain in pred:
                # calculate domain error as min distance to any true domain
                domain_error = min(abs(predicted_domain - ts) for ts in true)
                # append to all errors
                all_errors.append(domain_error)
            # get minimum error among all predicted domains
            err = min(all_errors)

        # append error and residues to lists
        errors.append(err)
        # append all predicted boundary postions to all_residues
        all_residues.append(residues)

    # Final error statistics without None values
    errors_wo_nones = [e for e in errors if e is not None]
    # print final stats & end messages
    print(
        f"Mean absolute error over {len(errors)} positives: {np.mean(errors_wo_nones):.1f} residues"
    )
    print(f"Median absolute error: {np.median(errors_wo_nones):.1f} residues\n")

    print(
        "#####################################################################################################"
    )
    print(
        f"----------------------------------------- Cut {name} done --------------------------------------------"
    )
    print(
        "#####################################################################################################"
    )

    return all_residues, errors


# -------------------------
# 10. Helper Functions
# -------------------------


def list_creater(df, pos_indices, all_residues_start, all_residues_end, predictions):
    """
    Create a list of dictionaries for each sequence with its details. Used for final result compilation.
    Args:
        df: DataFrame containing the sequences.
        pos_indices: List of indices of positive predictions.
        all_residues_start: List of predicted start boundary positions for each positive sequence.
        all_residues_end: List of predicted end boundary positions for each positive sequence.
        predictions: List of predicted classes.
    """
    # init endlist
    endlist = []
    # loop through positive indices and create entries
    for i, seq_idx in enumerate(pos_indices):
        entry = {
            "ID": df["ID"][seq_idx],  # Sequence ID
            "Class": predictions[seq_idx],  # Predicted class
            "Domain_Start": all_residues_start[i],  # Predicted start boundaries
            "Domain_End": all_residues_end[i],  # Predicted end boundaries
        }
        # append entry to endlist
        endlist.append(entry)

    # status prints
    print("Endlist length:", len(endlist))
    print("First few items:", endlist[:3])

    return endlist


def list_concatenater(endlist):
    """ 
    Concat the previously created list of dictionaries in list_creater by merging entries with the same ID that follow each other up in the list..
    Done to have a clean final result csv with all predicted domains per sequence in one row and not multiple rows per sequence ID.
    Args:
        endlist: List of dictionaries created by list_creater function.
    """
    # Sort by ID first to group same IDs together
    endlist.sort(key=lambda x: x["ID"])

    # init list & index
    concatenated_list = []
    i = 0

    # loop through endlist
    while i < len(endlist):
        # start with current row & copy it
        current_row = endlist[i].copy()

        # Check if next rows (j) have the same ID and concatenate them
        j = i + 1
        # while next rows have same ID and within bounds
        while j < len(endlist) and endlist[j]["ID"] == current_row["ID"]:
            # Concatenate Domain_Start and Domain_End lists
            if isinstance(current_row["Domain_Start"], list):
                current_row["Domain_Start"].extend([endlist[j]["Domain_Start"]])
            # if not list, convert to list and append
            else:
                current_row["Domain_Start"] = [
                    current_row["Domain_Start"],
                    endlist[j]["Domain_Start"],
                ]
            
            # same for Domain_End
            if isinstance(current_row["Domain_End"], list):
                current_row["Domain_End"].extend([endlist[j]["Domain_End"]])
            else:
                current_row["Domain_End"] = [
                    current_row["Domain_End"],
                    endlist[j]["Domain_End"],
                ]

            # move to next row
            j += 1

        # append the concatenated current row to the new list
        concatenated_list.append(current_row)
        # move i to j for next iteration
        i = j

    # Convert to DataFrame and save as CSV
    concatenated_df = pd.DataFrame(concatenated_list)
    concatenated_df.to_csv(
        "/global/research/students/sapelt/Masters/MasterThesis/Results/Predicter_from_ESM_final_result.csv",
        index=False,
    )
    print("Saved results to predicted_boundaries_after.csv")


# -------------------------
# 11. Plotter Functions
# -------------------------


def plotter(
    errors,
    Name,
    bin_width=5,
):
    """
    Plotter function to create histogram of absolute errors. Done for the plot 'Histogram of domain boundary prediction start and end errors' in the theis
    Args:
        errors: List of absolute errors, previouly gathered in cutter loop.
        Name: String indicating whether plotting "Start" or "End" errors.
        bin_width: Width of histogram bins.  
    """
    # filter out None errors
    errors = [e for e in errors if e is not None] 
    # determine max error for binning
    max_err = int(max(errors))
    bins = np.arange(0, max_err + bin_width, bin_width)

    # plt histogram & save
    plt.figure(figsize=(27, 10))
    plt.hist(errors, bins=bins, density=True, edgecolor="k", alpha=0.7)
    plt.xticks(bins, fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlim(0, 100)
    plt.title(f"Normalized Distribution of {Name} Boundary Absolute Errors")
    plt.xlabel(f"Absolute Error (residues), bins of {bin_width}")
    plt.ylabel("Probability Density")
    plt.grid(axis="y", alpha=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(f"/home/sapelt/Documents/Master/FINAL/Normalized Distribution of {Name} Boundary Absolute Errors", dpi=600)
    # plt.show()


def boxplotter(errors, classes, Name):
    """
    Boxplotter function to create boxplots of absolute errors grouped by class. Done for the plot 'Box plots of domain boundary prediction start and end errors'.
    Args:
        errors: List of absolute errors, previouly gathered in cutter loop.
        classes: List of true classes corresponding to each error.
        Name: String indicating whether plotting "Start" or "End" errors.
    """

    # Filter out None values from errors and their corresponding classes
    filtered_data = [(e, c) for e, c in zip(errors, classes) if e is not None]
    # unzip to separate lists
    filtered_errors, filtered_classes = (
        zip(*filtered_data) if filtered_data else ([], [])
    )

    # Dynamically group errors by class in a dictionary
    class_dict = {}
    # loop through filtered errors and classes
    for err, cls in zip(filtered_errors, filtered_classes):
        class_dict.setdefault(cls, []).append(err)

    # Sort classes and build data+labels
    keys = sorted(class_dict)
    data = [class_dict[k] for k in keys]
    labels = [f"Class {k}" for k in keys]

    # Create boxplot & save
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="C0", edgecolor="k"),
        medianprops=dict(color="yellow"),
    )
    ax.set_title(f"Absolute {Name} Error by Class")
    ax.set_ylabel("Absolute Residue Error")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0, top=100)  # Set y-axis limits
    plt.tight_layout()
    plt.savefig(f"/home/sapelt/Documents/Master/FINAL/boxplot_{Name}_errors.png", dpi=600)
    plt.show()


# -------------------------
# 12. Main
# -------------------------


def main():
    """
    Main function to orchestrate the loading, predicting, cutting, gathering, and plotting of domain boundary predictions using ESM embeddings.
    Uses all helper functions defined above.
    """

    # Open File and get DataLoader for Evaluation
    eval_loader = opener()
    if RANK == 0:
        print("\nStarting first prediction run...")

    # Initial Prediction on uncut sequences to gather predictions for classes and raw scores
    predictions, raw_scores_nocut = predict(MODEL_PATH, eval_loader, firstrun=True)

    # new exit because unused relic cut loop below
    # dist.barrier()  # Ensure all processes reach this point before quitting
    # if RANK == 0:
    #     print("Exiting after first run.")
    # dist.destroy_process_group()  # Clean up the process group
    # quit()
    
    # ------------------------------------
    # RELIC PART OF THE OLD CUT-AND-RE-EMBED APPROACH, NOT USED ANYMORE
    # ------------------------------------
    
    # set df to none for first cut loop
    df = pd.read_csv(CSV_PATH)
    
    # Gather boundaries and true classes from initial predictions
    pos_indices, pos_true_start, pos_true_end, true_class = boundary_gatherer(
        predictions, eval_loader
    )

    if "./pickle/checkpoint_cutloop.pkl" in os.listdir("./pickle/"):
        if RANK == 0:
            print("Loading checkpoint data from pickle...")
            with open("./pickle/checkpoint_cutloop.pkl", "rb") as f:
                checkpoint_data = pickle.load(f)
            predictions = checkpoint_data["predictions"]
            pos_indices = checkpoint_data["pos_indices"]
            all_residues_start = checkpoint_data["all_residues_start"]
            all_residues_end = checkpoint_data["all_residues_end"]
            errors_start = checkpoint_data["errors_start"]
            errors_end = checkpoint_data["errors_end"]
            true_class = checkpoint_data["true_class"]
        # Broadcast loaded data to all ranks
        data_to_broadcast = [
            predictions,
            pos_indices,
            all_residues_start,
            all_residues_end,
            errors_start,
            errors_end,
            true_class,
        ]
        dist.broadcast_object_list(data_to_broadcast, src=0)
        (
            predictions,
            pos_indices,
            all_residues_start,
            all_residues_end,
            errors_start,
            errors_end,
            true_class,
        ) = data_to_broadcast
        dist.barrier()  # Ensure all ranks are synchronized after loading

    else:
        # Cutter loops for start boundaries and end boundaries
        all_residues_start, errors_start = cutter_loop(
            predictions,        # predictions made in predict()             
            true_class,         # true classes gathered in boundary_gatherer()
            df,                 # dataframe, set to none for first cut loop
            pos_indices,        # positive sequences with domain class indices gathered in boundary_gatherer()
            pos_true_start,     # true start positions gathered in boundary_gatherer()
            raw_scores_nocut,   # raw scores from first predict() run
            name="start",       # cutting start boundaries
        )
        all_residues_end, errors_end = cutter_loop(
            predictions,        # predictions made in predict()
            true_class,         # true classes gathered in boundary_gatherer()
            df,                 # dataframe, set to none for first cut loop
            pos_indices,        # positive sequences with domain class indices gathered in boundary_gatherer()
            pos_true_end,       # true end positions gathered in boundary_gatherer()
            raw_scores_nocut,   # raw scores from first predict() run
            name="end",         # cutting end boundaries
        )

    # ------------------------------------
    # create checkpoint pickle
    if RANK == 0:
        checkpoint_data = {
            "predictions": predictions,
            "pos_indices": pos_indices,
            "all_residues_start": all_residues_start,
            "all_residues_end": all_residues_end,
            "errors_start": errors_start,
            "errors_end": errors_end,
            "true_class": true_class,
        }
        with open("./pickle/checkpoint_cutloop.pkl", "wb") as f:
            pickle.dump(checkpoint_data, f)
        print("Checkpoint data saved to checkpoint_cutloop.pkl")



    # Create final result list in list_creater() and concatenate entries with same ID in list_concatenater()
    endlist = list_creater(
        df, pos_indices, all_residues_start, all_residues_end, predictions
    )
    list_concatenater(endlist)

    # Plotting of errors in hists and boxplots
    plotter(errors_start, Name="Start", bin_width=5)
    plotter(errors_end, Name="End", bin_width=5)
    boxplotter(errors_start, true_class, Name="Start")
    boxplotter(errors_end, true_class, Name="End")


###############################################################################################################
if __name__ == "__main__":
    main()
