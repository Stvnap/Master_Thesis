import os
import pickle
import tqdm
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    # confusion_matrix,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from ESM_Embeddings_HP_search import ESMDataset, LitClassifier, FFNClassifier
from ESM_Embedder import DEVICE, RANK
import torch.distributed as dist    
torch.set_float32_matmul_precision("high")

os.environ["NCCL_P2P_DISABLE"] = "1"
# -------------------------
# 1. Global settings
# -------------------------

GLOBAL_RUN = 0  # keep 0

CSV_PATH = "./Dataframes/v3/FoundEntriesSwissProteins_Eval.csv"
CATEGORY_COL = "Pfam_id"
SEQUENCE_COL = "Sequence"
MODEL_PATH = "./models/FINAL/t33_ALL_2d.pt"
CACHE_PATH = "./temp/embeddings_classification_2d_EVAL.h5"
TENSBORBOARD_LOG_DIR = "./models/2d_uncut_ALL"

ESM_MODEL = "esm2_t33_650M_UR50D"


NUM_CLASSES = 3
BATCH_SIZE = 1000
EMB_BATCH = 1
NUM_WORKERS = min(16, os.cpu_count())
NUM_WORKERS_EMB = min(16, os.cpu_count())

THRESHOLD = 5  # Number of consecutive drops to trigger exclusion
if RANK == 0:
    print("Treshold:", THRESHOLD)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 2. Opener Function
# -------------------------


def opener():
    """
    Opens the ESM model, loads the embeddings and labels from cache or computes them if not cached.
    """
    if RANK == 0:
        print("Opening data...")
    
    if not os.path.exists(CACHE_PATH):
        if RANK == 0:
            print(f"Cache file {CACHE_PATH} not found. Running the embedding script first.")

        ESMDataset(
            esm_model=ESM_MODEL,
            FSDP_used=False,
            training=False,
            domain_boundary_detection=False,
            num_classes=NUM_CLASSES,
            csv_path=CSV_PATH,
            category_col=CATEGORY_COL,
            sequence_col=SEQUENCE_COL,
        )


    class EvalDataset(Dataset):
        def __init__(self, h5_file):
            self.h5_file = h5_file

            with h5py.File(self.h5_file, "r") as f:
                # Get all embedding keys
                self.embedding_keys = sorted(
                    [k for k in f.keys() if k.startswith("embeddings_")]
                )
                
                # Debug: Print all keys and their shapes
                if len(self.embedding_keys) == 0:
                    print("No embedding keys found. Available keys:")
                    for key in f.keys():
                        print(f"  {key}: shape {f[key].shape}")
                
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
                labels_key = embedding_key.replace("embeddings_", "labels_")
                starts_key = embedding_key.replace("embeddings_", "starts_")
                ends_key = embedding_key.replace("embeddings_", "ends_")

                # Get embeddings (2D array -> 1D vector)
                embeddings = torch.tensor(
                    f[embedding_key][local_idx], dtype=torch.float32
                )
                
                # Get labels, starts, ends (1D arrays -> scalars)
                labels = torch.tensor(f[labels_key][local_idx], dtype=torch.long)
                starts = torch.tensor(f[starts_key][local_idx], dtype=torch.long)
                ends = torch.tensor(f[ends_key][local_idx], dtype=torch.long)

                # Debug print for first few items only
                # if RANK == 0:
                #     if idx < 10:
                #         print(f"Item {idx}: embeddings.shape={embeddings.shape}, labels={labels.item()}, starts={starts.item()}, ends={ends.item()}")

                return embeddings, labels, starts, ends


    # Load the dataset from the HDF5 file
    evaldataset = EvalDataset(CACHE_PATH)
    
    # print(evaldataset)

    eval_loader = DataLoader(
        evaldataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        persistent_workers=True,
        num_workers=NUM_WORKERS_EMB,
        pin_memory=True,
        prefetch_factor=4,
    )



    

    # print(eval_loader)

    if RANK == 0:
        print(f"Loaded {len(evaldataset)} sequences from {CACHE_PATH}")

    return eval_loader


# -------------------------
# 3. Dataset & embedding
# -------------------------

# PLACEHOLDER: The ESMDataset class is imported

# -------------------------
# 4. Predicter
# -------------------------


def predict(modelpath, loader, firstrun=False):
    model = torch.load(modelpath, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs for prediction")
    #     model = nn.DataParallel(model)
    model.eval()



    # print(model)
    # print("Model loaded successfully.\n")

    # sd = model.state_dict()

    # first_key = next(iter(sd.keys()))
    # print("First key:", first_key)
    # print("First-weight tensor:\n", sd[first_key])

    all_predictions = []
    all_predictions_raw = []
    all_accuracies = []

    all_precisions_per_class = {i: [] for i in range(1, NUM_CLASSES)}
    all_recalls_per_class = {i: [] for i in range(1, NUM_CLASSES)}


    with torch.no_grad():
        for batch in loader:
            predictions = []
            predictions_raw = []
            inputs, labels, starts, ends = batch
            # if len(predictions) == 0:
                # print("\nTENSORS IN PREDICT():",inputs[0])
            inputs = inputs.to(DEVICE)
            output = model(inputs)
            # if len(predictions) == 0:   
                # print("OUTPUT:",output)

            # probs = torch.softmax(output, dim=1)
            # print(output[0])
            preds_raw, preds = torch.max(
                output, dim=1
            )  #### SWITCH TO PROBABILITIES MAYBE? ####
            # print("PREDS:",preds)

            predictions.extend(preds.cpu().numpy())
            predictions_raw.extend(preds_raw.cpu().numpy())


            true_labels = labels

    # print("TRUE:",true_labels[:30],print(len(true_labels)))
    # print("PRED:",predictions[:30],print(len(predictions)))
    # print("PRED RAW:",predictions_raw[:30],print(len(predictions_raw)))

            if firstrun is True:
                # Initialize TensorBoard writer

                os.makedirs(TENSBORBOARD_LOG_DIR, exist_ok=True)
                writer = SummaryWriter(TENSBORBOARD_LOG_DIR)

                # Overall metrics
                accuracy = accuracy_score(true_labels, predictions)
                weighted_precision = precision_score(
                    true_labels, predictions, average="weighted"
                )
                # confusion = confusion_matrix(true_labels, predictions, normalize="true")

                # Class‐specific precision & recall
                prec_per_class = precision_score(
                    true_labels, predictions, labels=list(range(1, NUM_CLASSES)), average=None
                )
                rec_per_class = recall_score(
                    true_labels, predictions, labels=list(range(1, NUM_CLASSES)), average=None
                )

                # Log overall metrics to TensorBoard
                writer.add_scalar("Metrics/Accuracy", accuracy, 0)
                writer.add_scalar("Metrics/Weighted_Precision", weighted_precision, 0)
                writer.add_scalar("Metrics/AvgPrecision", prec_per_class.mean(), 0)
                writer.add_scalar("Metrics/AvgRecall", rec_per_class.mean(), 0)


                prec_tensor = torch.tensor(prec_per_class)
                writer.add_histogram("Precision Distribution", prec_tensor, 0, bins=100)

                rec_tensor = torch.tensor(rec_per_class)
                writer.add_histogram("Recall Distribution", rec_tensor, 0, bins=100)

                for i in range(1, NUM_CLASSES):
                    writer.add_scalar(f"Precision/Class_{i}", prec_per_class[i - 1], 0)
                    writer.add_scalar(f"Recall/Class_{i}", rec_per_class[i - 1], 0)

                # if dist.get_rank() == 0:
                #     # Print results
                #     print(f"Accuracy: {accuracy:.4f}")
                #     # print(f"Weighted precision (all classes): {weighted_precision:.4f}")

                #     print("\nPrecision Metrics:")
                #     for i in range(1, NUM_CLASSES):
                #         print(f"Precision for class {i}: {prec_per_class[i - 1]:.4f}")

                #     print("\nRecall Metrics:")
                #     for i in range(1, NUM_CLASSES):
                #         print(f"Recall    for class {i}: {rec_per_class[i - 1]:.4f}")

                #     print("TRUE:",true_labels[0:100])
                #     print("PRED:",predictions[0:100])
                    # print(predictions_raw[0:5])

                    # print(f"Confusion Matrix (rows=true, cols=pred):\n{confusion}\n")


                all_predictions.extend(predictions)
                all_predictions_raw.extend(predictions_raw)
                all_accuracies.append(accuracy)
                all_accuracies.append(accuracy)

                for i in range(1, NUM_CLASSES):
                    all_precisions_per_class[i].append(prec_per_class[i - 1])
                    all_recalls_per_class[i].append(rec_per_class[i - 1])









                # Close the writer
                writer.close()
                # print(f"Metrics logged to TensorBoard in '{TENSBORBOARD_LOG_DIR}'")

                # print("raw", predictions_raw[:30])
                # print("maxed", predictions[:30])

            # print("Prediction run done.\n")

            # print(predictions_raw[0])



    if RANK == 0:
        print("Mean accuracy over all batches:", np.mean(all_accuracies))
        print("Mean accuracy over all batches:", np.mean(all_accuracies))
        for i in range(1, NUM_CLASSES):
            print(
                f"Mean precision for class {i}: {np.mean(all_precisions_per_class[i]):.4f}, Mean recall: {np.mean(all_recalls_per_class[i]):.4f}"
            )
    return all_predictions, all_predictions_raw


# -------------------------
# 5. Cutter / Re-embedder
# -------------------------


def cut_inputs_embedding(predictions, true_labels, df, cut_size, cut_front=True):
    positive_embeddings = []
    positive_labels = []

    for i in range(len(predictions)):
        if predictions[i] != 0:
            if cut_front is True:
                temp_seq = df[SEQUENCE_COL][i][cut_size:]
                # if i < 100:
                # print(temp_seq)

                positive_embeddings.append(temp_seq)
                positive_labels.append(true_labels[i].item())

            else:
                temp_seq = df[SEQUENCE_COL][i][:-cut_size]

                positive_embeddings.append(temp_seq)
                positive_labels.append(true_labels[i].item())

    # print("Len first seq", len(positive_embeddings[0]))
    # print("First seq", positive_embeddings[0])

    df = pd.DataFrame(
        {
            SEQUENCE_COL: positive_embeddings,
            "label": positive_labels,
        }
    )

    # print(df.head())

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

    df_embeddings = esm_data.embeddings
    df_labels = esm_data.labels
    df = esm_data.df

    df_embeddings = TensorDataset(df_embeddings, df_labels)

    # print("Dataset building complete")

    df_embeddings = DataLoader(
        df_embeddings,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return df_embeddings, df_labels


# -------------------------
# 6. Drop checker
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
    if len(logits_so_far) < threshold + 1:
        return False

    consecutive_drops = 0

    for i in range(1, len(logits_so_far)):
        if logits_so_far[i] <= logits_so_far[i - 1]:
            consecutive_drops += 1
            if consecutive_drops >= threshold:
                return True
        else:
            consecutive_drops = 0

    return False


# -------------------------
# 7. Boundary gatherer
# -------------------------
def boundary_gatherer(predictions, loader):
    """
    Gather the boundaries, true classes, true pos of predicted domains from the predictions list.

    Args:
        predictions: List of predicted domain starts and ends.
        loader: DataLoader with sequence data

    Returns:
        pos_indices, pos_true_start, pos_true_end, true_class
    """
    
    # Initialize lists to collect results across all batches
    pos_indices = []
    pos_true_start = []
    pos_true_end = []
    true_class = []
    
    for batch_idx, batch in enumerate(loader):
        inputs, labels, df_starts, df_ends = batch
        
        # Get predictions for this batch
        batch_start_idx = batch_idx * BATCH_SIZE
        batch_end_idx = min(batch_start_idx + inputs.size(0), len(predictions))
        batch_predictions = predictions[batch_start_idx:batch_end_idx]
        
        # Find positive indices within this batch (local indices)
        batch_pos_indices = [i for i, p in enumerate(batch_predictions) if p != 0]
        
        # Convert to global indices
        global_pos_indices = [batch_start_idx + i for i in batch_pos_indices]
        
        # Extend the global pos_indices list
        pos_indices.extend(global_pos_indices)
        
        # Get starts, ends, and labels for positive predictions in this batch
        if batch_pos_indices:  # Only process if there are positive predictions
            batch_pos_true_start = [df_starts[i].item() for i in batch_pos_indices]
            batch_pos_true_end = [df_ends[i].item() for i in batch_pos_indices]
            batch_true_class = [labels[i].item() for i in batch_pos_indices]
            
            # Extend the global lists
            pos_true_start.extend(batch_pos_true_start)
            pos_true_end.extend(batch_pos_true_end)
            true_class.extend(batch_true_class)
    
    if not pos_indices:
        if RANK == 0:
            print("No positives found.")
        return None, None, None, None
    
    if RANK == 0:
        print(f"Total positives found: {len(pos_indices)}")
        print(f"pos_indices sample: {pos_indices[:100000]}...")  # Show first 10
        print(f"max pos_index: {max(pos_indices)}")
        
    return pos_indices, pos_true_start, pos_true_end, true_class
# -------------------------
# 8. Cutter-Loop
# -------------------------


def cutter_loop(
    predictions, true_labels, df, pos_indices, pos_true, raw_scores_nocut, name="start"
):
    # Dynamic cut sizes based on sequence lengths
    seq_lengths = [len(seq) for seq in df[SEQUENCE_COL]]
    max_len = max(seq_lengths)
    mean_len = np.mean(seq_lengths)
    min_len = min(seq_lengths)
    cut_sizes = list(range(5, min(int(mean_len * 0.8), max_len // 4), 5))
    cut_sizes = list(
        range(5, min(int(mean_len * 0.8), max_len // 4), 5)
        if max(cut_sizes) < min_len
        else range(5, int(max_len / 3), 5)
    )
    print(f"\nMax cut size: {max(cut_sizes)} residues")

    Npos, K = len(pos_indices), len(cut_sizes)
    # Initialize matrices and tracking variables
    raw_matrix = np.zeros((Npos, K), dtype=float)
    for row_idx in range(Npos):
        raw_matrix[row_idx, 0] = raw_scores_nocut[pos_indices[row_idx]]
    max_domains = 1  # Maximum number of domains to track
    sequence_logits = [
        [[] for _ in range(max_domains)] for _ in range(Npos)
    ]  # [sequence][domain][logits]
    # Initialize the raw_scores_nocut into the first column of the matrix
    for row_idx in range(Npos):
        sequence_logits[row_idx][0].append(raw_scores_nocut[pos_indices[row_idx]])
    sequence_domain_status = [
        0 for _ in range(Npos)
    ]  # Track which domain each sequence is currently on
    stopped_sequences = set()

    for j, cut in enumerate(cut_sizes):
        print(f"Processing cut size {cut} ({j + 1}/{K})...")

        active_sequences = [
            pos_indices[row_idx]
            for row_idx in range(Npos)
            if row_idx not in stopped_sequences
        ]

        if not active_sequences:
            print("No active sequences left to process.")
            break

        filtered_predictions = [predictions[i] for i in active_sequences]
        filtered_true_labels = [true_labels[i] for i in active_sequences]
        filtered_df = df.iloc[active_sequences].reset_index(drop=True)

        decrease_percentage = ((Npos - len(filtered_predictions)) / Npos) * 100
        print(
            f"Filtered predictions: {len(filtered_predictions)} / {Npos} ({decrease_percentage:.1f}% excluded)"
        )

        # Decide whether to cut front or back
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
        _, _, raw_scores = predict(
            MODEL_PATH, sub_loader, sub_labels, firstrun=False
        )

        # Update logits and check for stopping sequences
        active_idx = 0
        for row_idx in range(Npos):
            if row_idx not in stopped_sequences:
                current_domain = sequence_domain_status[row_idx]
                raw_matrix[row_idx, j] = raw_scores[active_idx]
                sequence_logits[row_idx][current_domain].append(raw_scores[active_idx])

                # Check if this sequence should stop further prediction
                if check_dropping_logits_across_cuts(
                    sequence_logits[row_idx][current_domain], threshold=THRESHOLD
                ):
                    current_domain += 1
                    sequence_domain_status[row_idx] = current_domain

                    if current_domain < max_domains:
                        # Check if previous domain has logits and if current score is better
                        prev_domain_logits = sequence_logits[row_idx][
                            current_domain - 1
                        ]
                        if raw_scores[active_idx] > max(prev_domain_logits):
                            sequence_logits[row_idx][current_domain].append(
                                raw_scores[active_idx]
                            )
                    else:
                        stopped_sequences.add(row_idx)

                active_idx += 1

        print(f"Sample sequence logits: {sequence_logits[0]}")

    print(
        f"Stopped early prediction for {len(stopped_sequences)} out of {Npos} sequences"
    )

    # Process results
    all_residues = []
    errors = []
    for row_idx in range(Npos):
        curve = raw_matrix[row_idx]
        all_j = []
        residues = []

        # Find the best cut among the cuts that were actually computed for this sequence
        for domain_idx in range(max_domains):
            if sequence_logits[row_idx][domain_idx]:  # If this domain has logits
                # For stopped sequences, only consider cuts up to where they were stopped
                if row_idx in stopped_sequences:
                    non_zero_indices = np.nonzero(curve)[0]
                    if len(non_zero_indices) > 0:
                        best_j = non_zero_indices[np.argmax(curve[non_zero_indices])]
                    else:
                        best_j = 0  # Fallback to first cut
                else:
                    # For sequences that completed all cuts
                    best_j = int(np.argmax(curve))
                all_j.append(best_j)

        for j in all_j:
            residues.append(cut_sizes[j])

        pred = residues
        true = pos_true[row_idx]

        if not true:
            err = None
        else:
            all_errors = []
            for predicted_domain in pred:
                domain_error = min(abs(predicted_domain - ts) for ts in true)
                all_errors.append(domain_error)
            err = min(all_errors)

        errors.append(err)
        all_residues.append(residues)

    errors_wo_nones = [e for e in errors if e is not None]
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
# 9. Helper Functions
# -------------------------


def list_creater(df, pos_indices, all_residues_start, all_residues_end, predictions):
    """
    Create a list of dictionaries for each sequence with its details.
    """
    endlist = []

    for i, seq_idx in enumerate(pos_indices):
        entry = {
            "ID": df["ID"][seq_idx],
            "Class": predictions[seq_idx],
            "Domain_Start": all_residues_start[i],
            "Domain_End": all_residues_end[i],
        }
        endlist.append(entry)

    print("\n\n\nEndlist type:", type(endlist))
    print(
        "Endlist length:", len(endlist) if isinstance(endlist, list) else "Not a list"
    )
    print("First few items:", endlist[:3] if isinstance(endlist, list) else endlist)

    return endlist


def list_concatenater(endlist):
    # Sort by ID first to group same IDs together
    endlist.sort(key=lambda x: x["ID"])

    concatenated_list = []
    i = 0

    while i < len(endlist):
        current_row = endlist[i].copy()

        # Check if next rows have the same ID and concatenate them
        j = i + 1
        while j < len(endlist) and endlist[j]["ID"] == current_row["ID"]:
            # Concatenate Domain_Start and Domain_End lists
            if isinstance(current_row["Domain_Start"], list):
                current_row["Domain_Start"].extend([endlist[j]["Domain_Start"]])
            else:
                current_row["Domain_Start"] = [
                    current_row["Domain_Start"],
                    endlist[j]["Domain_Start"],
                ]

            if isinstance(current_row["Domain_End"], list):
                current_row["Domain_End"].extend([endlist[j]["Domain_End"]])
            else:
                current_row["Domain_End"] = [
                    current_row["Domain_End"],
                    endlist[j]["Domain_End"],
                ]

            j += 1

        concatenated_list.append(current_row)
        i = j

    # print("\n\n\nConcatenated Endlist:", endlist)

    concatenated_df = pd.DataFrame(concatenated_list)
    concatenated_df.to_csv("./Results/Predicter_from_ESM_final_result.csv", index=False)
    print("Saved results to predicted_boundaries_after.csv")


# -------------------------
# 10. Plotter Functions
# -------------------------


def plotter(
    errors,
    Name,
    bin_width=5,
):
    errors = [e for e in errors if e is not None]  # filter out None errors
    max_err = int(max(errors))
    bins = np.arange(0, max_err + bin_width, bin_width)

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
    plt.savefig(
        f"./Evalresults/ESM/histogram_{Name}_errors.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def boxplotter(errors, classes, Name):
    # Filter out None values from errors and their corresponding classes
    filtered_data = [(e, c) for e, c in zip(errors, classes) if e is not None]
    filtered_errors, filtered_classes = (
        zip(*filtered_data) if filtered_data else ([], [])
    )

    # Dynamically group errors by class
    class_dict = {}
    for err, cls in zip(filtered_errors, filtered_classes):
        class_dict.setdefault(cls, []).append(err)

    # Sort classes and build data+labels
    keys = sorted(class_dict)
    data = [class_dict[k] for k in keys]
    labels = [f"Class {k}" for k in keys]

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
    plt.savefig(f"./Evalresults/ESM/boxplot_{Name}.png", dpi=300, bbox_inches="tight")
    plt.show()


# -------------------------
# 11. Main
# -------------------------


def main():
    ### Load embeddings, labels abd predicting ###
    eval_loader= opener()
    print('\n')
    # if dist.get_rank() == 0:
    #     first_batch = next(iter(df_embeddings))
        # print(f"First batch shape: {first_batch[0].shape}, First label: {df_labels[0]}, DataFrame head:")
        # print(df.head())
        # print(len(df_labels),len(df_starts),len(df_ends))


    

    predictions, raw_scores_nocut = predict(
        MODEL_PATH, eval_loader, firstrun=True
    )




    # temp quit
    dist.barrier()  # Ensure all processes reach this point before quitting
    if RANK == 0:
        print("Exiting after first run for debugging purposes.")
    dist.destroy_process_group()  # Clean up the process group
    quit()
    df = None





    pos_indices, pos_true_start, pos_true_end, true_class = boundary_gatherer(
        predictions, eval_loader
    )

    ### Cut loop start ###
    all_residues_start, errors_start = cutter_loop(
        predictions,
        true_class,
        df,
        pos_indices,
        pos_true_start,
        raw_scores_nocut,
        name="start",
    )

    all_residues_end, errors_end = cutter_loop(
        predictions,
        true_class,
        df,
        pos_indices,
        pos_true_end,
        raw_scores_nocut,
        name="end",
    )

    ### List Creater & Plotter ###
    endlist = list_creater(
        df, pos_indices, all_residues_start, all_residues_end, predictions
    )

    list_concatenater(endlist)

    plotter(errors_start, Name="Start", bin_width=5)
    plotter(errors_end, Name="End", bin_width=5)

    boxplotter(errors_start, true_class, Name="Start")
    boxplotter(errors_end, true_class, Name="End")

    boxplotter(errors_end, true_class, Name="End")


###############################################################################################################
if __name__ == "__main__":
    main()
