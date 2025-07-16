import math
import os
import pickle
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from ESM_Embeddings_HP_search import ESMDataset, LitClassifier, FFNClassifier 
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
MODEL_PATH = "./models/Optuna_100d_uncut_t33.pt"
CACHE_PATH = "./pickle/FoundEntriesSwissProteins_100d_predicter.pkl"
TENSBORBOARD_LOG_DIR = "./models/100d_uncut_logs"

ESM_MODEL = "esm2_t33_650M_UR50D"


NUM_CLASSES = 101
BATCH_SIZE = 128
EMB_BATCH = 1
NUM_WORKERS = 0
NUM_WORKERS_EMB = 0

THRESHOLD = 5  # Number of consecutive drops to trigger exclusion
print("Treshold:", THRESHOLD)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 2. Opener Function
# -------------------------


def opener():
    """
    Opens the ESM model, loads the embeddings and labels from cache or computes them if not cached.
    """
    print("Opening data...")
    
    os.makedirs("pickle", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        # Get file size for progress tracking
        file_size = os.path.getsize(CACHE_PATH)
        print(f"Loading pickle file ({file_size / (1024**3):.2f} GB)...")
        
        with open(CACHE_PATH, "rb") as f:
            # Wrap the file object with tqdm for progress
            with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading pickle") as pbar:
                class ProgressFile:
                    def __init__(self, file_obj, progress_bar):
                        self.file_obj = file_obj
                        self.progress_bar = progress_bar
                        self.bytes_read = 0
                    
                    def read(self, size=-1):
                        data = self.file_obj.read(size)
                        if data:
                            self.bytes_read += len(data)
                            self.progress_bar.update(len(data))
                        return data
                    
                    def __getattr__(self, name):
                        return getattr(self.file_obj, name)
                
                progress_file = ProgressFile(f, pbar)
                df_embeddings, df_labels = pickle.load(progress_file)
        
        print("Loaded cached embeddings & labels from disk.")
        return df_embeddings, df_labels, None, None, None

    else:
        esm_data = ESMDataset(
            esm_model=ESM_MODEL,
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
        df_starts = esm_data.starts
        df_ends = esm_data.ends
        df = esm_data.df

        print(df_embeddings[0][0].shape)
        print(df_labels[0])


        df_embeddings = TensorDataset(df_embeddings, df_labels)

        df_embeddings = DataLoader(
            df_embeddings,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )


        with open(CACHE_PATH, "wb") as f:
            pickle.dump(
                (df_embeddings, df_labels),
                f,
            )

        if dist.get_rank()  == 0:
            print("Computed embeddings & labels, then wrote them to cache.")

        return df_embeddings, df_labels, df_starts, df_ends,df


# -------------------------
# 3. Dataset & embedding
# -------------------------

# PLACEHOLDER: The ESMDataset class is imported

# -------------------------
# 4. Predicter
# -------------------------


def predict(modelpath, loader, df_labels, firstrun=False):
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

    predictions = []
    predictions_raw = []
    with torch.no_grad():
        for batch in loader:
            inputs, _ = batch
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

    true_labels = df_labels

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
        confusion = confusion_matrix(true_labels, predictions, normalize="true")

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

        # Log per-class metrics to TensorBoard
        for i, (prec, rec) in enumerate(zip(prec_per_class, rec_per_class)):
            class_id = i + 1  # Classes 1 to NUM_CLASSES-1
            writer.add_scalar(f"Precision/Class_{class_id}", prec, 0)
            writer.add_scalar(f"Recall/Class_{class_id}", rec, 0)


        if dist.get_rank() == 0:
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            # print(f"Weighted precision (all classes): {weighted_precision:.4f}")

            print("\nPrecision Metrics:")
            for i in range(1, NUM_CLASSES):
                print(f"Precision for class {i}: {prec_per_class[i - 1]:.4f}")

            print("\nRecall Metrics:")
            for i in range(1, NUM_CLASSES):
                print(f"Recall    for class {i}: {rec_per_class[i - 1]:.4f}")

            # print("TRUE:",true_labels[0:5])
            # print("PRED:",predictions[0:5])
            # print(predictions_raw[0:5])

            # print(f"Confusion Matrix (rows=true, cols=pred):\n{confusion}\n")

        # Close the writer
        writer.close()
        # print(f"Metrics logged to TensorBoard in '{TENSBORBOARD_LOG_DIR}'")

        # print("raw", predictions_raw[:30])
        # print("maxed", predictions[:30])

    # print("Prediction run done.\n")

    # print(predictions_raw[0])

    return predictions, true_labels, predictions_raw


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
def boundary_gatherer(predictions, df, df_starts, df_ends):
    """
    Gather the boundaries, true classes, true pos of predicted domains from the predictions list.

    Args:
        predictions: List of predicted domain starts and ends.
        df: DataFrame with sequence data
        df_starts: Tensor/list of domain start positions
        df_ends: Tensor/list of domain end positions

    Returns:
        pos_indices, pos_true_start, pos_true_end, true_class
    """
    # Debug prints to understand the data structure
    print(f"Debug info:")
    print(f"predictions length: {len(predictions)}")
    print(f"df length: {len(df)}")
    print(f"df_starts length: {len(df_starts)}")
    print(f"df_ends length: {len(df_ends)}")
    print(df.head())
    
    pos_indices = [i for i, p in enumerate(predictions) if p != 0]
    if not pos_indices:
        print("No positives found.")
        return None, None, None, None
    
    print(f"pos_indices: {pos_indices[:10]}...")  # Show first 10
    print(f"max pos_index: {max(pos_indices)}")

    # Convert tensors to lists if needed
    if hasattr(df_starts, 'tolist'):
        starts_list = df_starts.tolist()
        ends_list = df_ends.tolist()
    else:
        starts_list = df_starts
        ends_list = df_ends

    # Check if indices are within bounds
    max_valid_index = min(len(starts_list), len(ends_list), len(df)) - 1
    valid_pos_indices = [i for i in pos_indices if i <= max_valid_index]
    
    if len(valid_pos_indices) != len(pos_indices):
        print(f"Warning: {len(pos_indices) - len(valid_pos_indices)} indices out of bounds, filtering them out")
        pos_indices = valid_pos_indices

    # Get starts and ends for positive predictions only
    pos_true_start = [starts_list[i] for i in pos_indices]
    pos_true_end = [ends_list[i] for i in pos_indices]

    # Get true classes for positive predictions - use different variable name
    all_true_classes = df["label"].to_list()
    true_class = [all_true_classes[i] for i in pos_indices]

    print(f"\nLengths:")
    print(f"pos_true_start: {len(pos_true_start)}")
    print(f"pos_true_end: {len(pos_true_end)}")
    print(f"true_class: {len(true_class)}")
    
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
        seq_idx = pos_indices[row_idx]
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
    df_embeddings, df_labels, df_starts, df_ends, df = opener()

    if dist.get_rank() == 0:
        first_batch = next(iter(df_embeddings))
        print(f"First batch shape: {first_batch[0].shape}, First label: {df_labels[0]}, DataFrame head:")
        # print(df.head())
        # print(len(df_labels),len(df_starts),len(df_ends))


    

    predictions, true_labels, raw_scores_nocut = predict(
        MODEL_PATH, df_embeddings, df_labels, firstrun=True
    )

    pos_indices, pos_true_start, pos_true_end, true_class = boundary_gatherer(
        predictions, df, df_starts, df_ends
    )

    ### Cut loop start ###
    all_residues_start, errors_start = cutter_loop(
        predictions,
        true_labels,
        df,
        pos_indices,
        pos_true_start,
        raw_scores_nocut,
        name="start",
    )

    all_residues_end, errors_end = cutter_loop(
        predictions,
        true_labels,
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
