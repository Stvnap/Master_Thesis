import math
import time
import os
import pickle
import torch.nn as nn
import pytorch_lightning as pl
import esm
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from ESM_Embeddings_HP_search import LitClassifier, FFNClassifier
from torch.nn.parallel import DistributedDataParallel as DDP
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

# -------------------------
# 1. GLobal settings
# -------------------------

CSV_PATH = "./Dataframes/Evalsets/DataEvalSwissPro_esm_10d_150w_shuffled.csv"
CATEGORY_COL = "categories"
SEQUENCE_COL = "Sequences"
MODEL_PATH = "./models/Optuna_10d_uncut_10000.pt"
CACHE_PATH = "./pickle/Predicer_embeddings_10d_150w_shuffled.pkl"
TENSBORBOARD_LOG_DIR = "./models/10d_uncut_logs"


NUM_CLASSES = 11
BATCH_SIZE = 128
EMB_BATCH = 16
NUM_WORKERS = max(16, os.cpu_count())
NUM_WORKERS_EMB = max(16, os.cpu_count())

EPOCHS = 500

THRESHOLD = 2  # Number of consecutive drops to trigger exclusion
print("Treshold:", THRESHOLD)

# DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# 2. Dataset & embedding
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]

class ESMDataset:
    def __init__(self, skip_df=None,FSDP_used=False):
        def map_label(cat):
            # Adaptive mapping based on NUM_CLASSES
            # Classes 0-9 get mapped to 1-10
            # Classes 10 and 11 get mapped to 0
            if cat <= NUM_CLASSES - 2:
                return cat + 1
            else:
                return 0

        if skip_df is None:
            df = pd.read_csv(CSV_PATH)  # Load only the first 10,000 rows for testing
            df["label"] = df[CATEGORY_COL].apply(map_label)
            df.drop(columns=[CATEGORY_COL], inplace=True)

            self.df = df
            print("Data loaded")
        else:
            self.df = skip_df
            
        self.FSDP_used = FSDP_used

        sequences = self.df[SEQUENCE_COL].tolist()

        # Model initialization
        self.model, self.batch_converter = self.esm_loader()


        start_time = time.time()
        self.embeddings = self._embed(sequences)
        end_time = time.time()
        embedding_time = end_time - start_time
        self.labels = torch.tensor(self.df["label"].values, dtype=torch.long)

        # Print timing information
        print("--------------------------")
        print("Embedding generation completed!")
        print(
            f"Total time: {embedding_time:.2f} seconds ({embedding_time / 60:.2f} minutes)"
        )
        print(
            f"Time per sequence: {embedding_time / len(sequences):.4f} seconds | Sequences per second: {len(sequences) / embedding_time:.2f}"
        )
        print("--------------------------")



    def esm_loader(self):
        # init the distributed world with world_size 1
        if not torch.distributed.is_initialized():        
            url = "tcp://localhost:23457"
            torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

        # download model data from the hub
        model_name = "esm2_t33_650M_UR50D"
        model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)


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
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
            model.eval()  # disables dropout for deterministic results
            model=model.cuda()  # Move model to GPU if available
            model=model.half()


        return model, batch_converter
   

    def _embed(self, seqs):
        """
        Generate embeddings for a list of sequences using ESM model.
        """
        all_embeddings = []
        seqs = list(seqs)  # Ensure it's a list

        total_batches = math.ceil(len(seqs) / EMB_BATCH)
        start_time = time.time()
        print(f"Starting embedding generation for {len(seqs)} sequences in {total_batches} batches...")

        for batch_idx, batch_indices in enumerate(DataLoader(
            range(len(seqs)),
            batch_size=EMB_BATCH,
            shuffle=False,
            num_workers=NUM_WORKERS_EMB, 
            pin_memory=False,  
        )):
            
            try:
                batch = [(f"seq{i}", seqs[i]) for i in batch_indices]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
                batch_tokens = batch_tokens.cuda()
                
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
                    embeddings = results["representations"][33].float()

                    # Pool over sequence dimension (dim=1), ignoring padding (token > 1)
                    mask = (batch_tokens != 1)  # 1 is usually the padding index for ESM
                    mask = mask[:, 1:-1]  # Remove BOS/EOS if present
                    embeddings = embeddings[:, 1:-1, :]  # Remove BOS/EOS if present

                    lengths = mask.sum(dim=1, keepdim=True)
                    pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / lengths
                    pooled = pooled.cpu()  # Move to CPU

                    all_embeddings.append(pooled)

                # Calculate and print ETA every 5 batches or at the end
                if (batch_idx + 1) % 1 == 0 or batch_idx + 1 == total_batches:
                    elapsed_time = time.time() - start_time
                    avg_time_per_batch = elapsed_time / (batch_idx + 1)
                    remaining_batches = total_batches - (batch_idx + 1)
                    eta_seconds = remaining_batches * avg_time_per_batch
                    
                    # Format ETA
                    eta_hours = int(eta_seconds // 3600)                    
                    eta_minutes = int((eta_seconds % 3600) // 60)
                    eta_seconds_remainder = int(eta_seconds % 60)
                    
                    print(f"Batch {batch_idx + 1}/{total_batches} | "
                        f"Processed {(batch_idx + 1) * EMB_BATCH}/{len(seqs)} sequences | "
                        f"ETA: {eta_hours}h {eta_minutes}m {eta_seconds_remainder}s",end='\r', flush=True)
                    


            except Exception as e:
                print(f"\nError processing batch {batch_idx + 1}: {e},processing current batch sequences individually.")
                for i in batch_indices:
                    try:
                        single_seq = [(f"seq{i}", seqs[i])]
                        single_labels, single_strs, single_tokens = self.batch_converter(single_seq)
                        single_tokens = single_tokens.cuda()

                        with torch.no_grad():
                            results = self.model(single_tokens, repr_layers=[33], return_contacts=True)
                            embedding = results["representations"][33].float()
                            
                            # Pool over sequence dimension (dim=1), ignoring padding (token > 1)
                            mask = (single_tokens != 1)  # 1 is usually the padding index for ESM
                            mask = mask[:, 1:-1]  # Remove BOS/EOS if present
                            embedding = embedding[:, 1:-1, :]  # Remove BOS/EOS if present

                            length = mask.sum(dim=1, keepdim=True)
                            pooled = (embedding * mask.unsqueeze(-1)).sum(dim=1) / length
                            pooled = pooled.cpu()  # Move to CPU

                            all_embeddings.append(pooled)

                    except Exception as single_e:
                        print(f"Error processing sequence {i}: {single_e}")
                        # Add a zero embedding as placeholder to maintain tensor dimensions
                        zero_embedding = torch.zeros(1, 1280)  # ESM2-650M embedding dimension
                        all_embeddings.append(zero_embedding)
                        print(f"Added zero embedding for failed sequence {i}")



        return torch.cat(all_embeddings, dim=0)



# -------------------------
# 3. Predicter
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
            # print("\nTENSORS IN PREDICT():",inputs[0:10])
            inputs = inputs.to(DEVICE)
            output = model(inputs)
            # print("OUTPUT:",output)

            # probs = torch.softmax(output, dim=1)
            preds_raw, preds = torch.max(
                output, dim=1
            )  #### SWITCH TO PROBABILITIES MAYBE? ####
            # print("PREDS:",preds)

            predictions.extend(preds.cpu().numpy())
            predictions_raw.extend(preds_raw.cpu().numpy())

    true_labels = df_labels

    # print("TRUE:",true_labels[:30],print(len(true_labels)))
    # print("PRED:",predictions[:30],print(len(predictions)))

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
        writer.add_scalar('Metrics/Accuracy', accuracy, 0)
        writer.add_scalar('Metrics/Weighted_Precision', weighted_precision, 0)
        
        # Log per-class metrics to TensorBoard
        for i, (prec, rec) in enumerate(zip(prec_per_class, rec_per_class)):
            class_id = i + 1  # Classes 1 to NUM_CLASSES-1
            writer.add_scalar(f'Precision/Class_{class_id}', prec, 0)
            writer.add_scalar(f'Recall/Class_{class_id}', rec, 0)

        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        # print(f"Weighted precision (all classes): {weighted_precision:.4f}")


        print("\nPrecision Metrics:")
        for i in range(1, NUM_CLASSES):
            print(f"Precision for class {i}: {prec_per_class[i-1]:.4f}")
            
        print('\nRecall Metrics:')
        for i in range(1, NUM_CLASSES):

            print(f"Recall    for class {i}: {rec_per_class[i-1]:.4f}")

        # print(f"Confusion Matrix (rows=true, cols=pred):\n{confusion}\n")

        # Close the writer
        writer.close()
        # print(f"Metrics logged to TensorBoard in '{TENSBORBOARD_LOG_DIR}'")

        # print("raw", predictions_raw[:30])
        # print("maxed", predictions[:30])

    print("Prediction run done.\n")

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
                
                temp_seq= df[SEQUENCE_COL][i][cut_size:]
                # if i < 100:
                    # print(temp_seq)


                positive_embeddings.append(temp_seq)
                positive_labels.append(true_labels[i].item())

            else:
                temp_seq=df[SEQUENCE_COL][i][:-cut_size]


                positive_embeddings.append(temp_seq)
                positive_labels.append(true_labels[i].item())

    # print("Len first seq", len(positive_embeddings[0]))

    df = pd.DataFrame(
        {
            SEQUENCE_COL: positive_embeddings,
            "label": positive_labels,
        }
    )

    # print(df.head())

    esm_data = ESMDataset(skip_df=df)

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
        # drop_last=False,
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
        if logits_so_far[i] < logits_so_far[i-1]:
            consecutive_drops += 1
            if consecutive_drops >= threshold:
                return True
        else:
            consecutive_drops = 0

    return False


# -------------------------
# 7. Main
# -------------------------

def main():

    os.makedirs("pickle", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            df_embeddings, df_labels, df = pickle.load(f)
        print("Loaded cached embeddings & labels from disk.")

    else:
        esm_data = ESMDataset()
        df_embeddings = esm_data.embeddings
        df_labels = esm_data.labels
        df = esm_data.df

        df_embeddings = TensorDataset(df_embeddings, df_labels)

        df_embeddings = DataLoader(
            df_embeddings,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False, 
        )

        with open(CACHE_PATH, "wb") as f:
            pickle.dump(
                (df_embeddings, df_labels, df),
                f,
            )
        print("Computed embeddings & labels, then wrote them to cache.")

    # print(df["label"][0:10])

    predictions, true_labels, _ = predict(
        MODEL_PATH, df_embeddings, df_labels, firstrun=True
    )
    pos_indices = [i for i, p in enumerate(predictions) if p != 0]
    if not pos_indices:
        print("No positives found.")
        return

    true_start = df["Boundaries"].to_list()
    pos_true_start = [true_start[i] for i in pos_indices]
    # print(pos_true_start)

    boundary_intervals = []

    for b in pos_true_start:
        if isinstance(b, str):
            segs = [seg.strip() for seg in b.split(",") if seg.strip()]
            row_pairs = []
            for seg in segs:
                if "-" in seg:
                    s_str, e_str = seg.split("-", 1)
                    s, e = int(s_str), int(e_str)
                else:
                    s = e = int(seg)
                row_pairs.append((s, e))
            boundary_intervals.append(row_pairs)

        elif isinstance(b, float) and math.isnan(b):
            # no boundary at all
            boundary_intervals.append([])

        else:
            raise ValueError(f"Unexpected type for boundary: {b!r}")

    starts_nested = [[s for s, e in row] for row in boundary_intervals]
    ends_nested = [[e for s, e in row] for row in boundary_intervals]

    # print(boundary_intervals)
    # print(starts_nested)
    # print(ends_nested)

    true_class = df["label"].to_list()

    true_class = [true_class[i] for i in pos_indices]

    # print("\nLengths:")
    # print(len(starts_nested))
    # print(len(ends_nested))
    # print(len(true_class))
    # print(true_class[0], starts_nested[0], ends_nested[0])
    # print("\n")

    pos_true_start = starts_nested
    pos_true_end = ends_nested
    # print(pos_true_start)

    # Dynamic cut sizes based on sequence lengths
    seq_lengths = [len(seq) for seq in df[SEQUENCE_COL]]
    max_len = max(seq_lengths)
    mean_len = np.mean(seq_lengths)
    min_len = min(seq_lengths)
    cut_sizes = [0] + list(range(5, min(int(mean_len * 0.8), max_len // 4), 5))
    cut_sizes = [0] + list(
        range(5, min(int(mean_len * 0.8), max_len // 4), 5) 
        if max(cut_sizes) < min_len 
        else range(5, max_len, 5)
    )
    
    print(f"Max cut size: {max(cut_sizes)} residues\n")

    Npos, K = len(pos_indices), len(cut_sizes)

    # cut front
    raw_matrix = np.zeros((Npos, K), dtype=float)
    max_domains = 10  # Maximum number of domains to track
    sequence_logits = [[[] for _ in range(max_domains)] for _ in range(Npos)]  # [sequence][domain][logits]
    sequence_domain_status = [0 for _ in range(Npos)]  # Track which domain each sequence is currently on
    stopped_sequences_start=set()

    for j, cut in enumerate(cut_sizes):
        print(f"Processing cut size {cut} ({j + 1}/{K})...")
        sub_loader, sub_labels = cut_inputs_embedding(
            predictions, true_labels, df, cut_size=cut, cut_front=True
        )
        _, _, raw_scores = predict(MODEL_PATH, sub_loader, sub_labels, firstrun=False)

        # Update logits and check for stopping sequences
        for row_idx in range(Npos):
            if row_idx not in stopped_sequences_start:
                # print(len(raw_matrix),len(raw_scores))
                # print(row_idx)
                current_domain=sequence_domain_status[row_idx]
                raw_matrix[row_idx, j] = raw_scores[row_idx]
                sequence_logits[row_idx][current_domain].append(raw_scores[row_idx])


                # Check if this sequence should stop further prediction
                if check_dropping_logits_across_cuts(
                    sequence_logits[row_idx][current_domain], threshold=THRESHOLD
                ):
                    current_domain += 1
                    sequence_domain_status[row_idx] = current_domain

                    if current_domain < max_domains:
                        sequence_logits[row_idx][current_domain].append(raw_scores[row_idx])

                    else:
                        stopped_sequences_start.add(row_idx)

        if j < 100:
            print(sequence_logits[0]) 




                    # print(
                    #     f"Stopping further prediction for sequence {seq_idx} after cut {j} due to {THRESHOLD} consecutive logit drops"
                    # )
            # If sequence is stopped, we don't update its logits for this cut size
            # but the raw_matrix retains the previous values (zeros for uncomputed cuts)

    # print(
    #     f"Stopped early prediction for {len(stopped_sequences)} out of {Npos} sequences"
    # )




    all_start_residues = []
    errors_start = []
    for row_idx in range(Npos):  # Use all sequences, not just valid ones
        seq_idx = pos_indices[row_idx]
        curve = raw_matrix[row_idx]
        all_j = []
        start_residues=[]
        # Find the best cut among the cuts that were actually computed for this sequence
        for domain_idx in range(max_domains):
            if sequence_logits[row_idx][domain_idx]:  # If this domain has logits
                # For stopped sequences, only consider cuts up to where they were stopped
                if row_idx in stopped_sequences_start:
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
            start_residues.append(cut_sizes[j])

        pred_start = start_residues
        true_start = pos_true_start[row_idx]

        if not true_start:
            err = None
        else:
            all_errors = []
            for predicted_domain in pred_start:
                domain_error = min(abs(predicted_domain - ts) for ts in true_start)
                all_errors.append(domain_error)
            err = min(all_errors)

        errors_start.append(err)

        status = "stopped early" if row_idx in stopped_sequences_start else "completed"
        # print(f"Sequence #{seq_idx} ({status}) → domain starts at residue {start_residue}")
        # print(f"Seq#{seq_idx}: predicted={pred_start}, true={true_start}, error={err}", end='\r', flush=True)
        all_start_residues.append(start_residues)

    # print("\nAll domain‐start predictions done.")

    errors_start_wo_nones = [
        e for e in errors_start if e is not None
    ]  # filter out None errors

    print(
        f"Mean absolute error over {len(errors_start)} positives: {np.mean(errors_start_wo_nones):.1f} residues"
    )
    print(f"Median absolute error: {np.median(errors_start_wo_nones):.1f} residues")

    print(
        "#####################################################################################################"
    )
    print(
        "----------------------------------------- CUT FRONT DONE --------------------------------------------"
    )
    print(
        "#####################################################################################################"
    )

    raw_matrix = np.zeros((Npos, K), dtype=float)
    max_domains = 10  # Maximum number of domains to track
    sequence_logits = [[[] for _ in range(max_domains)] for _ in range(Npos)]  # [sequence][domain][logits]
    sequence_domain_status = [0 for _ in range(Npos)]  # Track which domain each sequence is currently on
    stopped_sequences_end=set()

    for j, cut in enumerate(cut_sizes):
        print(f"Processing cut size {cut} ({j + 1}/{K})...")
        sub_loader, sub_labels = cut_inputs_embedding(
            predictions, true_labels, df, cut_size=cut, cut_front=True
        )
        _, _, raw_scores = predict(MODEL_PATH, sub_loader, sub_labels, firstrun=False)

        # Update logits and check for stopping sequences
        for row_idx in range(Npos):
            if row_idx not in stopped_sequences_end:
                # print(len(raw_matrix),len(raw_scores))
                # print(row_idx)
                current_domain=sequence_domain_status[row_idx]
                raw_matrix[row_idx, j] = raw_scores[row_idx]
                sequence_logits[row_idx][current_domain].append(raw_scores[row_idx])


                # Check if this sequence should stop further prediction
                if check_dropping_logits_across_cuts(
                    sequence_logits[row_idx][current_domain], threshold=THRESHOLD
                ):
                    current_domain += 1
                    sequence_domain_status[row_idx] = current_domain

                    if current_domain < max_domains:
                        sequence_logits[row_idx][current_domain].append(raw_scores[row_idx])

                    else:
                        stopped_sequences_end.add(row_idx)

        if j < 100:
            print(sequence_logits[0]) 

    all_end_residues = []
    errors_end = []
    for row_idx in range(Npos):  # Use all sequences, not just valid ones
        seq_idx = pos_indices[row_idx]
        curve = raw_matrix[row_idx]
        all_j = []
        end_residues=[]
        # Find the best cut among the cuts that were actually computed for this sequence
        for domain_idx in range(max_domains):
            if sequence_logits[row_idx][domain_idx]:  # If this domain has logits
                # For stopped sequences, only consider cuts up to where they were stopped
                if row_idx in stopped_sequences_end:
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
            end_residues.append(cut_sizes[j])

        pred_end = end_residues
        true_end = pos_true_end[row_idx]

        if not true_end:
            err = None
        else:
            all_errors = []
            for predicted_domain in pred_end:
                domain_error = min(abs(predicted_domain - ts) for ts in true_end)
                all_errors.append(domain_error)
            err = min(all_errors)

        errors_end.append(err)

        status = "stopped early" if row_idx in stopped_sequences_end else "completed"
        # print(f"Sequence #{seq_idx} ({status}) → domain ends at residue {end_residue}")
        # print(f"Seq#{seq_idx}: predicted={pred_end}, true={true_end}, error={err}", end='\r', flush=True)
        all_end_residues.append(end_residues)

        # print(f"Seq#{seq_idx}: predicted={pred_end}, true={true_ends}, error={err}", end='\r', flush=True)
    
        # print("\nAll domain‐end predictions done.")

    errors_end_wo_nones = [e for e in errors_end if e is not None]

    print(
        f"Mean absolute error over {len(errors_end)} positives: {np.mean(errors_end_wo_nones):.1f} residues"
    )
    print(f"Median absolute error: {np.median(errors_end_wo_nones):.1f} residues")


    print(
        "#####################################################################################################"
    )
    print(
        "----------------------------------------- CUT BACK DONE --------------------------------------------"
    )
    print(
        "#####################################################################################################"
    )
    print("All predictions done.")

    print(all_start_residues[0:10])
    print(all_end_residues[0:10])
    # print(errors_start)
    # print(errors_end)
    # print(true_class)

    def list_creater(
        df, pos_indices, all_start_residues, all_end_residues, predictions
    ):
        """
        Create a list of dictionaries for each sequence with its details.
        """
        endlist = []

        for i, seq_idx in enumerate(pos_indices):
            entry = {
                "ID": df["ID"][seq_idx],
                "Class": predictions[seq_idx],
                "Domain_Start": all_start_residues[i],
                "Domain_End": all_end_residues[i],
            }
            endlist.append(entry)

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

        return pd.DataFrame(concatenated_list)

    endlist = list_creater(
        df, pos_indices, all_start_residues, all_end_residues, predictions
    )
    print("\n\n\nEndlist type:", type(endlist))
    print(
        "Endlist length:", len(endlist) if isinstance(endlist, list) else "Not a list"
    )
    print("First few items:", endlist[:3] if isinstance(endlist, list) else endlist)

    endlist = list_concatenater(endlist)
    print("\n\n\nConcatenated Endlist:", endlist)

    endlist.to_csv("./Results/Predicter_from_ESM_final_result.csv", index=False)
    print(f"Saved results to predicted_boundaries_after.csv")

    return all_start_residues, errors_start, all_end_residues, errors_end, true_class



#####################################################################################################################3



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
    ax.set_ylim(bottom=0,top=100)  # Set y-axis limits

    plt.tight_layout()
    plt.savefig(f"./Evalresults/ESM/boxplot_{Name}.png", dpi=300, bbox_inches="tight")
    plt.show()


###############################################################################################################


if __name__ == "__main__":
    all_start_residues, errors_start, all_end_residue, errors_end, true_class = main()

    plotter(errors_start, Name="Start", bin_width=5)
    plotter(errors_end, Name="End", bin_width=5)

    boxplotter(errors_start, true_class, Name="Start")
    boxplotter(errors_end, true_class, Name="End")

    boxplotter(errors_end, true_class, Name="End")
