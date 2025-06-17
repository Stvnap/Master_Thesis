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
from ESM_Embeddings_HP_search import LitClassifier, FFNClassifier

# -------------------------
# 1. GLobal settings
# -------------------------

CSV_PATH = "./Dataframes/Evalsets/DataEvalSwissProt2d_esm_150wsize.csv"
CATEGORY_COL = "categories"
SEQUENCE_COL = "Sequences"
MODEL_PATH = "./models/optuna_bestmodel.pt"
CACHE_PATH = "pickle/Predicer_embeddings_test_wsize150.pkl"


NUM_CLASSES = 3
BATCH_SIZE = 16
EMB_BATCH = 16
NUM_WORKERS = 64
NUM_WORKERS_EMB = 8
LR = 1e-5
WEIGHT_DECAY = 1e-2
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
            df = pd.read_csv(
                CSV_PATH,
            )  # Load only the first 100 rows for testing)
            df["label"] = df[CATEGORY_COL].apply(map_label)
            # print(df["label"][0:10])

            df.drop(columns=[CATEGORY_COL], inplace=True)

            self.df = df

            print("Data loaded")
        else:
            self.df = skip_df

        self.model, self.alphabet, self.batch_converter = self.esm_loader()
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


        self.labels = torch.tensor(self.df["label"].values, dtype=torch.long)

        print("Embeddings computed")


    def esm_loader(self):
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

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
                    print(f"Batch {batch_idx + 1}/{total_batches} | "
                        f"Elapsed: {elapsed:.1f}s | ETA: {eta / 60:.1f}min")

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
                                repr_layers=[33],
                                return_contacts=False,
                            )["representations"][33]
                        else:
                            out = self.model(
                                batch_tokens,
                                repr_layers=[33],
                                return_contacts=False,
                            )["representations"][33]

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
                    print(f"OOM in batch {batch_idx}, falling back to single sequence processing")
                    
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
                                        repr_layers=[33],
                                        return_contacts=False,
                                    )["representations"][33]
                                else:
                                    single_out = self.model(
                                        single_tokens,
                                        repr_layers=[33],
                                        return_contacts=False,
                                    )["representations"][33]

                            single_mask = (single_tokens == self.alphabet.padding_idx)
                            single_out_masked = single_out.masked_fill(single_mask.unsqueeze(-1), 0.0)
                            single_lengths = (~single_mask).sum(dim=1, keepdim=True).float()
                            single_pooled = single_out_masked.sum(dim=1) / single_lengths.clamp(min=1)

                            all_outputs.append(single_pooled.half())
                            
                            del single_out, single_out_masked, single_mask, single_lengths, single_tokens
                            
                        except Exception as single_e:
                            print(f"Failed single sequence {seq_idx}: {single_e}")
                            # Only use dummy as absolute last resort for individual sequences
                            dummy = torch.zeros((1, 1280), device=DEVICE, dtype=torch.float16)
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
                                        repr_layers=[33],
                                        return_contacts=False,
                                    )["representations"][33]
                                else:
                                    single_out = self.model(
                                        single_tokens,
                                        repr_layers=[33],
                                        return_contacts=False,
                                    )["representations"][33]

                            single_mask = (single_tokens == self.alphabet.padding_idx)
                            single_out_masked = single_out.masked_fill(single_mask.unsqueeze(-1), 0.0)
                            single_lengths = (~single_mask).sum(dim=1, keepdim=True).float()
                            single_pooled = single_out_masked.sum(dim=1) / single_lengths.clamp(min=1)

                            all_outputs.append(single_pooled.half())
                            
                            del single_out, single_out_masked, single_mask, single_lengths, single_tokens
                            
                        except Exception as single_e:
                            print(f"Failed single sequence {seq_idx}: {single_e}")
                            dummy = torch.zeros((1, 1280), device=DEVICE, dtype=torch.float16)
                            all_outputs.append(dummy)

            # Track batch timing
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)



        # 6) Simple concatenation (no reordering needed)
        if all_outputs:
            embeddings = torch.cat(all_outputs, dim=0).float().cpu()
        else:
            embeddings = torch.zeros((len(seqs), 1280))


        total_embed_time = time.time() - embed_start_time
        print(f"\n=== Embedding Generation Summary ===")
        print(f"Total time: {total_embed_time:.2f}s ({total_embed_time / 60:.2f}min)")
        print(f"Sequences per second: {len(seqs) / total_embed_time:.2f}")
        print("=====================================\n")

        return embeddings


# -------------------------
# 3. Predicter
# -------------------------


def predict(modelpath, loader, df_labels, firstrun=False):
    model = torch.load(modelpath, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
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

    # print(true_labels[:30],print(len(true_labels)))
    # print(predictions[:30],print(len(predictions)))

    if firstrun is True:
        # Overall metrics
        accuracy = accuracy_score(true_labels, predictions)
        weighted_precision = precision_score(
            true_labels, predictions, average="weighted"
        )
        confusion = confusion_matrix(true_labels, predictions, normalize="true")

        # Class‐specific precision & recall for classes 1 and 2
        prec_per_class = precision_score(
            true_labels, predictions, labels=[1, 2], average=None
        )
        rec_per_class = recall_score(
            true_labels, predictions, labels=[1, 2], average=None
        )

        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted precision (all classes): {weighted_precision:.4f}")
        print(f"Precision for class 1: {prec_per_class[0]:.4f}")
        print(f"Recall    for class 1: {rec_per_class[0]:.4f}")
        print(f"Precision for class 2: {prec_per_class[1]:.4f}")
        print(f"Recall    for class 2: {rec_per_class[1]:.4f}\n")
        print(f"Confusion Matrix (rows=true, cols=pred):\n{confusion}\n")

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
                if i == 0:
                    print(
                        "Len pre cut",
                        len((df[SEQUENCE_COL][i])),
                        "seq",
                        df[SEQUENCE_COL][i][cut_size:],
                    )
                positive_embeddings.append((df[SEQUENCE_COL][i])[cut_size:])
                positive_labels.append(true_labels[i].item())

            else:
                if i == 0:
                    print(
                        "Len pre cut",
                        len((df[SEQUENCE_COL][i])),
                        "seq",
                        df[SEQUENCE_COL][i][:-cut_size],
                    )
                positive_embeddings.append((df[SEQUENCE_COL][i])[:-cut_size])
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
        pin_memory=True,
    )

    return df_embeddings, df_labels


# -------------------------
# 6. Main
# -------------------------


def main():
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

        if len(logits_so_far) >= 4:
            window_size = max(2, len(logits_so_far) // 3)  # Dynamic window size
            recent_avg = np.mean(logits_so_far[-window_size:])
            early_avg = np.mean(logits_so_far[:window_size])

            # Stop if recent average is significantly lower than early average
            decline_threshold = 0.1  # Adjust this value based on your data
            if recent_avg < early_avg - decline_threshold:
                return True

        # Alternative: Linear regression slope approach
        if len(logits_so_far) >= 4:
            x = np.arange(len(logits_so_far))
            y = np.array(logits_so_far)

            # Calculate slope using least squares
            n = len(logits_so_far)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
                n * np.sum(x**2) - np.sum(x) ** 2
            )

            # Stop if slope is significantly negative
            slope_threshold = -0.05  # Adjust based on your data scale
            if slope < slope_threshold:
                return True

        return False

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

        print("Dataset building complete")

        df_embeddings = DataLoader(
            df_embeddings,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        print("Data loader created")

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
    cut_sizes = [0] + list(range(5, min(int(mean_len * 0.8), max_len // 4), 5))

    print(f"Max cut size: {max(cut_sizes)} residues\n")

    Npos, K = len(pos_indices), len(cut_sizes)

    # cut front
    raw_matrix = np.zeros((Npos, K), dtype=float)
    sequence_logits = [[] for _ in range(Npos)]  # Track logits for each sequence
    stopped_sequences = set()  # Track sequences that stopped predicting

    for j, cut in enumerate(cut_sizes):
        print(f"Processing cut size {cut} ({j + 1}/{K})...")
        sub_loader, sub_labels = cut_inputs_embedding(
            predictions, true_labels, df, cut_size=cut, cut_front=True
        )
        _, _, raw_scores = predict(MODEL_PATH, sub_loader, sub_labels, firstrun=False)

        # Update logits and check for stopping sequences
        for row_idx in range(Npos):
            if row_idx not in stopped_sequences:
                raw_matrix[row_idx, j] = raw_scores[row_idx]
                sequence_logits[row_idx].append(raw_scores[row_idx])

                # Check if this sequence should stop further prediction
                if check_dropping_logits_across_cuts(
                    sequence_logits[row_idx], threshold=THRESHOLD
                ):
                    stopped_sequences.add(row_idx)
                    seq_idx = pos_indices[row_idx]
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

        # Find the best cut among the cuts that were actually computed for this sequence
        if row_idx in stopped_sequences:
            # For stopped sequences, only consider cuts up to where they were stopped
            non_zero_indices = np.nonzero(curve)[0]
            if len(non_zero_indices) > 0:
                best_j = non_zero_indices[np.argmax(curve[non_zero_indices])]
            else:
                best_j = 0  # Fallback to first cut
        else:
            # For sequences that completed all cuts
            best_j = int(np.argmax(curve))

        start_residue = cut_sizes[best_j]
        pred_start = cut_sizes[best_j]
        true_start = pos_true_start[row_idx]

        if not true_start:
            err = None
        else:
            err = min(abs(pred_start - ts) for ts in true_start)

        errors_start.append(err)

        status = "stopped early" if row_idx in stopped_sequences else "completed"
        # print(f"Sequence #{seq_idx} ({status}) → domain starts at residue {start_residue}")
        print(f"Seq#{seq_idx}: predicted={pred_start}, true={true_start}, error={err}")
        all_start_residues.append(start_residue)

    print("All domain‐start predictions done.")

    errors_start_wo_nones = [
        e for e in errors_start if e is not None
    ]  # filter out None errors

    print(
        f"\nMean absolute error over {len(errors_start)} positives: {np.mean(errors_start_wo_nones):.1f} residues"
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

    # cut behind
    raw_matrix = np.zeros((Npos, K), dtype=float)
    sequence_logits_end = [[] for _ in range(Npos)]  # Track logits for each sequence
    stopped_sequences_end = set()  # Track sequences that stopped predicting

    for j, cut in enumerate(cut_sizes):
        sub_loader, sub_labels = cut_inputs_embedding(
            predictions, true_labels, df, cut_size=cut, cut_front=False
        )
        _, _, raw_scores = predict(MODEL_PATH, sub_loader, sub_labels)

        # Update logits and check for stopping sequences
        for row_idx in range(Npos):
            if row_idx not in stopped_sequences_end:
                raw_matrix[row_idx, j] = raw_scores[row_idx]
                sequence_logits_end[row_idx].append(raw_scores[row_idx])

                # Check if this sequence should stop further prediction
                if check_dropping_logits_across_cuts(
                    sequence_logits_end[row_idx], threshold=THRESHOLD
                ):
                    stopped_sequences_end.add(row_idx)
                    seq_idx = pos_indices[row_idx]
                    # print(
                    #     f"Stopping further prediction for sequence {seq_idx} after cut {j} due to {THRESHOLD} consecutive logit drops (end prediction)"
                    # )
            # If sequence is stopped, we don't update its logits for this cut size

    # print(
    #     f"Stopped early prediction for {len(stopped_sequences_end)} out of {Npos} sequences (end prediction)"
    # )

    all_end_residues = []
    errors_end = []

    for row_idx, seq_idx in enumerate(pos_indices):
        curve = raw_matrix[row_idx]

        # Find the best cut among the cuts that were actually computed for this sequence
        if row_idx in stopped_sequences_end:
            # For stopped sequences, only consider cuts up to where they were stopped
            non_zero_indices = np.nonzero(curve)[0]
            if len(non_zero_indices) > 0:
                best_j = non_zero_indices[np.argmax(curve[non_zero_indices])]
            else:
                best_j = 0  # Fallback to first cut
        else:
            # For sequences that completed all cuts
            best_j = int(np.argmax(curve))

        sequence_length = len(df[SEQUENCE_COL][seq_idx])
        pred_end = sequence_length - cut_sizes[best_j]

        if seq_idx == 300:
            print(
                f"DEBUG: Seq#{seq_idx} → domain ends at residue {pred_end} (cut size {cut_sizes[best_j]})"
            )
            print(f"DEBUG: {df[SEQUENCE_COL][seq_idx]},{pred_end}")

        true_ends = pos_true_end[row_idx]
        if not true_ends:
            err = None
        else:
            err = min(abs(pred_end - te) for te in true_ends)

        errors_end.append(err)
        all_end_residues.append(pred_end)

        status = "stopped early" if row_idx in stopped_sequences_end else "completed"
        print(
            f"Seq#{seq_idx} ({status}): true_ends={true_ends}, pred_end={pred_end}, error={err}"
        )

    print("All domain‐end predictions done.")

    errors_end_wo_nones = [e for e in errors_end if e is not None]

    print(
        f"\nMean absolute error over {len(errors_end)} positives: {np.mean(errors_end_wo_nones):.1f} residues"
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
                # Concatenate start_residue and end_residue lists
                if isinstance(current_row["start_residue"], list):
                    current_row["start_residue"].extend(endlist[j]["start_residue"])
                else:
                    current_row["start_residue"] = [
                        current_row["start_residue"],
                        endlist[j]["start_residue"],
                    ]

                if isinstance(current_row["end_residue"], list):
                    current_row["end_residue"].extend(endlist[j]["end_residue"])
                else:
                    current_row["end_residue"] = [
                        current_row["end_residue"],
                        endlist[j]["end_residue"],
                    ]

                j += 1

            concatenated_list.append(current_row)
            i = j

        return pd.DataFrame(
            concatenated_list
        ) 

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

    # Separate errors by class
    class1 = [e for e, c in zip(filtered_errors, filtered_classes) if c == 1]
    class2 = [e for e, c in zip(filtered_errors, filtered_classes) if c == 2]

    data = [class1, class2]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(
        data,
        labels=["Class 1", "Class 2"],
        patch_artist=True,
        boxprops=dict(facecolor="C0", edgecolor="k"),
        medianprops=dict(color="yellow"),
    )

    ax.set_title(f"Absolute {Name} Error by Class")
    ax.set_ylabel("Absolute Residue Error")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

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
