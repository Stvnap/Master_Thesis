"""
tf.dataset_to_csv.py

This script converts a saved TensorFlow tf.data.Dataset into a CSV file. Used in early development to inspect dataset contents.

# Table of contents:
=========================
1. converter
"""

# -------------------------
# Imports and Globals
# -------------------------

import tensorflow as tf
import polars as pl
import time

DS_PATH = "./trainsetSP2d"  # Path to the saved tf.data.Dataset
OUT_PATH = "./DataTrainALLOHenc.csv"  # Output CSV file path


# -------------------------
# Functions
# -------------------------


def converter(ds_path, out_path):
    """
    Converts a saved tf.data.Dataset to CSV with columns: ID, Sequences, Labels
    Args:
        ds_path (str): Path to the saved tf.data.Dataset
        out_path (str): Output CSV file path
    """
    # Start timer
    start_time = time.time()

    # Load dataset (must be a TF-saved dataset directory)
    ds = tf.data.Dataset.load(ds_path)

    # init lists
    ids = []
    seqs = []
    labels = []

    # Iterate through dataset and extract values
    for seq, label, id_ in ds:
        try:
            seqs.append(seq.numpy().tolist())  # convert numpy array to list
            labels.append(int(label.numpy()))   # convert to int
            ids.append(id_.numpy().decode("utf-8"))  # decode bytes to string
        except:
            # If already decoded
            seqs.append(seq)
            labels.append(label)
            ids.append(id_)

    # Create Polars DataFrame and save to CSV
    df = pl.DataFrame({
        "ID": ids,
        "Sequences": seqs,
        "Labels": labels,
    })

    # Save to CSV & final print
    df.write_csv(out_path)
    print(f"✅ Saved CSV in {time.time() - start_time:.2f} seconds | {len(df)} entries → {out_path}")


##########################
if __name__ == "__main__":
    converter(DS_PATH, OUT_PATH)