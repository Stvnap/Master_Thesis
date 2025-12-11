"""
importh5.py

This script reads embeddings and labels from an H5 file, inspects the data structure, and prints detailed information about the first example.
Used to detect bug where some gpus produced nan values in the embeddings.

Table of contents:
=========================
1. read_embeddings_file
2. inspect_first_example
3. main
"""

# -------------------------
# Imports and Globals
# -------------------------

import h5py
import numpy as np

H5FILE = "/scratch/tmp/sapelt/Master_Thesis/temp/embeddings_classification_24380d.h5"
NUM_CLASSES = int(H5FILE.split("_")[-1].split(".h5")[0].split("d")[0])
print(f"Number of classes: {NUM_CLASSES}")
# -------------------------
# Functions
# -------------------------


def read_embeddings_file(h5_file_path):
    """
    Read embeddings and labels from your H5 file format
    Args:
        h5_file_path (str): Path to the H5 file
    Returns:
        combined_embeddings (np.ndarray): Combined embeddings array
        combined_labels (np.ndarray): Combined labels array
    """

    # Open the H5 file
    with h5py.File(h5_file_path, "r") as f:
        # print h5 file keys
        print("Available keys:", list(f.keys()))

        # Get embedding keys (based on your naming pattern)
        embedding_keys = sorted(
            [
                k
                for k in f.keys()
                if k.startswith("embeddings_")
                or k.startswith("train_embeddings_")
                or k.startswith("val_embeddings_")
            ]
        )

        # Get corresponding label keys
        label_keys = sorted(
            [
                k
                for k in f.keys()
                if k.startswith("labels_")
                or k.startswith("train_labels_")
                or k.startswith("val_labels_")
            ]
        )

        # Print found keys
        print(f"Found {len(embedding_keys)} embedding chunks")
        print(f"Found {len(label_keys)} label chunks")

        # init lists for all embeddings and labels
        all_embeddings = []
        all_labels = []

        # Read only the first embeddings and labels
        if embedding_keys:
            embeddings = f[embedding_keys[0]][:]
            all_embeddings.append(embeddings)
            print(f"{embedding_keys[0]}: shape {embeddings.shape}")
        if label_keys:
            labels = f[label_keys[0]][:]
            all_labels.append(labels)
            print(f"{label_keys[0]}: shape {labels.shape}")

        # Use the first chunks directly
        if all_embeddings:
            combined_embeddings = all_embeddings[0]
        if all_labels:
            combined_labels = all_labels[0]

    return combined_embeddings, combined_labels


def inspect_first_example():
    """
    Print detailed information about the first example
    """

    print("Debugging HDF5 file contents...")
    with h5py.File(H5FILE, "r") as f:
        print("Available keys:", list(f.keys()))

        # Track statistics across all datasets
        total_nan_count = 0
        total_inf_count = 0
        total_samples = 0

        for key in f.keys():
            if "labels" in key:
                labels_data = f[key][:]
                labels_sample = labels_data[:100]  # Sample first 100 labels
                print(
                    f"{key}: shape={f[key].shape}, dtype={f[key].dtype}, sample={labels_sample}"
                )
                print(f"  Min: {labels_data.min()}, Max: {labels_data.max()}")

                # Check for invalid labels
                invalid_labels = labels_data < 0
                if invalid_labels.any():
                    print(
                        f"  WARNING: Found {invalid_labels.sum()} invalid labels (< 0)"
                    )

                out_of_range = labels_data >= NUM_CLASSES
                if out_of_range.any():
                    print(
                        f"  WARNING: Found {out_of_range.sum()} labels >= {NUM_CLASSES}"
                    )

            if "embeddings" in key:
                embeddings_data = f[key][:]
                print(f"{key}: shape={f[key].shape}, dtype={f[key].dtype}")

                # Check for NaN values
                nan_mask = np.isnan(embeddings_data)
                nan_count = nan_mask.sum()
                total_nan_count += nan_count

                # Check for infinite values
                inf_mask = np.isinf(embeddings_data)
                inf_count = inf_mask.sum()
                total_inf_count += inf_count

                total_samples += embeddings_data.size

                print(
                    f"  NaN values: {nan_count} ({nan_count / embeddings_data.size * 100:.4f}%)"
                )
                print(
                    f"  Inf values: {inf_count} ({inf_count / embeddings_data.size * 100:.4f}%)"
                )
                print(
                    f"  Min: {np.nanmin(embeddings_data):.4f}, Max: {np.nanmax(embeddings_data):.4f}"
                )
                print(
                    f"  Mean: {np.nanmean(embeddings_data):.4f}, Std: {np.nanstd(embeddings_data):.4f}"
                )

                # Check for suspiciously large values
                large_values = np.abs(embeddings_data) > 100
                if large_values.any():
                    print(
                        f"  WARNING: Found {large_values.sum()} values with |x| > 100"
                    )

                # Check for zero embeddings (could indicate processing errors)
                zero_embeddings = np.all(embeddings_data == 0, axis=1)
                if zero_embeddings.any():
                    print(f"  WARNING: Found {zero_embeddings.sum()} zero embeddings")

                # Sample some actual values for manual inspection
                sample_indices = np.random.choice(
                    embeddings_data.shape[0],
                    min(3, embeddings_data.shape[0]),
                    replace=False,
                )
                for idx in sample_indices:
                    sample_embedding = embeddings_data[idx]
                    print(f"  Sample {idx}: first 5 dims = {sample_embedding[:5]}")

        print("\nOverall statistics:")
        print(
            f"  Total NaN values: {total_nan_count} ({total_nan_count / total_samples * 100:.6f}%)"
        )
        print(
            f"  Total Inf values: {total_inf_count} ({total_inf_count / total_samples * 100:.6f}%)"
        )
        print(f"  Total embedding values: {total_samples}")

        # If significant issues found, recommend actions
        if total_nan_count > 0:
            print("  🚨 CRITICAL: Found NaN values in embeddings!")
        if total_inf_count > 0:
            print("  🚨 CRITICAL: Found infinite values in embeddings!")


# -------------------------
# Main
# -------------------------


def main():
    """Main function to read H5 file and inspect first example"""
    try:
        print(f"Reading H5 file: {H5FILE}")
        # read embeddings and labels
        # embeddings, labels = read_embeddings_file(H5FILE)

        # Inspect the first example in detail
        inspect_first_example()

    # except handling
    except FileNotFoundError:
        print("H5 file not found. Make sure the path is correct.")
    except Exception as e:
        print(f"Error reading H5 file: {e}")


###########################################
if __name__ == "__main__":
    main()
