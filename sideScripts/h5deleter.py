"""
h5deleter.py

Script to delete specific keys from an HDF5 file. Was used due to currupted gpu generating nan values in embeddings.

Table of contents:
=========================
1. delete_h5_file
"""

# -------------------------
# Imports and Globals
# -------------------------

import os

import h5py
import fnmatch

EMBEDDINGS_H5FILE = (
    "/scratch/tmp/sapelt/Master_Thesis/temp/embeddings_classification_d.h5"
)
KEYS_TO_DELETE = [
    "train_embeddings_chunk16*",
    "train_embeddings_chunk17*", 
    "train_embeddings_chunk18*",
    "train_labels_chunk16*",
    "train_labels_chunk17*",
    "train_labels_chunk18*",
    "val_embeddings_chunk16*",
    "val_embeddings_chunk17*",
    "val_embeddings_chunk18*",
    "val_labels_chunk16*",
    "val_labels_chunk17*",
    "val_labels_chunk18*"
]

# False = only print keys, True = delete set keys
DELETE = False

# -------------------------
# Functions
# -------------------------

def matches_pattern(key, patterns):
    """Check if a key matches any of the wildcard patterns."""
    for pattern in patterns:
        if fnmatch.fnmatch(key, pattern):
            return True
    return False

def delete_h5_file(file_path, delete=False):
    """
    Deletes a specific hd5f key from an HDF5 fi
    """

    # Check if file exists, else return
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    # Open the HDF5 file and delete specified keys
    try:
        with h5py.File(file_path, "a") as h5file:
            # List all keys in the file
            keys = list(h5file.keys())
            # Check if there are any keys to delete
            if not keys:
                print(f"No keys found in {file_path}.")
                return

            # Print all keys before deletion and asign keys to delete
            print(f"Keys in the file: {keys}")
            
            # early exit for only printing keys
            if not delete:
                print("Delete flag is set to False. No keys will be deleted.")
                return
            
            # init list 
            keys_to_delete = []
            # Find keys matching patterns
            for key in keys:
                if matches_pattern(key, KEYS_TO_DELETE):
                    keys_to_delete.append(key)
            
            # Delete the matching keys
            for key in keys_to_delete:
                del h5file[key]
                print(f"Key '{key}' has been deleted from {file_path}.")
            # If no keys matched print message
            if not keys_to_delete:
                print(f"No keys matched the patterns in {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


####################################################
if __name__ == "__main__":
    delete_h5_file(EMBEDDINGS_H5FILE, delete=DELETE)