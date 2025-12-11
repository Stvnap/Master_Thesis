"""
Combiner.py

Combines the artifically generated sequences and uniprot sequences without domains with the trainings data to create a training set for the MAX model.
For this the two csv are concatinated and missing columns are filled with empty strings.

Table of Contents:
=========================
1. combiner
"""

# -------------------------
# Imports & Globals
# -------------------------
import pandas as pd

LARGECSV = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/SampledEntriesCompleteProteins.csv"
SMALLCSV = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/Unknown_class_test.csv"
OUTPUT = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/SampledEntriesCompleteProteins_MAX.csv"

# -------------------------
# Function
# -------------------------
def combiner():
    """
    Combines the class 0 csv for the max model with the training set. Saves it directly to OUTPUT.
    """
    # Load both files, training st and class 0
    df_large = pd.read_csv(LARGECSV)
    df_small = pd.read_csv(SMALLCSV)

    # Get the union of all columns
    all_columns = list(df_large.columns) + [
        c for c in df_small.columns if c not in df_large.columns
    ]

    # check columns
    print(all_columns)

    # Reindex both DataFrames to have the same columns (missing ones get filled with empty string)
    df_large = df_large.reindex(columns=all_columns, fill_value="")
    df_small = df_small.reindex(columns=all_columns, fill_value="")

    # Concatenate
    combined = pd.concat([df_large, df_small], ignore_index=True)

    # Free up memory
    del df_large, df_small

    # Shuffle the combined DataFrame
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to new file
    combined.to_csv(OUTPUT, index=False)

    print(f"Combined file written to {OUTPUT}")


#################################
if __name__ == "__main__":
    combiner()