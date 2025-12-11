"""
EvalCreaterALL.py

This script processes the training data to create evaluation sequences by trimming the protein sequence.
Goal is to single out one domain with additional context on each side, removing other domains.
Idea was to leave the noisy sequence before and after the domain, to set a more realistic performance benchmark, yet avoid the problem of multiple domains confusing the model.
For this we cut the sequence from the start of the previous domain to the start of the next domain (if applicable), or from the beginning/end of the sequence otherwise.

Table of contents:
=========================
1. Load CSV
2. ReDefiner
3. Save to CSV
4. main
"""

# -------------------------
# Imports and Globals
# -------------------------
import os

import pandas as pd

CSV_PATH = "/global/students/research/sapelt/Masters/MasterThesis/Dataframes/v3/RemainingEntriesCompleteProteins.csv"
START_COL = "start"
END_COL = "end"
ID_COL = "id"
PFAM_COL = "Pfam_id"
SEQUENCE_COL = "Sequence"
OUTPUT_PATH = "/global/students/research/sapelt/Masters/MasterThesis/Dataframes/v3/RemainingEntriesCompleteProteins_Eval.csv"
pd.set_option("display.max_rows", None)  # Show all rows in DataFrame

# -------------------------
# Functions
# -------------------------


def load_csv():
    """
    Load the CSV file and return a DataFrame. check if file exists
    """
    # file check
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

    # return dataframe
    print(f"Loading CSV from {CSV_PATH}")
    return pd.read_csv(
        CSV_PATH,
        usecols=[START_COL, END_COL, ID_COL, PFAM_COL, SEQUENCE_COL],
        index_col=False,
    )


def ReDefiner(df):
    """
    Function to trim sequences based on known domain boundaries.
    For each domain entry, the sequence is cut from the start of the previous domain to the start of the next domain (if applicable),
    or from the beginning/end of the sequence otherwise.
    Args:
        df (pd.DataFrame): DataFrame containing domain entries with columns for ID, start,
    Returns:
        trimmed_df (pd.DataFrame): DataFrame with trimmed sequences.
    """
    # Sort by ID and START
    df = df.sort_values(by=[ID_COL, START_COL]).reset_index(drop=True)

    # Initialize list to hold trimmed rows and group by ID
    result = []
    grouped = df.groupby(ID_COL)

    # Process each group
    for id_val, group in grouped:
        # Reset index for the group
        group = group.reset_index(drop=True)
        # num_rows = number of domains for this ID
        num_rows = len(group)
        # Process each domain in the group
        for i, row in group.iterrows():
            # Get the original sequence
            sequence = row[SEQUENCE_COL]
            # new cut sequence
            new_seq = ""

            # Determine trimming based on position
            if num_rows == 1:
                # Only one entry for this ID → keep entire sequence
                new_seq = sequence
            else:
                # Multiple entries → trim accordingly
                if i == 0:
                    try:
                        # First entry: cut from 0 to start of next domain
                        next_start = group.loc[i + 1, START_COL]
                        new_seq = sequence[0:next_start]
                    except:
                        # in case of artifical sequence, keep full sequence
                        new_seq = sequence
                elif i < num_rows - 1:
                    # Middle entries: cut from end of previous to start of next
                    prev_end = group.loc[i - 1, END_COL]
                    next_start = group.loc[i + 1, START_COL]
                    new_seq = sequence[prev_end:next_start]
                else:
                    # Last entry: cut from end of previous to end of sequence
                    prev_end = group.loc[i - 1, END_COL]
                    new_seq = sequence[prev_end:]

            # Store trimmed row
            trimmed = row.copy()
            # apply to sequence column
            trimmed[SEQUENCE_COL] = new_seq
            # append to result list
            result.append(trimmed)

    # Create final DataFrame
    trimmed_df = pd.DataFrame(result)
    print("Data trimmed")
    return trimmed_df


def save_to_csv(df, output_path):
    """
    Save the DataFrame to a CSV file at given output path.
    """
    df.to_csv(output_path, index=False)
    print(f"DataFrame saved to {output_path}")


# -------------------------
# main
# -------------------------
def main():
    # load in
    df = load_csv()
    print(f"Loaded DataFrame with {len(df)} entries")

    # redefine sequences
    trimmed_df = ReDefiner(df)

    # save csv
    save_to_csv(trimmed_df, OUTPUT_PATH)


########################################################################################################################
if __name__ == "__main__":
    main()