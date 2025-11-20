"""
TestsetCreater.py

Used to split the whole available data into a sampled testset and a remaining training set.

Table of Contents:
=========================
1. opener
2. ncbi_taxonomy_split
3. subsampler
4. subsaver
5. main
"""
# -------------------------
# 1. Imports & Globals
# -------------------------

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ete3 import NCBITaxa

CSV_FILE = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/FoundEntriesCompleteProteins_tax.csv"
SUB_OUTPUTFILE = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/SampledEntriesCompleteProteins.csv"
OUTPUT_FILE_TRAIN = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/RemainingEntriesCompleteProteins.csv"
SPLIT_PERCENTAGE = 0.2
pd.set_option("display.max_rows", None)  # Show all rows in DataFrame output

# -------------------------
# 2. Functions
# -------------------------


def opener():
    """
    Opens the CSV file and returns a DataFrame. Early check for file existence.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    # check if file exists
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    # read in & print success message
    df = pd.read_csv(
        CSV_FILE, dtype={"Pfam_id": "category", "Sequence": str, "taxid": "int32"}
    )
    print(f"CSV file {CSV_FILE} opened successfully.")
    return df


def ncbi_taxonomy_split(df):
    """
    Gathers NCBI taxonomy information and append LifeDomain column to the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing a 'taxid' column.
    Returns:
        pd.DataFrame: The DataFrame with an added 'LifeDomain' column.
    """
    # init class
    ncbi = NCBITaxa()

    # Get unique taxids
    unique_taxids = df["taxid"].unique()

    # Batch process lineages for all unique taxids at once
    lineages = {}
    all_lineage_ids = set()

    # loop through unique taxids and get lineages
    for taxid in unique_taxids:
        try:
            # get lineage for taxid
            lineage = ncbi.get_lineage(taxid)
            lineages[taxid] = lineage
            all_lineage_ids.update(lineage)
        # catch exceptions for missing taxids
        except Exception as e:
            lineages[taxid] = []
            print(f"Error processing taxid {taxid}: {e}")

    # Single batch call to get all names at once
    if all_lineage_ids:
        all_names = ncbi.get_taxid_translator(list(all_lineage_ids))
    # If no lineage IDs were found, set all_names to empty dict
    else:
        all_names = {}

    # Create lookup dictionary for taxid -> LifeDomain
    taxid_to_domain = {}
    # Domains of Life
    target_domains = {"Bacteria", "Archaea", "Eukaryota"}

    # Map taxids to LifeDomain
    for taxid, lineage in lineages.items():
        # default to other if not changed (i.e. virus)
        domain = "Other"
        if lineage:
            for tid in lineage:
                name = all_names.get(tid, "")
                if name in target_domains:
                    domain = name
                    break
        # If lineage is empty (error case) for artificial sequences
        else:
            domain = "Unknown"
        # sort bacj to dict
        taxid_to_domain[taxid] = domain

    print("Done tax lookup")
    # Vectorized mapping to DataFrame
    df["LifeDomain"] = df["taxid"].map(taxid_to_domain).astype("category")

    return df


def subsampler(df):
    """
    Subsamples the DataFrame to create a smaller dataset, tries to ensure all Pfam IDs are present in both sets.
    Args:
        df (pd.DataFrame): The original DataFrame to be subsampled.
    Returns:
        df_sampled (pd.DataFrame): The subsampled DataFrame.
        df_remaining (pd.DataFrame): The remaining DataFrame after subsampling.
    """
    # Shuffle and identical sequence duplicates
    df = df.sample(frac=1, random_state=42)
    df = df.drop_duplicates(subset=["Pfam_id", "Sequence"])
    # Keep only Pfam IDs with more than one occurrence. If only one, can't be really confirmed during ml training, so useless.
    df = df[df["Pfam_id"].map(df["Pfam_id"].value_counts()) > 1]

    # start time stamp
    time_start = time.time()

    # Vectorized approach to select sampled IDs
    unique_ids = df["id"].unique()

    # shuffle again
    np.random.seed(42)
    np.random.shuffle(unique_ids)

    # decide how many ids to sample based on split percentage
    n_sampled_ids = max(1, int(len(unique_ids) * SPLIT_PERCENTAGE))
    sampled_ids = set(unique_ids[:n_sampled_ids])

    # create a mask and apply the positive mask for subset and negative for train set
    mask_sampled = df["id"].isin(sampled_ids)
    df_sampled = df[mask_sampled].copy()
    df_remaining = df[~mask_sampled].copy()

    # Immediate cleanup to save memory
    del mask_sampled, sampled_ids, unique_ids
    # Store len before deleting
    original_len = len(df)
    del df

    # Overlap check to ensure no IDs are in both sets
    sampled_ids_set = set(df_sampled["id"].unique())
    remaining_ids_set = set(df_remaining["id"].unique())
    assert sampled_ids_set.isdisjoint(remaining_ids_set), (
        "Found overlapping IDs between sets!"
    )
    print(
        "Sanity check passed: no overlapping IDs between sampled and remaining DataFrames."
    )
    # more memory cleanup
    del sampled_ids_set, remaining_ids_set

    # check if there are overlapping entries (same Pfam_id, Sequence and taxid) in both dataframes
    df_sampled_check = df_sampled[["Pfam_id", "Sequence", "taxid"]]
    df_remaining_check = df_remaining[["Pfam_id", "Sequence", "taxid"]]

    # hash approach for comparison
    sampled_hash = set(pd.util.hash_pandas_object(df_sampled_check, index=False))
    remaining_hash = set(pd.util.hash_pandas_object(df_remaining_check, index=False))

    # assertion to check for overlaps
    assert sampled_hash.isdisjoint(remaining_hash), (
        "Found overlapping entries between sets!"
    )
    print(
        "Sanity check passed: no overlapping entries between sampled and remaining DataFrames.\n"
    )

    # more memory cleanup
    del df_sampled_check, df_remaining_check, sampled_hash, remaining_hash

    # Apply taxonomy split to subsetps
    df_sampled = ncbi_taxonomy_split(df_sampled)
    df_remaining = ncbi_taxonomy_split(df_remaining)

    # final time stamp
    print(f"Subsampling completed in {time.time() - time_start:.2f} seconds.")

    ### RESULT PRINTS ###
    # print distribution of LifeDomains in both dataframes
    distribution_relative = df_remaining["LifeDomain"].value_counts(normalize=True)
    print("\nDistribution Main df_remaining:", distribution_relative, "\n")
    distribution_sampled_relative = df_sampled["LifeDomain"].value_counts(
        normalize=True
    )
    print("Distribution Sampled df:", distribution_sampled_relative, "\n")

    # print percentage of sampled entries
    print(len(df_sampled), "samples selected from total length of", original_len)
    print(
        "Percentage of original DataFrame in sampled DataFrame:",
        len(df_sampled) / original_len * 100,
        "%\n",
    )

    # print average sequence lengths in both dataframes
    avg_length_df_remaining = df_remaining["Sequence"].apply(len).mean()
    avg_length_sampled = df_sampled["Sequence"].apply(len).mean()
    print(
        f"Average sequence length in original DataFrame: {avg_length_df_remaining:.2f} characters"
    )
    print(
        f"Average sequence length in sampled DataFrame: {avg_length_sampled:.2f} characters\n"
    )

    # Create Sequence_Length columns
    df_remaining["Sequence_Length"] = df_remaining["Sequence"].apply(len)
    df_sampled["Sequence_Length"] = df_sampled["Sequence"].apply(len)

    return df_sampled, df_remaining


def plotter(df_sampled, df_remaining):
    """
    Plots the distribution of sequence lengths in both DataFrames and prints missing/overlapping Pfam IDs of both sets.
    Args:
        df_sampled (pd.DataFrame): The sampled DataFrame.
        df_remaining (pd.DataFrame): The remaining DataFrame.
    """

    # init first plot: distribution of sequence lengths
    plt.figure(figsize=(12, 6))
    # first subplot
    plt.subplot(1, 2, 1)
    sns.set_palette("Set2")
    sns.histplot(
        df_remaining["Sequence_Length"], bins=50, alpha=0.7, stat="probability"
    )
    plt.title("Distribution of Sequence Lengths in Original DataFrame")
    plt.xlabel("Sequence Length")
    plt.ylabel("Fraction")

    # second subplot
    plt.subplot(1, 2, 2)
    sns.histplot(df_sampled["Sequence_Length"], bins=50, alpha=0.7, stat="probability")
    plt.title("Distribution of Sequence Lengths in Sampled DataFrame")
    plt.xlabel("Sequence Length")
    plt.ylabel("Fraction")

    plt.tight_layout()
    plt.savefig(
        "/global/research/students/sapelt/Masters/MasterThesis/sequence_length_distribution.png"
    )
    plt.show()

    # find overlapping and missing pfam ids in both dataframes
    # overlapping pfam ids
    overlapping_pfam_ids = set(df_sampled["Pfam_id"]).intersection(
        set(df_remaining["Pfam_id"])
    )
    print(f"\nNumber of overlapping pfam_ids: {len(overlapping_pfam_ids)}\n")

    #
    percent_in_sampled = (
        len(overlapping_pfam_ids) / df_sampled["Pfam_id"].nunique() * 100
    )
    percent_in_original = (
        len(overlapping_pfam_ids) / df_remaining["Pfam_id"].nunique() * 100
    )
    print(
        f"Percentage of overlapping pfam_ids in sampled DataFrame: {percent_in_sampled:.2f}%"
    )
    print(
        f"Percentage of overlapping pfam_ids in original DataFrame: {percent_in_original:.2f}%\n"
    )

    # missing pfam ids in the sampled DataFrame
    missing_pfam_ids = set(df_remaining["Pfam_id"]) - set(df_sampled["Pfam_id"])
    percent_missing = len(missing_pfam_ids) / df_remaining["Pfam_id"].nunique() * 100
    print(
        f"Number of missing pfam_ids in sampled DataFrame: {len(missing_pfam_ids)} ({percent_missing:.2f}%)\n"
    )

    # share of each pfam id in the sampled and train DataFrame
    pfam_counts = df_sampled["Pfam_id"].value_counts(normalize=True)
    original_pfam_counts = df_remaining["Pfam_id"].value_counts(normalize=True)

    # Create a boxplot comparing pfam_id distributions
    plt.figure(figsize=(10, 6))
    data_for_boxplot = [pfam_counts.values, original_pfam_counts.values]
    labels = ["Sampled Dataframe", "Remaining Dataframe"]
    plt.boxplot(data_for_boxplot, labels=labels)
    plt.title("Distribution of Pfam ID Frequencies")
    plt.ylabel("Frequency (Normalized)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "/global/research/students/sapelt/Masters/MasterThesis/pfam_frequency_boxplot.png"
    )
    plt.show()


def subsaver(df_sampled, df_remaining):
    """
    Saves the sampled DataFrame to two CSV based on given global paths.
    """
    df_sampled.to_csv(SUB_OUTPUTFILE, index=False)
    print(f"\nSampled DataFrame saved to {SUB_OUTPUTFILE}")

    # remove the entries taken from the original DataFrame
    df_remaining.to_csv(OUTPUT_FILE_TRAIN, index=False)
    print(f"\nRemaining DataFrame saved to {OUTPUT_FILE_TRAIN}\n")


# -------------------------
# 3. Main
# -------------------------


def main():
    """
    Main function to execute the subsampling process.
    """
    df = opener()
    df_sampled, df_remaining = subsampler(df)
    # memory cleanup
    del df
    plotter(df_sampled, df_remaining)
    subsaver(df_sampled, df_remaining)


###############################################################################################################
if __name__ == "__main__":
    # Redirect stdout to both terminal and a file to log the process
    original_stdout = sys.stdout
    with open(
        "/global/research/students/sapelt/Masters/MasterThesis/Results/TestsetCreater.txt",
        "w",
    ) as f:

        class TeeOutput:
            """
            Tee class to duplicate stdout to both terminal and file. Copied from StackOverflow.
            """

            def write(self, text):
                original_stdout.write(text)
                f.write(text)

            def flush(self):
                original_stdout.flush()
                f.flush()

        # Redirect stdout
        sys.stdout = TeeOutput()

        # Execute main process
        print("Starting subsampling process...")
        main()
        print("Subsampling completed successfully.")

        # Restore original stdout
        sys.stdout = original_stdout

    # # Used for the creation of the class 0 data for the final model, as they need also the column life domain and sequence length. Those are the artifical sequences and uniprot sequences without any domain in it.
    # print("Starting subsampling process...")
    # df = pd.read_csv(CSV_FILE, dtype={'taxid': 'Int32'})
    # df =ncbi_taxonomy_split(df)
    # df["Sequence_Length"] = df['Sequence'].apply(len)

    # df.to_csv("/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/Unknown_class_test_with_domains_and_lengths.csv", index=False)
