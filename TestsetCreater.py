import os
import pandas as pd
import seaborn as sns
from ete3 import NCBITaxa
import sys
import numpy as np
import time 
import gc
CSV_FILE = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/FoundEntriesCompleteProteins_tax.csv"
SUB_OUTPUTFILE = '/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/SampledEntriesCompleteProteins.csv'
OUTPUT_FILE_TRAIN = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/RemainingEntriesCompleteProteins.csv"
SPLIT_PERCENTAGE=0.2

pd.set_option('display.max_rows', None)  # Show all rows in DataFrame output


def opener():
    """
    Opens the CSV file and returns a DataFrame.
    """
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
    
    df = pd.read_csv(CSV_FILE, dtype={'Pfam_id': 'category', 'Sequence': str, 'taxid': "int32"}) 
    print(f"CSV file {CSV_FILE} opened successfully.")
    return df


def ncbi_taxonomy_split(df):
    """
    gets the distribution of Kingdoms of life in the sampled DataFrames
    """
    ncbi = NCBITaxa()
    
    # Get unique taxids to avoid redundant lookups
    unique_taxids = df['taxid'].unique()
    

    
    # Batch process lineages for all unique taxids at once
    lineages = {}
    all_lineage_ids = set()
    
    for taxid in unique_taxids:
        try:
            lineage = ncbi.get_lineage(taxid)
            lineages[taxid] = lineage
            all_lineage_ids.update(lineage)
        except Exception as e:
            lineages[taxid] = []
            print(f"Error processing taxid {taxid}: {e}")
            

    # Single batch call to get all names at once
    if all_lineage_ids:
        all_names = ncbi.get_taxid_translator(list(all_lineage_ids))
    else:
        all_names = {}
    
    # Create lookup dictionary for taxid -> LifeDomain
    taxid_to_domain = {}
    target_domains = {"Bacteria", "Archaea", "Eukaryota"}
    
    for taxid, lineage in lineages.items():
        domain = "Other"
        if lineage:
            for tid in lineage:
                name = all_names.get(tid, "")
                if name in target_domains:
                    domain = name
                    break
        else:
            domain = "Unknown"
        taxid_to_domain[taxid] = domain
    
    print("Done tax lookup")
    
    # Vectorized mapping instead of apply
    df['LifeDomain'] = df['taxid'].map(taxid_to_domain).astype('category')
    
    return df



def subsampler(df):
    """
    Subsamples the DataFrame to create a smaller dataset, ensuring all Pfam IDs are present in both sets.
    """
    df = df.sample(frac=1, random_state=42)  # Shuffle
    df = df.drop_duplicates(subset=['Pfam_id', 'Sequence'])
    df = df[df['Pfam_id'].map(df['Pfam_id'].value_counts()) > 1]

    time_start = time.time()

    # Vectorized approach - much faster
    unique_ids = df['id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_ids)
    
    n_sampled_ids = max(1, int(len(unique_ids) * SPLIT_PERCENTAGE))
    sampled_ids = set(unique_ids[:n_sampled_ids])
    
    # Single vectorized operation instead of groupby loop
    mask_sampled = df['id'].isin(sampled_ids)
    df_sampled = df[mask_sampled].copy()
    df_remaining = df[~mask_sampled].copy()

    # Immediate cleanup
    del mask_sampled, sampled_ids, unique_ids
    original_len = len(df)  # Store before deleting
    del df
    gc.collect()


    # Sanity checks with immediate cleanup
    sampled_ids_set = set(df_sampled['id'].unique())
    remaining_ids_set = set(df_remaining['id'].unique())
    assert sampled_ids_set.isdisjoint(remaining_ids_set), "Found overlapping IDs between sets!"
    del sampled_ids_set, remaining_ids_set
    print("Sanity check passed: no overlapping IDs between sampled and remaining DataFrames.")

    # More memory-efficient overlap check
    df_sampled_check = df_sampled[['Pfam_id', 'Sequence', 'taxid']]
    df_remaining_check = df_remaining[['Pfam_id', 'Sequence', 'taxid']]
    
    sampled_hash = set(pd.util.hash_pandas_object(df_sampled_check, index=False))
    remaining_hash = set(pd.util.hash_pandas_object(df_remaining_check, index=False))
    
    assert sampled_hash.isdisjoint(remaining_hash), "Found overlapping entries between sets!"
    del df_sampled_check, df_remaining_check, sampled_hash, remaining_hash

    print("Sanity check passed: no overlapping entries between sampled and remaining DataFrames.\n")

    # Apply taxonomy split and optimize memory
    df_sampled = ncbi_taxonomy_split(df_sampled)

    df_remaining = ncbi_taxonomy_split(df_remaining)


    print(f"Subsampling completed in {time.time() - time_start:.2f} seconds.")



    ### TEST PRINTS ###

    distribution_relative = df_remaining['LifeDomain'].value_counts(normalize=True)
    print("\nDistribution Main df_remaining:",distribution_relative, "\n")


    distribution_sampled_relative = df_sampled['LifeDomain'].value_counts(normalize=True)
    print("Distribution Sampled df:",distribution_sampled_relative, "\n")
    

    print(len(df_sampled), "samples selected from total length of", original_len)
    print("Percentage of original DataFrame in sampled DataFrame:", len(df_sampled) / original_len * 100, "%\n")
    

    avg_length_df_remaining = df_remaining['Sequence'].apply(len).mean()
    avg_length_sampled = df_sampled['Sequence'].apply(len).mean()
    print(f"Average sequence length in original DataFrame: {avg_length_df_remaining:.2f} characters")
    print(f"Average sequence length in sampled DataFrame: {avg_length_sampled:.2f} characters\n")


    # plot the distribution of sequence lengths in both DataFrames
    df_remaining['Sequence_Length'] = df_remaining['Sequence'].apply(len)
    df_sampled['Sequence_Length'] = df_sampled['Sequence'].apply(len)


    # Plotting the distribution of sequence lengths
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.set_palette('Set2')
    sns.histplot(df_remaining['Sequence_Length'], bins=50, alpha=0.7, stat='probability')
    plt.title('Distribution of Sequence Lengths in Original DataFrame')
    plt.xlabel('Sequence Length')
    plt.ylabel('Fraction')

    plt.subplot(1, 2, 2)
    sns.histplot(df_sampled['Sequence_Length'], bins=50, alpha=0.7, stat='probability')
    plt.title('Distribution of Sequence Lengths in Sampled DataFrame')
    plt.xlabel('Sequence Length')
    plt.ylabel('Fraction')

    plt.tight_layout()
    plt.savefig("/global/research/students/sapelt/Masters/MasterThesis/sequence_length_distribution.png")
    plt.show()





    # share of overlapping pfam ids found in both dfs
    overlapping_pfam_ids = set(df_sampled['Pfam_id']).intersection(set(df_remaining['Pfam_id']))
    print(f"\nNumber of overlapping pfam_ids: {len(overlapping_pfam_ids)}\n")

    percent_in_sampled = len(overlapping_pfam_ids) / df_sampled['Pfam_id'].nunique() * 100
    percent_in_original = len(overlapping_pfam_ids) / df_remaining['Pfam_id'].nunique() * 100
    print(f"Percentage of overlapping pfam_ids in sampled DataFrame: {percent_in_sampled:.2f}%")
    print(f"Percentage of overlapping pfam_ids in original DataFrame: {percent_in_original:.2f}%\n")

    # missing pfam ids in the sampled DataFrame
    missing_pfam_ids = set(df_remaining['Pfam_id']) - set(df_sampled['Pfam_id'])
    percent_missing = len(missing_pfam_ids) / df_remaining['Pfam_id'].nunique() * 100
    print(f"Number of missing pfam_ids in sampled DataFrame: {len(missing_pfam_ids)} ({percent_missing:.2f}%)\n")

    # share of each pfam id in the sampled DataFrame
    pfam_counts = df_sampled['Pfam_id'].value_counts(normalize=True)
    # print("Share of each pfam_id in the sampled DataFrame:")
    # print(pfam_counts, "\n")

    # share of each pfam id in the original DataFrame
    original_pfam_counts = df_remaining['Pfam_id'].value_counts(normalize=True)
    # print("Share of each pfam_id in the original DataFrame:")
    # print(original_pfam_counts, "\n")


    # Create a boxplot comparing pfam_id distributions
    plt.figure(figsize=(10, 6))

    # Prepare data for boxplot
    data_for_boxplot = [pfam_counts.values, original_pfam_counts.values]
    labels = ['Sampled Dataframe', 'Remaining Dataframe']

    plt.boxplot(data_for_boxplot, labels=labels)
    plt.title('Distribution of Pfam ID Frequencies')
    plt.ylabel('Frequency (Normalized)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/global/research/students/sapelt/Masters/MasterThesis/pfam_frequency_boxplot.png")
    plt.show()


    
    return df_sampled, df_remaining  # Return df_remaining instead of df


def subsaver(df_sampled, df_remaining):
    """
    Saves the sampled DataFrame to a CSV file.
    """
    df_sampled.to_csv(SUB_OUTPUTFILE, index=False)
    print(f"\nSampled DataFrame saved to {SUB_OUTPUTFILE}")

    #remove the entries taken from the original DataFrame
    df_remaining.to_csv(OUTPUT_FILE_TRAIN, index=False)
    print(f"\nRemaining DataFrame saved to {OUTPUT_FILE_TRAIN}\n")


def main():
    """
    Main function to execute the subsampling process.
    """
    df = opener()
    df_sampled, df_remaining = subsampler(df)
    del df
    subsaver(df_sampled, df_remaining)

if __name__ == "__main__":
    # Redirect stdout to both terminal and file
    original_stdout = sys.stdout
    with open('/global/research/students/sapelt/Masters/MasterThesis/Results/TestsetCreater.txt', 'w') as f:
        class TeeOutput:
            def write(self, text):
                original_stdout.write(text)
                f.write(text)
            def flush(self):
                original_stdout.flush()
                f.flush()
        
        sys.stdout = TeeOutput()
        
        print("Starting subsampling process...")
        main()
        print("Subsampling completed successfully.")
        
        sys.stdout = original_stdout

    # print("Starting subsampling process...")
    # df = pd.read_csv(CSV_FILE, dtype={'taxid': 'Int32'})
    # df =ncbi_taxonomy_split(df)
    # df["Sequence_Length"] = df['Sequence'].apply(len)

    # df.to_csv("/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/Unknown_class_test_with_domains_and_lengths.csv", index=False)

