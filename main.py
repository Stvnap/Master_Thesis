"""
========================================================================
PROTEIN PFAM DOMAIN PREDICTION PIPELINE - MAIN SCRIPT
========================================================================

Author: Steven Apelt
Project: Master's Thesis - EBB Group, University of Münster
Version: 0.1
Date: 17 November 2025

Description:
Main entry point for the protein domain prediction pipeline. This script
orchestrates the complete workflow from FASTA input to final predictions.
Based on ESM embeddings, transformer-based domain boundary detection, and FFN domain classifier.
Work in progress.       

Pipeline Overview:
1. Input Processing: Convert FASTA to CSV format
2. Domain Boundary Detection: Use transformer model to find domain regions
3. Region Extraction: Extract and prepare domain sequences
4. Classification: Classify domains using pre-trained FFN model
5. Output Generation: Create final CSV with predictions and metadata

Usage:
uv main.py --input sequences.fasta --output results.csv [--model MODEL] [--gpus N]

Dependencies:
- should be auto managed by uv
"""
# -------------------------
# 1. Imports
# -------------------------
import argparse
import os
import pickle
import subprocess
import sys
import shutil
import pandas as pd

from sideScripts import fasta_to_csv

# -------------------------
# 2. Parse Arguments
# -------------------------


def parse_args():
    """
    Function to parse the input arguments.
    This function will be used to parse the input arguments for the whole program.
    1. input_file: Path to the input fasta file.
    2. output_file: Path to the output csv file.
    3. ESM_Model: Model to use for evaluation.
    4. gpus: Number of GPUs to use for evaluation.
    """

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="This script searches Pfam Domains in protein sequences. The current pipeline is:"
        "1) Embedder embedds sequences using ESM model, 2) Transformer searches for domain boundaries, "
        "3) Regions Cutter cuts out the domain regions and re-embeds them, 4) FFN model classifies the domain regions, 5) Final output is a csv file with predicted Pfam ids and domain boundaries."
        "Available flags: --input (required fasta file), --output (required csv file), --model (ESM model name, default: esm2_t33_650M_UR50D), --gpus (number of GPUs, default: 1), --help (show this help message)."
        "Version 0.1 ~ 12 November 2024. Developed by Steven Apelt as part of the Master's Thesis project in the EBB group of the university of Muenster."
    )

    # input arguments, which fasta file to use
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input fasta file."
    )

    # output arguments, where the final csv will be saved
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output csv file."
    )

    # model arguments, which model to use for evaluation, if more models are trained on the same data
    parser.add_argument(
        "--model",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="Model to use for evaluation.",
    )

    # gpus arguments, how many gpus to use for evaluation
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs to use for evaluation."
    )

    # Parse the arguments
    args = parser.parse_args()

    return args.input, args.output, args.model, args.gpus


# -------------------------
# 3. Formatter from fasta to csv
# -------------------------


def Inputter(filepath):
    """
    Function to call the fasta_to_csv script.
    This function will convert fasta files to csv format.
    """
    # Ensure the directory exists
    os.makedirs("tempTest", exist_ok=True)
    # execute the fasta_to_csv function from the sideScripts.fasta_to_csv module
    fasta_to_csv.fasta_to_csv(fasta_path=filepath, csv_path="./tempTest/seqs.csv")


# -------------------------
# 2. Call of Transformer script to search for structured regions via DomainFinder.py
# -------------------------


def Transformer(input_file, ESM_Model, gpus):
    """
    Function to call the Transformer script.
    This function will be used to transform the input data into a format suitable for the model.
    Also, it will call the DomainBoundaryFinder script to find domain boundaries within DomainFinder.py.
    Args:
        input_file (str): Path to the input file. Parsed fasta converted to csv.
        ESM_Model (str): Name of the ESM model to use. Default is esm2_t33_650M_UR50D.
        gpus (int): Number of GPUs to use for evaluation. Parsed by user.
    Returns:
        all_regions (list): List of all predicted domain regions.
        sequence_metadata (list): List of metadata for each sequence.
    """

    print("\nStarting the Transformer script...")

    # fast exit if predicted_domain_regions.pkl already exists
    if os.path.exists("./tempTest/predicted_domain_regions.pkl"):
        print("Found existing predicted_domain_regions.pkl, loading it directly.")
        # Load the predicted domain regions from pickle file
        with open("./tempTest/predicted_domain_regions.pkl", "rb") as f:
            all_regions = pickle.load(f)
        return all_regions

    # Call the main function of DomainBoundaryFinder using torchrun for multi-GPU support
    cmd = [
        "torchrun",
        f"--nproc-per-node={gpus}",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        "DomainFinder.py",
        "--input",
        input_file,
        "--output",
        "./tempTest/regions.csv",
        "--model",
        ESM_Model,
    ]
    # Print the command being run
    cmd_str = " ".join(cmd)
    print(f"Running command: {cmd_str}\n\n")
    os.system(cmd_str)

    # Run the command and stream output in real-time
    process = subprocess.Popen(
        cmd,  # command to run
        stdout=subprocess.PIPE,  # capture stdout
        stderr=subprocess.STDOUT,  # redirect stderr to stdout
        universal_newlines=True,  # text mode
        bufsize=1,  # line-buffered
        env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Force Python to be unbuffered
    )

    # Wait for process to complete and get return code
    return_code = process.wait()

    # Check for errors, then early exit if any
    if return_code != 0:
        print(f"Error: DomainBoundaryFinder exited with return code {return_code}")
        sys.exit(1)

    # print the success message
    print("DomainBoundaryFinder completed successfully")

    # Load the predicted domain regions from pickle file
    with open("./tempTest/predicted_domain_regions.pkl", "rb") as f:
        all_regions = pickle.load(f)

    print("Transformer script completed. Regions found:", len(all_regions), "\n")

    return all_regions


# -------------------------
# 3. Filter and sort the cut out regions to one df
# -------------------------


def Regions_Sorter(all_regions, input_file):
    """
    Function to sort out the domain regions from the sequences.
    This function will take the regions found by the Transformer script and sort them accordingly to the orignal users input file.
    No real filtering is done here, as its already done in the DomainFinder script.
    Its to append the regions with their sequence ID, length, and actual sequence at question.
    Args:
        all_regions (list): List of all predicted domain regions.
        input_file (str): Path to the input file. Parsed fasta converted to csv.
    Returns:
        cut_out_df (pd.DataFrame): DataFrame containing the cut out domain regions.
    """
    print("Selecting possbile domain regions from sequences...")

    # Load the input file as a DataFrame
    df = pd.read_csv(input_file)

    # Create a new list to store the cut out regions
    cut_out_regions = []

    # Iterate through each sequence and its corresponding regions
    for seq_idx, (_, row) in enumerate(df.iterrows()):
        # gather sequence info
        sequence = row["Sequence"]

        # Boundary check
        if seq_idx < len(all_regions):
            # gather regions for this sequence
            regions_for_this_seq = all_regions[seq_idx]

            # Extract each domain from this sequence, if multiple regions exist
            for domain_idx, (start, end) in enumerate(regions_for_this_seq):
                # Ensure valid indices and not out of bounds
                start = max(0, int(start))
                end = min(int(end), len(sequence))

                # Only add if we have a valid region in correct order
                if start < end:
                    # df structure: Sequence_ID, Sequence_Length, Domain_Start, Domain_End, Sequence
                    cut_out_regions.append(
                        {
                            "Sequence_ID": row["ID"],
                            "Sequence_Length": len(sequence),
                            "Domain_Start": start,
                            "Domain_End": end,
                            "Sequence": sequence,
                        }
                    )

    # Convert to DataFrame
    cut_out_df = pd.DataFrame(cut_out_regions)

    # Save to CSV for the next step and as a checkpoint
    cut_out_df.to_csv("./tempTest/cut_out_regions.csv", index=False)

    print(
        f"Extracted {len(cut_out_regions)} domains from initally {len(df)} sequences."
    )

    return cut_out_df


# -------------------------
# 3. FFN classifier to classify the cut out regions via ESM_Embeddings_HP_search.py
# -------------------------


def classifier(gpus):
    """
    Function to classify the sequences using the ESM model.
    This function will take the cut out regions from the transformer and classify them using the pretrained FFN model.
    Calls the ESM_Embeddings_HP_search.py script.
    Args:
        gpus (int): Number of GPUs to use for evaluation. Parsed by user.
    Returns:
        all_predictions (list): List of all predicted Pfam IDs for the cut out regions. Softmax outputs as class indices.
        all_predictions_raw (list): List of all raw scores for the predictions before softmax.
    """
    print("\n\nClassifying domains...")

    # Call the main function of ESM_Embeddings_HP_search using torchrun for multi-GPU support
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        "ESM_Embeddings_HP_search.py",
        "--csv_path",
        "./tempTest/cut_out_regions.csv",
    ]

    # Print the command being run
    cmd_str = " ".join(cmd)
    print(f"Running ESM embeddings command: {' '.join(cmd)}\n\n")
    os.system(cmd_str)

    # Run the command and stream output in real-time
    process = subprocess.Popen(
        cmd,  # command to run
        stdout=subprocess.PIPE,  # capture stdout
        stderr=subprocess.STDOUT,  # redirect stderr to stdout
        universal_newlines=True,  # text mode
        bufsize=1,  # line-buffered
        env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Force Python to be unbuffered
    )

    # Wait for process to complete and get return code
    return_code = process.wait()

    # check for successful completion, if not exit early
    if return_code != 0:
        print(f"Error: ESM_Embeddings_HP_search exited with return code {return_code}")
        sys.exit(1)

    # Load the predictions from the output file generated in ESM_Embeddings_HP_search
    predictions_df = pd.read_csv("./tempTest/predictions.csv")

    # extract predictions and raw scores as lists
    all_predictions = predictions_df["prediction"].tolist()
    all_predictions_raw = predictions_df["raw_scores"].tolist()

    print("Classification completed.\n")

    return all_predictions, all_predictions_raw


# -------------------------
# 4. Downstream Functions to make a final csv with the results. For this the sequence metadata is used saved during embedding.
# -------------------------


def analyze_windowing_structure(cut_df):
    """
    Sub-function to dataframer.
    Goes through the cut_df to find sequences longer than 1000 residues.
    For each such sequence, it calculates how many windows would have been created during embedding.
    This helps to explain any length mismatches between predictions and cut_df rows.
    It it the same logic as in ESM_Embedder for windowing, done to gather
    Args:
        cut_df (pd.DataFrame): DataFrame containing the cut out domain regions.
    Returns:
        long_sequences (list): List of sequences that were windowed, with their details.
    """
    print("\n=== Windowing Analysis ===")

    # Check for sequences longer than 1000 in cut_df
    long_sequences = []
    # loop through cut_df to find sequences longer than 1000
    for idx, row in cut_df.iterrows():
        # gather seq
        sequence = row.get("Sequence", "")
        # & length
        seq_length = len(sequence)
        if seq_length > 1000:
            # Calculate how many windows this sequence would create
            # Using stepsize=500, dimension=1000 (same as in ESM_Embedder) to have same logic
            num_windows = len(range(0, seq_length - 1000 + 1, 500))
            if seq_length % 1000 != 0:
                num_windows += 1  # Final window, same logic if len(seq) not divisible by windowsize

            # make df entry
            long_sequences.append(
                {
                    "cut_df_index": idx,  # index in cut_df
                    "sequence_id": row.get("Sequence_ID", "N/A"),  # sequence ID
                    "length": seq_length,  # length of the sequence
                    "num_windows": num_windows,  # number of windows created
                    "extra_predictions": num_windows
                    - 1,  # -1 because one is always expected (no cut)
                }
            )

    # Summarize total extra predictions expected and prints
    total_extra_predictions = 0
    for seq_info in long_sequences:
        # print(f"  Row {seq_info['cut_df_index']}: ID={seq_info['sequence_id']}, "
        #       f"Length={seq_info['length']}, Windows={seq_info['num_windows']}")
        total_extra_predictions += seq_info["extra_predictions"]
    print(f"Total extra predictions expected from windowing: {total_extra_predictions}")

    return long_sequences


def dataframer(all_predictions, cut_df, output_file):
    """
    Function to create a final DataFrame with the results.
    It uses analyze_windowing_structure to check for length mismatches and fix them.
    The fix is based on the same windowing logic as in ESM_Embedder.py, leading to more rows in the prediction than in the original users input.
    Args:
        all_predictions (list): List of all predicted Pfam IDs for the cut out regions.
        cut_df (pd.DataFrame): DataFrame containing the cut out domain regions.
        output_file (str): Path to the output file. Final csv with results.
    Returns:
        df (pd.DataFrame): Final DataFrame with predictions added.
    """

    # Check for length mismatches between predictions and cut_df, almost always true due to windowing of long sequences
    if len(all_predictions) != len(cut_df):
        print(
            f"Length mismatch: {len(all_predictions)} predictions vs {len(cut_df)} DataFrame rows. Fixing..."
        )

        # Analyze windowing structure to find sequences longer than 1000
        long_sequences = analyze_windowing_structure(cut_df)

        # If long sequences found, proceed to fix
        if long_sequences:
            print(f"Found {len(long_sequences)} sequences longer than 1000 residues")

            # Calculate total extra predictions needed and compare to actual extra predictions
            total_extra_needed = sum(seq["extra_predictions"] for seq in long_sequences)
            actual_extra = len(all_predictions) - len(cut_df)
            print(f"Expected extra predictions: {total_extra_needed}")
            print(f"Actual extra predictions: {actual_extra}")

            # if both match, proceed to create expanded DataFrame. Check passed!
            if total_extra_needed == actual_extra:
                print("✓ The mismatch is explained by sequence windowing!")

                # Create the expanded DataFrame by creating windows from sequences
                expanded_rows = []

                # loop through df
                for idx, row in cut_df.iterrows():
                    # Check if this row corresponds to a long sequence, then break loop
                    matching_long_seq = None
                    for long_seq in long_sequences:
                        if long_seq["cut_df_index"] == idx:
                            matching_long_seq = long_seq
                            break

                    if matching_long_seq:
                        # This sequence needs to be windowed
                        sequence = row["Sequence"]
                        seq_length = len(sequence)
                        dimension = 1000  # Window size as in ESM_Embedder
                        stepsize = 500  # Step size as in ESM_Embedder

                        # Create sliding windows
                        window_start_positions = list(
                            range(0, seq_length - dimension + 1, stepsize)
                        )

                        # Add the last window if needed as in ESM_Embedder
                        if seq_length % dimension != 0 and seq_length > dimension:
                            window_start_positions.append(seq_length - dimension)

                        # Create a row for each window and all needed data
                        for window_idx, start_pos in enumerate(window_start_positions):
                            end_pos = start_pos + dimension
                            window_seq = sequence[start_pos:end_pos]

                            # Create a copy of the row for this window
                            row_copy = row.copy()
                            row_copy["Sequence"] = window_seq                   # Update sequence to windowed sequence
                            row_copy["Sequence_ID"] = (                         # Update Sequence_ID to indicate window for all windows
                                f"{row_copy['Sequence_ID']}_{window_idx + 1}"
                            )
                            row_copy["Window_Start_Pos"] = int(start_pos)       # Add window start position
                            row_copy["Window_End_Pos"] = int(end_pos)           # Add window end position

                            # Adjust domain coordinates for this window
                            domain_start = row["Domain_Start"]                  # Original global domain start
                            domain_end = row["Domain_End"]                      # Original global domain end

                            # Calculate domain positions relative to window, to check for overlap and delete if not found
                            window_domain_start = max(0, domain_start - start_pos)
                            window_domain_end = min(dimension, domain_end - start_pos)

                            # Set domain coordinates even if not overlapping. -1 if no overlap, domain not in window
                            row_copy["Domain_Start"] = (
                                window_domain_start
                                if window_domain_end > 0
                                and window_domain_start < dimension
                                else -1
                            )
                            row_copy["Domain_End"] = (
                                window_domain_end
                                if window_domain_end > 0
                                and window_domain_start < dimension
                                else -1
                            )

                            # Add all windows
                            expanded_rows.append(row_copy)
                    else:
                        # Normal sequence, add as is with window positions as 0 and len(seq)
                        expanded_rows.append(row)
                        row["Window_Start_Pos"] = 0
                        row["Window_End_Pos"] = len(row["Sequence"])

                # Create new DataFrame with expanded rows
                df = pd.DataFrame(expanded_rows).reset_index(drop=True)

            # If not, exit with error as the mismatch cant be explained by windowing
            else:
                print(
                    "✗ The windowing doesn't fully explain the mismatch.\nPlease check the input data and predictions carefully.\nExiting to avoid incorrect results."
                )
                sys.exit(1)

        # No long sequences found, exit with error as the mismatch cant be explained by windowing
        else:
            print(
                "No sequences longer than 1000 found. Using original DataFrame and truncating predictions.\nExiting to avoid incorrect results."
            )
            sys.exit(1)

    # If lengths match, proceed as is
    else:
        print("Lengths match perfectly!")
        df = cut_df.copy()

    # Final verification for safety
    if len(all_predictions) != len(df):
        raise ValueError(
            f"Still have length mismatch: {len(all_predictions)} predictions vs {len(df)} DataFrame rows"
        )

    # convert all_predictions to the actual pfam IDs via the selected_pfam_ids.txt file of the used model (gonna be set for final model)
    with open("./temp/selected_pfam_ids_1000.txt", "r") as f:
        pfam_ids = [line.strip() for line in f.readlines()]
        all_predictions = [
            pfam_ids[pred - 1] if pred > 0 and pred < len(pfam_ids) else "Unknown"
            for pred in all_predictions
        ]

    # Add predictions to DataFrame
    df.insert(2, "Prediction", all_predictions)

    # Mark sequences with no domain overlap as "No Domain"
    for idx, row_copy in df.iterrows():
        if row_copy["Domain_Start"] == -1 and row_copy["Domain_End"] == -1:
            df.at[idx, "Prediction"] = "No Domain"

    # Reorder columns to put Window_Start_Pos and Window_End_Pos after Sequence_Length
    if "Window_Start_Pos" in df.columns and "Window_End_Pos" in df.columns:
        df["Window_Start_Pos"] = df["Window_Start_Pos"].astype(int)
        df["Window_End_Pos"] = df["Window_End_Pos"].astype(int)

        # Get all column names
        cols = df.columns.tolist()

        # Remove Window position columns from current positions
        cols.remove("Window_Start_Pos")
        cols.remove("Window_End_Pos")

        # Find position of Sequence_Length
        seq_length_pos = cols.index("Sequence_Length")

        # Insert Window position columns after Sequence_Length
        cols.insert(seq_length_pos + 1, "Window_Start_Pos")
        cols.insert(seq_length_pos + 2, "Window_End_Pos")

        # Reorder DataFrame columns
        df = df[cols]

    # Save to CSV
    df.to_csv(output_file, index=False)

    return df


# -------------------------
# Main function
# -------------------------


def main(input_file, output_file, ESM_Model, gpus):
    """
    Main function to run the whole pipeline.
    This function will call all the other functions in the correct order to run the whole pipeline.
    It is sorted by the steps of the pipeline from input to output.
    Args:
        input_file (str): Path to the input fasta file.
        output_file (str): Path to the output csv file.
        ESM_Model (str): Name of the ESM model to use.
        gpus (int): Number of GPUs to use for evaluation.
    
    """

    # --------------------------------------------------------------------

    # Call the Inputter function to convert fasta to csv
    if input_file.endswith(".fa") or input_file.endswith(".fasta"):
        Inputter(input_file)
        # set new input file to the converted csv
        input_file = "./tempTest/seqs.csv"
    # if not fasta or fasta format, exit with error stating unsupported format
    else:
        print(
            f"Unsupported input file format: {input_file}. Please provide a .fa or .fasta."
        )
        sys.exit(1)

    # --------------------------------------------------------------------

    # Call the Transformer script to search for domain boundaries
    all_regions = Transformer(input_file, ESM_Model, gpus)

    # --------------------------------------------------------------------

    # Call Region sorter script to sort out domain regions
    cut_df = Regions_Sorter(all_regions, input_file)

    # --------------------------------------------------------------------

    # Call classifier script to classify the domains
    all_predictions, _ = classifier(gpus)

    # --------------------------------------------------------------------

    # Create final DataFrame with results
    final_df = dataframer(all_predictions, cut_df, output_file)

    # --------------------------------------------------------------------

    # Final print & clean up temporary files
    print(
        f"\n\nPipeline completed with {len(final_df)} predictions.\nResults saved to {output_file}.\nRemoving temp files...\n\n"
    )
    tempfolder = "./tempTest"
    embedding_dir_scratch = "/global/scratch2/sapelt/tempTest/embeddings"
    if os.path.exists(tempfolder):
        shutil.rmtree(tempfolder)
        shutil.rmtree(embedding_dir_scratch)

####################################################################
# Entry point, parse args and call main
input_file, output_file, ESM_Model, gpus = parse_args()
if __name__ == "__main__":
    main(input_file, output_file, ESM_Model, gpus)
