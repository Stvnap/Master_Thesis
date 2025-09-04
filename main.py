"""
WORK IN PROGRESS, NOT WORKING YET
Main file for the Program to run.
Idea is that all scripts are called from here.
All inputs and outputs are managed here.
This file is the entry point for the program.
"""

########### 
# FIX IF MULTIPLE WINDOWS ARE USED
# THE START AND END PREDICTIONS ARE NOT CORRECTLY MAPPED TO THE CUT OUT REGIONS
###########


import argparse
import os
import pickle
import subprocess
import sys

import pandas as pd

from sideScripts import fasta_to_csv

# from ESM_Embedder import ESM_Dataset
#######################################
# 0. Parser for the input arguments
#######################################


def parse_args():
    """
    Function to parse the input arguments.
    This function will be used to parse the input arguments for the program.
    """

    parser = argparse.ArgumentParser(
        description="Main script for the ESM evaluation program."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input fasta file."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output csv file."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="Model to use for evaluation.",
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs to use for evaluation."
    )
    parser.add_argument(
        "--vram", type=int, default=8056, help="VRAM available for evaluation."
    )

    args = parser.parse_args()

    return args.input, args.output, args.model, args.gpus, args.vram


######################################
# 1. Formatter from fasta to csv
######################################
def Inputter(filepath):
    """
    Function to call the fasta_to_csv script.
    This function will convert fasta files to csv format.
    """

    os.makedirs("tempTest", exist_ok=True)  # Ensure the directory exists
    fasta_to_csv.fasta_to_csv(fasta_path=filepath, csv_path="./tempTest/seqs.csv")


######################################
# 2. Call of Transformer script to search for structured regions | NOT YET CREATED
######################################


def Transformer(input_file, ESM_Model, gpus, vram):
    """
    Function to call the Transformer script.
    This function will be used to transform the input data into a format suitable for the model.
    """

    # Call the main function of DomainBoundaryFinder using torchrun
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
        "--vram",
        str(vram),
    ]

    print(f"Running command: {' '.join(cmd)}")

    # Run the command and stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    # Print output in real-time
    for line in process.stdout:
        print(line.strip())

    # Wait for process to complete and get return code
    return_code = process.wait()

    if return_code != 0:
        print(f"Error: DomainBoundaryFinder exited with return code {return_code}")
        sys.exit(1)

    print("DomainBoundaryFinder completed successfully")

    # Load the predicted domain regions from pickle file
    with open("./tempTest/predicted_domain_regions.pkl", "rb") as f:
        all_regions = pickle.load(f)
        sequence_metadata = pickle.load(f)

        
    # print("sequence_metadata:",sequence_metadata)

    return all_regions, sequence_metadata  # Return both all_regions and sequence_metadata


######################################
# 3. Cut out the domain regions from the sequences
######################################


def Regions_Cutter(all_regions, input_file):
    """
    Function to cut out the domain regions from the sequences.
    This function will take the regions found by the Transformer script and cut them out from the sequences.
    """
    import pandas as pd

    # Load the input file
    df = pd.read_csv(input_file)

    # Create a new DataFrame to store the cut out regions
    cut_out_regions = []

    # Iterate through each sequence and its corresponding regions
    for seq_idx, (index, row) in enumerate(df.iterrows()):
        sequence = row["Sequence"]

        # Check if we have regions for this sequence
        if seq_idx < len(all_regions):
            regions_for_this_seq = all_regions[seq_idx]

            # Extract each domain from this sequence
            for domain_idx, (start, end) in enumerate(regions_for_this_seq):
                # Ensure valid indices
                start = max(0, int(start))
                end = min(int(end), len(sequence))

                # Only add if we have a valid region
                if start < end:
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

    # Save to CSV
    cut_out_df.to_csv("./tempTest/cut_out_regions.csv", index=False)

    print(f"Extracted {len(cut_out_regions)} domains from {len(df)} sequences.")

    return cut_out_df  # Return the actual cut_out_df, not the original df


######################################
# 3. Predicter_for_ESM.py type script, that classifies the sequences | Potentially with domain boundaries detection loop
######################################


def classifier(gpus):
    """
    Function to classify the sequences using the ESM model.
    This function will take the cut out regions and classify them using the ESM model.
    """
    # Call the main function of ESM_Embeddings_HP_search using torchrun
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        "ESM_Embeddings_HP_search.py",
        "--csv_path",
        "./tempTest/cut_out_regions.csv",
    ]

    print(f"Running ESM embeddings command: {' '.join(cmd)}")

    # Run the command and stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    # Print output in real-time
    for line in process.stdout:
        print(line.strip())

    # Wait for process to complete and get return code
    return_code = process.wait()

    if return_code != 0:
        print(f"Error: ESM_Embeddings_HP_search exited with return code {return_code}")
        sys.exit(1)

    print("ESM_Embeddings_HP_search completed successfully")

    # Load the predictions from the output file generated by ESM_Embeddings_HP_search
    predictions_df = pd.read_csv("./tempTest/predictions.csv")
    # print(predictions_df)

    all_predictions = predictions_df["prediction"].tolist()
    all_predictions_raw = predictions_df["raw_scores"].tolist()

    return all_predictions, all_predictions_raw


######################################
# 4. Downstream script to make a final csv with the results
######################################


def analyze_windowing_structure(sequence_metadata, cut_df):
    """
    Analyze the windowing structure to understand the mapping between predictions and DataFrame rows
    """
    print("\n=== Windowing Analysis ===")
    
    # Check for sequences longer than 1000 in cut_df
    long_sequences = []
    for idx, row in cut_df.iterrows():
        sequence = row.get('Sequence', '')
        seq_length = len(sequence)
        if seq_length > 1000:
            # Calculate how many windows this sequence would create
            # Using stepsize=500, dimension=1000 (same as in ESM_Embedder)
            num_windows = len(range(0, seq_length - 1000 + 1, 500))
            if seq_length % 1000 != 0:
                num_windows += 1  # Final window
            
            long_sequences.append({
                'cut_df_index': idx,
                'sequence_id': row.get('Sequence_ID', 'N/A'),
                'length': seq_length,
                'num_windows': num_windows,
                'extra_predictions': num_windows - 1  # -1 because one is expected
            })
    
    print("Sequences longer than 1000 residues in cut_df:")
    total_extra_predictions = 0
    for seq_info in long_sequences:
        print(f"  Row {seq_info['cut_df_index']}: ID={seq_info['sequence_id']}, "
              f"Length={seq_info['length']}, Windows={seq_info['num_windows']}")
        total_extra_predictions += seq_info['extra_predictions']
    
    print(f"Total extra predictions expected from windowing: {total_extra_predictions}")
    
    return long_sequences


def dataframer(all_predictions, cut_df, output_file, sequence_metadata):
    """
    Function to create a final DataFrame with the results.
    This function will take the predictions and create a final DataFrame with the results.
    """

    # print(f"\n=== Dataframer Debug Info ===")
    # print(f"Number of predictions: {len(all_predictions)}")
    # print(f"Number of DataFrame rows: {len(cut_df)}")

    # Verify lengths match
    if len(all_predictions) != len(cut_df):
        print(f"Length mismatch: {len(all_predictions)} predictions vs {len(cut_df)} DataFrame rows")
        
        long_sequences = analyze_windowing_structure(sequence_metadata, cut_df)
        
        if long_sequences:
            print(f"\nFound {len(long_sequences)} sequences longer than 1000 residues")
            
            # Calculate total extra predictions needed
            total_extra_needed = sum(seq['extra_predictions'] for seq in long_sequences)
            actual_extra = len(all_predictions) - len(cut_df)
            
            print(f"Expected extra predictions: {total_extra_needed}")
            print(f"Actual extra predictions: {actual_extra}")
            
            if total_extra_needed == actual_extra:
                print("✓ The mismatch is explained by sequence windowing!")
                
                # Create the expanded DataFrame by duplicating rows for windowed sequences
                expanded_rows = []
                
                for idx, row in cut_df.iterrows():
                    # Check if this row corresponds to a long sequence
                    matching_long_seq = None
                    for long_seq in long_sequences:
                        if long_seq['cut_df_index'] == idx:
                            matching_long_seq = long_seq
                            break
                    
                    if matching_long_seq:
                        # This sequence was windowed, duplicate the row
                        num_windows = matching_long_seq['num_windows']
                        for window_idx in range(num_windows):
                            row_copy = row.copy()
                            # Optionally, you can add a suffix to indicate this is a windowed prediction
                            if window_idx > 0:  # Only add suffix for additional windows
                                row_copy['Sequence_ID'] = f"{row_copy['Sequence_ID']}_{window_idx+1}"
                            expanded_rows.append(row_copy)
                            print(f"  Added window {window_idx} for sequence {row_copy.get('Sequence_ID', 'N/A')}")
                    else:
                        # Normal sequence, add as-is
                        expanded_rows.append(row)
                
                # Create new DataFrame with expanded rows
                df = pd.DataFrame(expanded_rows).reset_index(drop=True)
                print(f"Expanded DataFrame to {len(df)} rows to match {len(all_predictions)} predictions")
                
            else:
                print("✗ The windowing doesn't fully explain the mismatch")
                print("Using original DataFrame and truncating predictions")
                df = cut_df.copy()
                all_predictions = all_predictions[:len(df)]
        else:
            print("No sequences longer than 1000 found. Using original DataFrame and truncating predictions")
            df = cut_df.copy()
            all_predictions = all_predictions[:len(df)]
    else:
        print("Lengths match perfectly!")
        df = cut_df.copy()

    # Final verification
    if len(all_predictions) != len(df):
        raise ValueError(f"Still have length mismatch: {len(all_predictions)} predictions vs {len(df)} DataFrame rows")


    # convert all_predictions to the actual pfam IDs
    with open("./temp/selected_pfam_ids_1000.txt", "r") as f:
        pfam_ids = [line.strip() for line in f.readlines()]
        all_predictions = [pfam_ids[pred] if pred < len(pfam_ids) else "Unknown" for pred in all_predictions]

    # Add predictions to DataFrame
    df.insert(2, "Prediction", all_predictions)

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"Final results saved to {output_file}")
    print(f"Total domains processed: {len(df)}")

    return df


#######################################################################


def main(input_file, output_file, ESM_Model, gpus, vram):

    # Call the Inputter function to convert fasta to csv if input is a fasta file
    if input_file.endswith(".fa") or input_file.endswith(".fasta"):
        Inputter(input_file)
        input_file = (
            "./tempTest/seqs.csv" 
        )
    else:
        print(
            f"Unsupported input file format: {input_file}. Please provide a .fa or .fasta."
        )
        sys.exit(1)

    # --------------------------------------------------------------------

    print("Starting the Transformer script...")

    all_regions,sequence_metadata = Transformer(input_file, ESM_Model, gpus, vram)

    print("Transformer script completed. Regions found:", all_regions,"\n")

    # --------------------------------------------------------------------

    # Call Cutter script to cut out domain regions
    print("Cutting out regions from sequences...")
    cut_df = Regions_Cutter(all_regions, input_file) 

    # --------------------------------------------------------------------

    # Call classifier script to classify the domains
    print("Classifying domains...")
    all_predictions, all_predictions_raw = classifier(gpus)  

    print("Classification completed.\n")

    # --------------------------------------------------------------------

    # Create final DataFrame with results
    final_df = dataframer(all_predictions, cut_df, output_file,sequence_metadata)

    # --------------------------------------------------------------------

    # clean up temporary files
    print(f"\n\nPipeline completed. Results saved to {output_file}\n\n")
    temp_files = [
        "./tempTest/seqs.csv",
        "./tempTest/regions.csv",
        "./tempTest/cut_out_regions.csv",
        "./tempTest/predictions.csv",
        "./tempTest/predicted_domain_regions.pkl",
    ]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            # print(f"Removed temporary file: {temp_file}")
        else:
            # print(f"Temporary file not found: {temp_file}")
            continue
    embeddings_dir = "./tempTest/embeddings"
    if os.path.exists(embeddings_dir):
        import shutil

        shutil.rmtree(embeddings_dir)
        # print(f"Removed embeddings cache directory: {embeddings_dir}")

    return final_df


####################################################################
input_file, output_file, ESM_Model, gpus, vram = parse_args()
if __name__ == "__main__":
    main(input_file, output_file, ESM_Model, gpus, vram)
