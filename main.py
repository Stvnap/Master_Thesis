"""
WORK IN PROGRESS, NOT WORKING YET
Main file for the Program to run.
Idea is that all scripts are called from here.
All inputs and outputs are managed here.
This file is the entry point for the program.
"""

import os
import sys
import subprocess
from sideScripts import fasta_to_csv 
import pandas as pd
import pickle

# from ESM_Embedder import ESM_Dataset
#######################################
# 0. Parser for the input arguments
#######################################

def parse_args():
    """
    Function to parse the input arguments.
    This function will be used to parse the input arguments for the program.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Main script for the ESM evaluation program.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input fasta file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output csv file.")
    
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D", help="Model to use for evaluation.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for evaluation.")
    parser.add_argument("--vram", type=int, default=8056, help="VRAM available for evaluation.")

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
    fasta_to_csv.fasta_to_csv(fasta_path=filepath, csv_path = './tempTest/seqs.csv')



######################################
# 2. Call of Transformer script to search for structured regions | NOT YET CREATED
######################################


def Transformer(input_file, ESM_Model, gpus, vram):
    """
    Function to call the Transformer script.
    This function will be used to transform the input data into a format suitable for the model.
    """
    
    # Call the main function of DomainFinder using torchrun
    cmd = [
        "torchrun",
        f"--nproc-per-node={gpus}",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        "DomainFinder.py",
        "--input", input_file,
        "--output", "./tempTest/regions.csv",
        "--model", ESM_Model,
        "--vram", str(vram)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command and stream output in real-time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
    # Print output in real-time
    for line in process.stdout:
        print(line.strip())
    
    # Wait for process to complete and get return code
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Error: DomainFinder exited with return code {return_code}")
        sys.exit(1)
    
    print("DomainFinder completed successfully")
    


    # Load the predicted domain regions from pickle file
    with open('./tempTest/predicted_domain_regions.pkl', 'rb') as f:
        all_regions = pickle.load(f)
    
    return all_regions

    # Assuming DomainFinder outputs results to a file, load the results
    # You'll need to modify this based on how DomainFinder outputs its results
    # all_regions = []  # Replace with actual loading logic



# QUESTION: Maybe resuse the embeddings directly and pool the per residue embeddings from the cut out sequence?





######################################
# 3. Cut out the domain regions from the sequences
######################################

##### START ENDS ARE BUGGY, NEEDS TO BE FIXED !!!! ################

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
        sequence = row['Sequence']
        
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
                    # Extract the actual domain sequence
                    domain_sequence = sequence[start:end]
                    
                    cut_out_regions.append({
                        'Sequence_ID': row['ID'],
                        'Sequence': sequence,
                        'Domain_Start': start,
                        'Domain_End': end,
                    })

    # Convert to DataFrame
    cut_out_df = pd.DataFrame(cut_out_regions)

    # Save to CSV
    cut_out_df.to_csv('./tempTest/cut_out_regions.csv', index=False)
    
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
        "--csv_path", "./tempTest/cut_out_regions.csv"
    ]
    
    print(f"Running ESM embeddings command: {' '.join(cmd)}")
    
    # Run the command and stream output in real-time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
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
    predictions_df = pd.read_csv('./tempTest/predictions.csv')
    # print(predictions_df)


    all_predictions = predictions_df['prediction'].tolist()
    all_predictions_raw = predictions_df['raw_scores'].tolist()

    return all_predictions, all_predictions_raw

######################################
# 4. Downstream script to make a final csv with the results
######################################


def dataframer(all_predictions, cut_df, output_file):
    """
    Function to create a final DataFrame with the results.
    This function will take the predictions and create a final DataFrame with the results.
    """
    import pandas as pd

    # Use the cut_df that was passed in (which is the cut_out_regions DataFrame)
    df = cut_df.copy()
    
    
    # print(all_predictions)
    # print(df)
    
    # Verify lengths match (they should now)
    if len(all_predictions) != len(df):
        raise ValueError(f"Number of predictions ({len(all_predictions)}) must match number of domains ({len(df)})")

    # If we reach this point, the lengths match
    print("Lengths match!")

    # Add predictions to DataFrame
    df['Prediction'] = all_predictions

    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Final results saved to {output_file}")
    print(f"Total domains processed: {len(df)}")
    
    return df






#######################################################################

def main(input_file, output_file, ESM_Model, gpus, vram):

    # Parse input arguments
    
    # Call the Inputter function to convert fasta to csv if input is a fasta file
    if input_file.endswith('.fa') or input_file.endswith('.fasta'):
        Inputter(input_file)
        input_file = './tempTest/seqs.csv'  # Update input file to the converted CSV path
    elif input_file.endswith('.csv'):
        # If input is already a CSV, we can directly use it
        print(f"Input file is already in CSV format: {input_file}")
    else:
        print(f"Unsupported input file format: {input_file}. Please provide a .fa, .fasta, or .csv file.")
        sys.exit(1)

    # --------------------------------------------------------------------

    # Here you would call the Transformer and Predicter scripts
    # For now, we will just print a message
    print("Starting the Transformer script...")

    all_regions = Transformer(input_file, ESM_Model, gpus, vram)

    print("Transformer script completed. Regions found:", all_regions)

    # --------------------------------------------------------------------

    # Call Cutter script to cut out domain regions
    print("Cutting out regions from sequences...")
    cut_df = Regions_Cutter(all_regions, input_file)  # This returns the cut_out_df now

    # --------------------------------------------------------------------

    # Call classifier script to classify the domains
    print("Classifying domains...")
    all_predictions, all_predictions_raw = classifier(gpus)  # Pass gpus parameter

    print("Classification completed.")

    # --------------------------------------------------------------------

    # Create final DataFrame with results
    print("Creating final results...")
    final_df = dataframer(all_predictions, cut_df, output_file)

    print(f"Pipeline completed. Results saved to {output_file}")
    
    return final_df




####################################################################
input_file, output_file, ESM_Model, gpus, vram = parse_args()
if __name__ == "__main__":
    main(input_file, output_file, ESM_Model, gpus, vram )