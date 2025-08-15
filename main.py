"""
WORK IN PROGRESS, NOT WORKING YET
Main file for the Program to run.
Idea is that all scripts are called from here.
All inputs and outputs are managed here.
This file is the entry point for the program.
"""

import os
import sys
from sidescript import fasta_to_csv 


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
    parser.add_argument("--num_classes", type=int, default=0MAX_PLACEHOLDER, help="Number of classes for classification.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for evaluation.")
    parser.add_argument("--vram", type=int, default=8056, help="VRAM available for evaluation.")

    args = parser.parse_args()
    
    return args.input, args.output, args.model, args.num_classes, args.gpus, args.vram




######################################
# 1. Formatter from fasta to csv 
######################################
def Inputter(filepath):
    """
    Function to call the fasta_to_csv script.
    This function will convert fasta files to csv format.
    """ 
    fasta_to_csv.fasta_to_csv(file_path="tempTest/input.fasta", csv_path="tempTest/output.csv")


######################################
# 2. Call of Transformer script to search for structured regions | NOT YET CREATED
######################################



# QUESTION: Maybe resuse the embeddings directly and pool the per residue embeddings from the cut out sequence?



######################################
# 3. Predicter_for_ESM.py type script, that classifies the sequences | Potentially with domain boundaries detection loop
######################################



######################################
# 4. Downstream script to make a final csv with the results
######################################








#######################################################################

def main():

    # Parse input arguments
    input_file, output_file, ESM_Model, num_classes, gpus, vram = parse_args()
    
    # Call the Inputter function to convert fasta to csv if input is a fasta file
    if input_file.endswith('.fa') or input_file.endswith('.fasta'):
        Inputter(input_file)
    
    # Here you would call the Transformer and Predicter scripts
    # For now, we will just print a message
    print("Transformer and Predicter scripts would be called here.")




    
    # Final output handling (e.g., saving results to output_file)
    print(f"Results would be saved to {output_file}.")













####################################################################
if __name__ == "__main__":
    main()