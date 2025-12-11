"""
old_main.py

Old main file for Usage purpuses. Was designed as first usable program for domain PF00177 prediction. Was quickly abandoned and not stress tested further.
Input file with fasta sequences is used. Outputed is a final
"""

# -------------------------
# Imports & globals
# -------------------------

import argparse

import Dataset_preprocess
import Predicter_for_normal_usage
from Dataset_preprocess import dimension_positive

MODEL_PATH = "./models/my_modelnewlabeling.keras"  # model data

# -------------------------
# main
# -------------------------


def main():
    """
    Parses arguments and runs the prediction pipeline.
    1. Loads input FASTA file.
    2. Creates dataset with sliding windows.
    3. Runs prediction using a pre-trained model.
    4. Saves output to specified CSV file.
    """
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Trained model to predict the domain PF00177. Input file must be a fasta file. Output must be defined with a basename and .csv file extension"
    )
    # input fastsa
    parser.add_argument(
        "-i", "--input", required=True, help="Path to your domains FASTA"
    )

    # output file
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to your final output file. Use full name to work properly. Save format is .csv (example: './out1.csv')",
    )
    # stepsize for sliding window
    parser.add_argument(
        "-s",
        "--stepsize",
        type=int,
        default=10,
        help="Sliding window step size (default: 10)",
    )

    # gather arguments
    args = parser.parse_args()

    # Data Preprocessing
    print("Loading in")
    fasta = Dataset_preprocess.DomainProcessing(args.input_fasta)
    seqarray_clean_rnd_trembl = fasta.load_in_Dataset()
    print("Starting data creation")
    dataset = Dataset_preprocess.databaseCreater(
        seqarray_clean_rnd_trembl,
        dimension_positive=dimension_positive,
        stepsize=args.stepsize,
    )
    print("All done creating Training dataset")


    # Start Prediction
    print("Start Data Prediction")
    df_path = dataset.seqarray_final
    Predicter_for_normal_usage.Predicter_pipeline(
        MODEL_PATH, # model path
        df_path,    # data path to be used 
        outfilepath=args.output,    # output path
        flank_size=30,  # flank size for prediction
        step=10,    # step size for sliding window
        batch_size=512, # batch size for prediction
    )


##################################################################################################

if __name__ == "__main__":
    main()

# paths saved to copy for testing
# /global/research/students/sapelt/Masters/domains_PF00177.fa
# ./Outtest.csv
