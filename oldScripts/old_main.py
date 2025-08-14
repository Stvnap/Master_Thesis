###################################################################################################################################
"""
Main file for Usage purpuses. Input file with fasta sequences is used.
Outputed is a final
"""

import argparse

import Dataset_preprocess
import Predicter_for_normal_usage
from Dataset_preprocess import dimension_positive

#################################################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Trained model to predict the domain PF00177. Input file must be a fasta file. Output must be defined with a basename and .csv file extension"
    )
    parser.add_argument(
        "-i", "--input-fasta", required=True, help="Path to your domains FASTA"
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to your final output file. Use full name to work properly. Save format is .csv (example: './out1.csv')",
    )
    parser.add_argument(
        "-s",
        "--stepsize",
        type=int,
        default=10,
        help="Sliding window step size (default: 10)",
    )

    args = parser.parse_args()

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

    ##################################################################################################

    print("Start Data Prediction")

    df_path = dataset.seqarray_final
    model_path = "./models/my_modelnewlabeling.keras"  # model data
    Predicter_for_normal_usage.Predicter_pipeline(
        model_path,
        df_path,
        outfilepath=args.output,
        flank_size=30,
        step=10,
        batch_size=512,
    )


##################################################################################################

if __name__ == "__main__":
    main()


# /global/research/students/sapelt/Masters/domains_PF00177.fa
# ./Outtest.csv
