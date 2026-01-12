"""
csv_to_fasta.py

Converts a CSV file containing sequence data into FASTA format. Globals define input/output paths and column names for ID and sequence.

Table of Contents:
=========================
1. csv_to_fasta
"""

# -------------------------
# Imports & Globals
# -------------------------
import pandas as pd

DF_INPUT = "/global/students/research/sapelt/Masters/MasterThesis/Dataframes/v3/SampledEntriesCompleteProteins_MAX.csv"
FASTA_OUTPUT = "/global/students/research/sapelt/Masters/MasterThesis/Dataframes/v3/SampledEntriesCompleteProteins_MAX.fasta"
ID_COL = "id"
SEQ_COL = "Sequence"


# -------------------------
# Function
# -------------------------
def csv_to_fasta():
    """
    Converts the given CSV file to a FASTA file.
    Expects the CSV to have columns 'id' and 'Sequence'.
    """

    # Read the CSV file
    df = pd.read_csv(DF_INPUT)

    # Open the FASTA output file for writing
    with open(FASTA_OUTPUT, "w") as fasta_file:
        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            # Get identifier and sequence
            identifier = row[ID_COL]
            sequence = row[SEQ_COL]

            # Write identifier with '>' prefix and then \n
            fasta_file.write(f">{identifier}\n")

            # Write sequence and wrap the sequence every 60 characters (FASTA convention)
            for i in range(0, len(sequence), 60):
                fasta_file.write(sequence[i : i + 60] + "\n")


#################################
if __name__ == "__main__":
    csv_to_fasta()
