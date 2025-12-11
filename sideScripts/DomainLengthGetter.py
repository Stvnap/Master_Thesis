"""
DomainLengthGetter.py

Usd to get the lengths of sequences from a target dataframe and plot their distribution.

Table of Contents:
=========================
1. DomainLengthGetter
2. main
"""

# -------------------------
# Imports & Globals
# -------------------------
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

INPUT = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/NO_DOMAINS_SWISSPROT.csv"
OUTPUT = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/domain_lengths.csv"


class DomainLengthGetter:
    """
    basic class to get domain lengths from a dataframe
        get_domain_lengths: returns a pandas series with the lengths of the sequences in the dataframe
    """

    def __init__(self, df):
        self.df = df

    def get_domain_lengths(self):
        # Calculate domain lengths
        self.length = self.df["Sequence"].str.len()
        return self.length


def main():
    # Load dataframe
    df = pd.read_csv(INPUT)

    # init DomainLengthGetter and get lengths
    getter = DomainLengthGetter(df)

    # get domain lengths
    lengths = getter.get_domain_lengths()

    # save lengths to a file with a clear column name
    pd.DataFrame({"Length": lengths}).to_csv(OUTPUT, index=False)
    print(f"Domain lengths saved to '{OUTPUT}'")

    # fig of length distribution
    plt.figure(figsize=(8, 5))
    plt.hist(
        lengths,
        bins=100,
        color=cm.get_cmap("tab10").colors[0],
        alpha=0.7,
        density=False,
    )
    plt.title("Length Distribution of non Pfam Sequences from SwissProt")
    plt.xlabel("Domain Length in residues")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/domain_length_distribution.png"
    )


######################################
if __name__ == "__main__":
    main()
