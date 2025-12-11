"""
LengthHistoPlotter.py

This script is a modified version of early Dataset_prepriocess_*.py scripts. It was used to create faster histogram plots of sequence length distributions for various protein domain datasets used in earlier models. It reads FASTA files, computes sequence lengths, and plots histograms to visualize the distributions.

Table of contents:
=========================
1. DomainProcessing class
2. _opener()
3. seq_array_returner()
4. len_finder()
5. distribution_finder_and_cleaner()
6. dimension_finder()
7. Main
"""

# -------------------------
# Imports and Globals
# -------------------------
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import time
from matplotlib import cm
import seaborn as sns

DOMAIN_PATH_1 = "/global/research/students/sapelt/Masters/domains_PF00177.fa"
DOMAIN_PATH_2 = "/global/research/students/sapelt/Masters/domains_PF00210.fa"
DOMAIN_PATH_3 = "/global/research/students/sapelt/Masters/domains_PF00079.fa"
DOMAIN_PATH_4 = "/global/research/students/sapelt/Masters/domains_PF00080.fa"
DOMAIN_PATH_5 = "/global/research/students/sapelt/Masters/domains_PF00118.fa"
DOMAIN_PATH_6 = "/global/research/students/sapelt/Masters/domains_PF00162.fa"
DOMAIN_PATH_7 = "/global/research/students/sapelt/Masters/domains_uniprot_sprot.fa"

# -------------------------
# DomainProcessing Class
# -------------------------


class DomainProcessing:
    """
    Light weight Domain Processing class to read FASTA files, compute sequence lengths, and plot histograms only.
    Args:
        domain_path (str): Path to the FASTA file of the protein domain
    """

    def __init__(self, domain_path):
        self.domain_path = domain_path
        # init lists
        self.boundaries_all = []
        self.sequence_all = []
        self.id_all = []
        # open file and read sequences
        self._opener()
        # compute sequence lengths
        seqlen = self.len_finder()
        # get plots and stats
        seqarray_clean, seqarraylen_clean, normaltest = (
            self.distribution_finder_and_cleaner(seqlen, domain_path.split("/")[-1])
        )
        # prints
        print("len:", len(seqarray_clean), "max:", max(seqarraylen_clean))
        print(normaltest)
        # get targeted dimension, 0.65 quantile
        dimension_positive = self.dimension_finder(seqarraylen_clean)
        print("targeted dimension", dimension_positive)

    def _opener(self):
        """
        Opens the FASTA file and reads sequences, IDs, and boundaries.
        """
        with open(self.domain_path, "r") as file:
            # init list
            sequence = []

            # read lines
            for line in file:
                line = line.strip()
                id_whole = line
                # check for header line
                if line.startswith(">"):
                    sequence_joined = "".join(sequence)
                    self.sequence_all.append(sequence_joined)

                    # extract boundaries using regex
                    try:
                        boundaries = re.search(r"_(\d+-\d+)", line)
                        boundaries = boundaries.group(1)
                        self.boundaries_all.append(boundaries)
                    # handle cases where boundaries are not found
                    except AttributeError:
                        continue

                    # append full ID
                    self.id_all.append(id_whole)

                    # get whole id
                    self.id_whole = line

                    # reset sequence list for next entry
                    sequence = []
                # if not header line, append to sequence
                else:
                    sequence.append(line)
            # # debug prints
            # print(id_all)
            # print(sequence_all)
            # print(len(id_all))
            # print(len(sequence_all))
            # print(len(boundaries_all))

    def seq_array_returner(self):
        """
        get seqs into a dataframe with column name "Sequences"
        """
        self.sequence_all = pd.DataFrame(self.sequence_all)
        self.sequence_all.columns = ["Sequences"]
        return self.sequence_all

    def len_finder(self):
        # simple length finder for all sequences
        return [len(seq) for seq in self.sequence_all]

    def distribution_finder_and_cleaner(self, seqlen, domain_name):
        """
        Hist plotter for sequence length distribution and Shapiro-Wilk normality test. Option for outlier removal commented out.
        Args:
            seqlen (list): List of sequence lengths
            domain_name (str): Name of the protein domain for plot title
        Returns:
            seqarray_clean (pd.DataFrame): DataFrame of sequences (optionally cleaned)
            seqarraylen_clean (np.array): Numpy array of sequence lengths (optionally cleaned)
            shapiro (tuple): Shapiro-Wilk test result
        """
        # get seqs and lengths into array for shapiro test
        seqarray = pd.DataFrame(self.sequence_all)
        seqarraylen = np.array(seqlen)
        shapiro = stats.shapiro(seqarraylen)
        
        # plot histogram
        colors = cm.get_cmap('tab10').colors
        plt.figure(figsize=(8, 5))
        sns.histplot(seqlen, bins=100, color=colors[0], stat='probability',edgecolor='black')
        plt.xlabel("Sequence Length in residues")
        plt.ylabel("Count")
        # plt.ylim(0, 0.31)
        plt.title(f"Distribution of Sequence Lengths for Target Domain {domain_name}")
        plt.tight_layout()
        plt.savefig(f'/home/sapelt/Documents/Master/FINAL/LengthDistribution_{domain_name}.png', dpi=600)
        plt.show()

        seqarraylen_clean = seqarraylen  # [(seqarraylen>=np.quantile(seqarraylen,0.125/2)) & (seqarraylen<=np.quantile(seqarraylen,0.875))]
        seqarray_clean = seqarray  # [(np.char.str_len(seqarray)>=np.quantile(seqarraylen,0.125/2)) & (np.char.str_len(seqarray)<=np.quantile(seqarraylen,0.875))]

        shapiro = stats.shapiro(seqarraylen_clean)

        return seqarray_clean, seqarraylen_clean, shapiro

    def dimension_finder(self, seqarray_len):
        """
        Finds the dimension for the target domain by using the 0.65 quantile of all seqlengths
        """
        # start time
        start_time = time.time()

        # clean by 0.65 quantile
        seqarray_len_clean = int(np.quantile(seqarray_len, 0.65))

        # final print
        elapsed_time = time.time() - start_time
        print(f"\tDone finding dimension\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_len_clean
# -------------------------
# Main
# -------------------------


def main():
    domain_paths = [
        DOMAIN_PATH_1,
        DOMAIN_PATH_2,
        DOMAIN_PATH_3,
        DOMAIN_PATH_4,
        DOMAIN_PATH_5,
        DOMAIN_PATH_6,
        DOMAIN_PATH_7,
    ]
    for domain_path in domain_paths:
        DomainProcessing(domain_path)

    # DomainProcessing(DOMAIN_PATH_2)

########################################
if __name__ == "__main__":
    main()
