"""
Dataset_preprocess_TRAIN_2d.py

ADDED SECOND POSITVE DOMAIN FOR A CLASSIFCIATION

This scipt creates a final dataset ready for training the model.
all input files need to be preprocessed annotated by InterproScan & striped by sequence-fairies-extractDomains
it uses two positive Domain file that is targeted to be classified, negative domains (4 in this test case)
and random protein sequences from swissprot and trembl (2.5 M)
the input length is determined by the positive datasets 0.65 quantile
all sequences above this length are cut into the targeted length using a sliding window apporoach with overlap
all sequences below this length are padded with 'X' to the targeted length (x=unknown amino acid)
the final dataset is saved as a numpy array for further use

Table of Contents:
=========================
databaseCreater Class:
1. __init__
2. _concat_double_delete
3. _sliding_window
4. _multiplier
5. _saver
"""

# -------------------------
# Imports & Globals
# -------------------------

import re
import time

import numpy as np
import pandas as pd
from Bio import SeqIO

# imported from main script
from Dataset_preprocess_TRAIN import DomainProcessing

ENDFILENAME = "DataTrainSwissProt2d.csv"

# -------------------------
# databaseCreater Class
# -------------------------


class databaseCreater:
    """
    Class for creating the actual file for model training.
    Used as input are the target domain df, random other domain df, sprot and trembl df,
    dimension of the target domain, stepsize (default no overlap), and all boudnaries of all domains.
    All variables are created with the class DomainProcessing
    """

    def __init__(
        self,
        seqarray_clean1,
        seqarray_clean2,
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
        seqarray_clean_rnd_sprot,
        # seqarray_clean_rnd_trembl,
        dimension_positive,
        stepsize,
    ):
        # init variables
        self.seqarray_clean1 = seqarray_clean1
        self.seqarray_clean2 = seqarray_clean2

        self.seqarray_clean_PF00079 = seqarray_clean_PF00079
        self.seqarray_clean_PF00080 = seqarray_clean_PF00080
        self.seqarray_clean_PF00118 = seqarray_clean_PF00118
        self.seqarray_clean_PF00162 = seqarray_clean_PF00162
        self.seqarray_clean_rnd_sprot = seqarray_clean_rnd_sprot
        self.seqarray_clean_rnd_all = self.seqarray_clean_rnd_sprot
        # self.seqarray_clean_rnd_trembl = seqarray_clean_rnd_trembl
        # self.seqarray_clean_rnd_all = pd.concat(
        #     [self.seqarray_clean_rnd_sprot, self.seqarray_clean_rnd_trembl]
        # )
        self.dimension_positive = dimension_positive
        self.stepsize = stepsize

        # concat and double delete
        self.seqarray_full = self._concat_double_delete()

        # sliding window
        self.sliding = self._sliding_window(
            self.seqarray_full,
            self.dimension_positive,
            (self.dimension_positive - self.stepsize),
        )

        # multiply the corresponding IDs, boundaries, categories
        self.seqarray_final = self._multiplier(self.seqarray_full, self.sliding)

        # save final df
        self._saver()

    def _concat_double_delete(self):
        """
        Concatinates and deletes identical entries to one df.
        Categories are added: 0 = target domain, 1 = rnd domains, 2 = rnd protein sequences
        Returns the full df with all sequences
        """
        # start time
        start_time = time.time()

        # remove first entry which is empty somehow
        seq_labels_positive1 = self.seqarray_clean1[1:]
        seq_labels_positive2 = self.seqarray_clean2[1:]

        # add category for positive domains
        seq_labels_positive1.loc[:, "categories"] = 0
        seq_labels_positive2.loc[:, "categories"] = 1

        # concat negative domains
        seq_labels_negative_domains = pd.concat(
            (
                self.seqarray_clean_PF00079,
                self.seqarray_clean_PF00080,
                self.seqarray_clean_PF00118,
                self.seqarray_clean_PF00162,
            )
        )
        # remove first empty entry as empty entry
        seq_labels_negative_domains = seq_labels_negative_domains[1:]
        # add category
        seq_labels_negative_domains.loc[:, "categories"] = 2

        # remove first empty entry as empty entry
        seqarray_clean_rnd_all = self.seqarray_clean_rnd_all[1:]
        # add category
        seqarray_clean_rnd_all.loc[:, "categories"] = 3

        # concat all positive and negative domains
        seq_labels_all_domains = pd.concat(
            [seq_labels_positive1, seq_labels_positive2, seq_labels_negative_domains]
        )

        # remove double entries from random sequences that are already in the domain datasets
        seqarray_clean_rnd_without_double_domains = seqarray_clean_rnd_all.loc[
            ~seqarray_clean_rnd_all["Sequences"].isin(
                seq_labels_all_domains["Sequences"]
            )
        ]

        # final concat for full dataset
        seqarray_full = pd.concat(
            [seqarray_clean_rnd_without_double_domains, seq_labels_all_domains]
        )

        # get rations and print infos
        ratio_positive = len(self.seqarray_clean1) / len(seqarray_full)
        print("ratio positive:", ratio_positive)

        ratio_negative_domains = (
            len(self.seqarray_clean_PF00079)
            + len(self.seqarray_clean_PF00080)
            + len(self.seqarray_clean_PF00118)
            + len(self.seqarray_clean_PF00162)
        ) / len(seqarray_full)
        print("ratio negative domains:", ratio_negative_domains, "\n")
        print("SEQARRAT", seqarray_full)
        elapsed_time = time.time() - start_time
        print(
            f"\tDone concat & double deleting\n\tElapsed Time: {elapsed_time:.4f} seconds"
        )
        return seqarray_full

    def _sliding_window(self, seqarray, dimension, stepsize=1):
        """
        Produces sliding windows of a fixed length (dimension) over each entry in the df, with a set stepsize.
        If the final window doesn't fit within the full dimension,
        the last positions (counting backward, len = dimension) are added to the list of windows.
        Returns a series with all sliding window sequences.
        """
        # start time
        start_time = time.time()

        # init list
        seqarray_sliding = []

        # loop through seqs
        for seq in seqarray["Sequences"]:
            # init sublist for slices
            seq_slice = []
            # check if seq longer than dimension, then cut
            if len(seq) > dimension:
                # cut with stepsize and dimension
                seq_slice = [
                    seq[i : i + dimension]
                    for i in range(0, len(seq), stepsize)
                    if len(seq[i : i + dimension]) == dimension
                ]
                # append last slice if not already in list, to not loose end of sequence
                if len(seq) % dimension != 0:
                    if seq[-dimension:] not in seq_slice:
                        seq_slice.append(seq[-dimension:])

                # append to final list
                seqarray_sliding.append(seq_slice)
            # if seq shorter than dimension, keep
            else:
                seqarray_sliding.append([seq])

        # convert to series
        seqarray_sliding = pd.Series(seqarray_sliding)

        # print info
        elapsed_time = time.time() - start_time
        print(f"\tDone sliding window\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_sliding

    def _multiplier(self, seqarray_full, sliding):
        """
        Multiplies the IDs, boundaries, categories corresponding to the number of additionally created windows,
        to have one sliding widnow sequence, with the corresponding IDS, boudnaries and Category.
        Additionally the window position of the windows are given in a new column.
        Returned is a final df with: Sequences, Categories, IDs, Boundaries, WindowPos.
        """
        # start time
        start_time = time.time()

        # init lists & counter
        sequences = []
        categories = []
        ids = []
        category_index = 0

        # loop through sliding windows
        for nested_list in sliding:
            # get current category and id
            current_category = seqarray_full.iloc[category_index]["categories"]
            current_id = seqarray_full.iloc[category_index]["ID"]

            # loop through each sequence in the nested list and append to final lists
            for seq in nested_list:
                sequences.append(seq)
                categories.append(current_category)
                ids.append(current_id)

            # add counter
            category_index += 1

            # print progress every 10,000 iterations
            if category_index % 10000 == 0:
                print(f"Iteration: {category_index} / {len(seqarray_full)}")

        # Create DataFrame once
        sliding_df = pd.DataFrame(
            {
                "Sequences": sequences,
                "categories": categories,
                "ID": ids,
            }
        )

        # print info
        elapsed_time = time.time() - start_time
        print(f"\t Done multiplying in {elapsed_time:.2f} seconds")

        return sliding_df

    def _saver(self):
        """
        Saves the final df in a .csv file. Name of file is hardcoded
        """
        # start time
        start_time = time.time()
        print("Final array:", self.seqarray_final)

        # save
        self.seqarray_final.to_csv(ENDFILENAME, index=False)

        # print info
        elapsed_time = time.time() - start_time
        print(f"\tDone saving\n\tElapsed Time: {elapsed_time:.4f} seconds")


# -------------------------
# Main
# -------------------------


def main():
    """
    FOR CREATING TRAINING DATASET, FOR TRAINING THE MODEL
    """
    # positive Domain PF00177
    print("Loading positive domain PF00177")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00177.fa"
    )
    seqarray_clean1, seqarraylen_clean1, normaltest = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive1 = fasta.dimension_finder(seqarraylen_clean1)
    print("targeted dimension", dimension_positive1)

    # 2nd positive Domain PF00210
    print("Loading positive domain PF00210")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00210.fa"
    )
    seqarray_clean2, seqarraylen_clean2, normaltest = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive2 = fasta.dimension_finder(seqarraylen_clean2)
    print("targeted dimension", dimension_positive2)

    # final targeted dimension, take the larger one
    dimension_positive = max(dimension_positive1, dimension_positive2)

    print("\n,\n")
    print("Final Target Dimension:", dimension_positive)

    # negative Domains:
    print("Loading negative PF00079")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00079.fa"
    )
    seqarray_clean_PF00079, seqarraylen_clean_PF00079, normaltest_PF00079 = (
        fasta.distribution_finder(fasta.len_finder())
    )
    print("Loading negative PF00080")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00080.fa"
    )
    seqarray_clean_PF00080, seqarraylen_clean_PF00080, normaltest_PF00080 = (
        fasta.distribution_finder(fasta.len_finder())
    )
    print("Loading negative PF00118")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00118.fa"
    )
    seqarray_clean_PF00118, seqarraylen_clean_PF00118, normaltest_PF00118 = (
        fasta.distribution_finder(fasta.len_finder())
    )
    print("Loading negative PF00162")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00162.fa"
    )
    seqarray_clean_PF00162, seqarraylen_clean_PF00162, normaltest_PF00162 = (
        fasta.distribution_finder(fasta.len_finder())
    )

    # load in swissprot and trembl
    print("Loading swissprot")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_uniprot_sprot.fa"
    )
    seqarray_clean_rnd_sprot = fasta._load_in_SwissProt()

    # print("Loading trembl")
    # fasta = DomainProcessing(
    #     "/global/research/students/sapelt/Masters/domains_uniprot_trembl.fa"
    # )
    # seqarray_clean_rnd_trembl = fasta._load_in_Trembl()

    # create final dataset
    print("Starting data creation")
    databaseCreater(
        seqarray_clean1,
        seqarray_clean2,
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
        seqarray_clean_rnd_sprot,
        # seqarray_clean_rnd_trembl,
        dimension_positive,
        10,
    )

    print("All done creating Training dataset")


##############################################################
if __name__ == "__main__":
    main()
