"""
Dataset_preprocess_TRAIN_10d_esm.py

ADDED 10 POSITVE DOMAINS FOR A CLASSIFCIATION

This scipt creates a final dataset ready for training the model.
all input files need to be preprocessed annotated by InterproScan & striped by sequence-fairies-extractDomains
it uses 10 positive Domain file that is targeted to be classified, negative domains (4 in this test case)
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

ENDFILENAME = "DataTrainSwissPro_esm_10d_150w"

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
        seqarray_clean3,
        seqarray_clean4,
        seqarray_clean5,
        seqarray_clean6,
        seqarray_clean7,
        seqarray_clean8,
        seqarray_clean9,
        seqarray_clean10,
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
        self.seqarray_clean3 = seqarray_clean3
        self.seqarray_clean4 = seqarray_clean4
        self.seqarray_clean5 = seqarray_clean5
        self.seqarray_clean6 = seqarray_clean6
        self.seqarray_clean7 = seqarray_clean7
        self.seqarray_clean8 = seqarray_clean8
        self.seqarray_clean9 = seqarray_clean9
        self.seqarray_clean10 = seqarray_clean10

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

        # concat all and delete doubles
        self.seqarray_full = self._concat_double_delete()

        # sliding window
        self.sliding = self._sliding_window(
            self.seqarray_full,
            self.dimension_positive,
            (self.dimension_positive - self.stepsize),
        )

        # mutliply the IDs, boundaries, categories according to the number of created windows
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

        # remove first entry (empty)
        seq_labels_positive1 = self.seqarray_clean1[1:]
        seq_labels_positive2 = self.seqarray_clean2[1:]
        seq_labels_positive3 = self.seqarray_clean3[1:]
        seq_labels_positive4 = self.seqarray_clean4[1:]
        seq_labels_positive5 = self.seqarray_clean5[1:]
        seq_labels_positive6 = self.seqarray_clean6[1:]
        seq_labels_positive7 = self.seqarray_clean7[1:]
        seq_labels_positive8 = self.seqarray_clean8[1:]
        seq_labels_positive9 = self.seqarray_clean9[1:]
        seq_labels_positive10 = self.seqarray_clean10[1:]

        # set categories
        seq_labels_positive1.loc[:, "categories"] = 0
        seq_labels_positive2.loc[:, "categories"] = 1
        seq_labels_positive3.loc[:, "categories"] = 2
        seq_labels_positive4.loc[:, "categories"] = 3
        seq_labels_positive5.loc[:, "categories"] = 4
        seq_labels_positive6.loc[:, "categories"] = 5
        seq_labels_positive7.loc[:, "categories"] = 6
        seq_labels_positive8.loc[:, "categories"] = 7
        seq_labels_positive9.loc[:, "categories"] = 8
        seq_labels_positive10.loc[:, "categories"] = 9

        # create negative domain labels
        seq_labels_negative_domains = pd.concat(
            (
                self.seqarray_clean_PF00079,
                self.seqarray_clean_PF00080,
                self.seqarray_clean_PF00118,
                self.seqarray_clean_PF00162,
            )
        )

        # remove first one (empty) and set category
        seq_labels_negative_domains = seq_labels_negative_domains[1:]
        seq_labels_negative_domains.loc[:, "categories"] = 10

        # remove first one (empty) and set category
        seqarray_clean_rnd_all = self.seqarray_clean_rnd_all[1:]
        seqarray_clean_rnd_all.loc[:, "categories"] = 11

        # concat all positive and negative domains
        seq_labels_all_domains = pd.concat(
            [
                seq_labels_positive1,
                seq_labels_positive2,
                seq_labels_positive3,
                seq_labels_positive4,
                seq_labels_positive5,
                seq_labels_positive6,
                seq_labels_positive7,
                seq_labels_positive8,
                seq_labels_positive9,
                seq_labels_positive10,
                seq_labels_negative_domains,
            ]
        )

        # remove doubles from random sequences that are already in the domain datasets
        seqarray_clean_rnd_without_double_domains = seqarray_clean_rnd_all.loc[
            ~seqarray_clean_rnd_all["Sequences"].isin(
                seq_labels_all_domains["Sequences"]
            )
        ]

        # concat to final set
        seqarray_full = pd.concat(
            [seqarray_clean_rnd_without_double_domains, seq_labels_all_domains]
        )

        # final prints and ration calculation
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
        self.seqarray_final.to_csv(f"./Dataframes/{ENDFILENAME}.csv", index=False)

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

    # 3rd positive Domain PF00211
    print("Loading positive domain PF00211")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00211.fa"
    )
    seqarray_clean3, seqarraylen_clean3, normaltest = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive3 = fasta.dimension_finder(seqarraylen_clean3)
    print("targeted dimension", dimension_positive3)

    # 4th positive Domain PF00215
    print("Loading positive domain PF00215")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00215.fa"
    )
    seqarray_clean4, seqarraylen_clean4, normaltest = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive4 = fasta.dimension_finder(seqarraylen_clean4)
    print("targeted dimension", dimension_positive4)

    # 5th positive Domain PF00217
    print("Loading positive domain PF00217")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00217.fa"
    )
    seqarray_clean5, seqarraylen_clean5, normaltest = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive5 = fasta.dimension_finder(seqarraylen_clean5)
    print("targeted dimension", dimension_positive5)

    # 6th positive Domain PF00406
    print("Loading positive domain PF00406")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00406.fa"
    )
    seqarray_clean6, seqarraylen_clean6, normaltest = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive6 = fasta.dimension_finder(seqarraylen_clean6)
    print("targeted dimension", dimension_positive6)

    # 7th positive Domain PF00303
    print("Loading positive domain PF00303")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00303.fa"
    )
    seqarray_clean7, seqarraylen_clean7, normaltest = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive7 = fasta.dimension_finder(seqarraylen_clean7)
    print("targeted dimension", dimension_positive7)

    # 8th positive Domain PF00246
    print("Loading positive domain PF00246")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00246.fa"
    )
    seqarray_clean8, seqarraylen_clean8, normaltest = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive8 = fasta.dimension_finder(seqarraylen_clean8)
    print("targeted dimension", dimension_positive8)

    # 9th positive Domain PF00457
    print("Loading positive domain PF00457")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00457.fa"
    )
    seqarray_clean9, seqarraylen_clean9, normaltest = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive9 = fasta.dimension_finder(seqarraylen_clean9)
    print("targeted dimension", dimension_positive9)

    # 10th positive Domain PF00502
    print("Loading positive domain PF00502")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00502.fa"
    )
    seqarray_clean10, seqarraylen_clean10, normaltest = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive10 = fasta.dimension_finder(seqarraylen_clean10)
    print("targeted dimension", dimension_positive10)

    dimension_positive = max(
        dimension_positive1,
        dimension_positive2,
        dimension_positive3,
        dimension_positive4,
        dimension_positive5,
        dimension_positive6,
        dimension_positive7,
        dimension_positive8,
        dimension_positive9,
        dimension_positive10,
    )

    print("\n,\n")
    print("Final Target Dimension:", dimension_positive)

    if __name__ == "__main__":
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
            seqarray_clean3,
            seqarray_clean4,
            seqarray_clean5,
            seqarray_clean6,
            seqarray_clean7,
            seqarray_clean8,
            seqarray_clean9,
            seqarray_clean10,
            seqarray_clean_PF00079,
            seqarray_clean_PF00080,
            seqarray_clean_PF00118,
            seqarray_clean_PF00162,
            seqarray_clean_rnd_sprot,
            # seqarray_clean_rnd_trembl,
            150,
            10,
        )

        print("All done creating Training dataset")


##############################################################
if __name__ == "__main__":
    main()
