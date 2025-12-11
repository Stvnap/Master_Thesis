"""
Dataset_preprocess.py

FOR CREATING TRAINING DATASET TO TRAIN THE MODEL
Script that prepares the inputed fasta file for normal usage purpuses.
This scipt creates a final dataset ready for predicting with the model.
the input length is determined by the positive datasets 0.65 quantile (previously done in Dataset_preprocess_TRAIN_v2.py)
all sequences above this length are cut into the targeted length using a sliding window apporoach with overlap
all sequences below this length are padded with 'X' to the targeted length (x=unknown amino acid)
the final dataset is saved as a csv for further use
df consists of sequence, ID, window_pos

Table of Contents:
=========================
Classes:
1. DomainProcessing
2. DatabaseCreater

Functions:
=========================
1. _opener
2. seq_array_returner
3. len_finder
4. distribution_finder
5. dimension_finder
6. load_in_Dataset
7. _sliding_window
8. _multiplier
9. _saver
10. main
"""
###################################################################################################################################

# -------------------------
# Imports & Globals
# -------------------------

import time

import numpy as np
import pandas as pd
from Bio import SeqIO
from Dataset_preprocess_TRAIN import (
    dimension_positive,
)  # Importing the dimension from the training preprocessing script, link accordingly to file location

INPUT_FASTA = "/global/research/students/sapelt/Masters/domains_PF00177.fa"
FINAL_OUT = "TESTESTESTSS.csv"


# -------------------------
# DomainProcessing Class
# -------------------------
class DomainProcessing:
    """
    Class processing the load in and creation of the seqarray.
    As well as the determination of the dimension length of the target domain, setting the dimension for the model.
    Args:
        domain_path: path to the fasta file containing all sequences of the target domain
    Returns:
        seqarray_clean(returns load_in_Dataset): df of Sequences and IDs
    """

    def __init__(self, domain_path):
        self.domain_path = domain_path
        # init lists
        self.boundaries_all = []
        self.sequence_all = []
        self.id_all = []
        self._opener()

    def _opener(self):
        """
        Opens the .fasta file and creates of type IDs, Sequence
        """
        # start time
        start_time = time.time()
        # open fasta file and parse with SeqIO
        with open(self.domain_path, "r") as file:
            for record in SeqIO.parse(file, "fasta"):
                # # debugging
                # print(record.id)
                # print(record.seq)

                # gather seqs and ids
                self.sequence_all.append(str(record.seq))
                self.id_all.append(str(record.id))

                # # debugging
                # print(self.sequence_all)
                # print(self.id_all)

            # time elapsed prints
            elapsed_time = time.time() - start_time
            print(f"\tDone opening\n\tElapsed Time: {elapsed_time:.4f} seconds")

    def seq_array_returner(self):
        """
        Returns a df of the form Sequences and IDs
        """
        # start time
        start_time = time.time()
        # create df
        self.sequence_all = pd.DataFrame(self.sequence_all, self.id_all)
        # set columns
        self.sequence_all.columns = ["Sequences", "ID"]
        # elapesed time print
        elapsed_time = time.time() - start_time
        print(f"\tDone Returning\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return self.sequence_all

    def len_finder(self):
        """
        Function to determine the Sequence length of every sequence. Returns a list of lengths from all seqs.
        """
        return [len(seq) for seq in self.sequence_all]

    def distribution_finder(self, seqlen):
        """
        Finds the distribution of sequence lengths of the corresponding domain, shapiro test are comment out and usable if needed.
        Addtionally, potential to filter out outliers is given, currently not used.

        Args:
            seqlen: list of sequence lengths
        Returns:
            seqarray_clean: df of Sequences and IDs (can be used for further filtering if needed)
            seqarraylen_clean: list of sequence lengths (can be used for further filtering if needed)
            shapiro: shapiro test result (can be used for further analysis if needed)
        """
        # start time
        start_time = time.time()
        # make array out of seqlen
        seqarraylen = np.array(seqlen)

        # potential to clean out outliers, currently not used
        seqarraylen_clean = seqarraylen  # [(seqarraylen>=np.quantile(seqarraylen,0.125/2)) & (seqarraylen<=np.quantile(seqarraylen,0.875))]
        # print(seqarraylen_clean)

        # create seqarray df
        seqarray = pd.DataFrame({"ID": self.id_all, "Sequences": self.sequence_all})
        # remove sequences corresponding to cleaned lengths
        seqarray_clean = seqarray  # [(np.char.str_len(seqarray)>=np.quantile(seqarraylen,0.125/2)) & (np.char.str_len(seqarray)<=np.quantile(seqarraylen,0.875))]

        # Shapiro tests
        # default shapiro None
        shapiro = None
        # shapiro = stats.shapiro(seqarraylen)
        # shapiro = stats.shapiro(seqarraylen_clean)

        # final print
        elapsed_time = time.time() - start_time
        print(
            f"\tDone finding distribution\n\tElapsed Time: {elapsed_time:.4f} seconds"
        )
        return seqarray_clean, seqarraylen_clean, shapiro

    def dimension_finder(self, seqarray_len):
        """
        Finds the dimension for the target domain by using the 0.65 quantile of all seqlengths
        """
        # Start time
        start_time = time.time()
        # get 0.65 quantile for the dimension
        seqarray_len_clean = int(np.quantile(seqarray_len, 0.65))
        # print out
        elapsed_time = time.time() - start_time
        print(f"\tDone finding dimension\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_len_clean

    def load_in_Dataset(self):
        """
        calls all necessary functions to load in the dataset and prepare it for further processing
        Returns:
            seqarray_clean: df of Sequences and IDs. Ready for further processing
        """
        # start time
        start_time = time.time()
        # len finder
        seqlen_rnd = self.len_finder()
        # distribution finder
        (
            seqarray_clean,
            seqarraylen_clean,
            normaltest,
        ) = self.distribution_finder(seqlen_rnd)
        # final prints
        elapsed_time = time.time() - start_time
        print(f"\tDone loading Trembl\n\tElapsed Time: {elapsed_time:.4f} seconds")
        # print(seqarray_clean.head(50))
        return seqarray_clean


# -------------------------
# DatabaseCreater Class
# -------------------------


class DatabaseCreater:
    """
    Class for creating the actual file for model training. Determines sliding windows, multiplies IDs accordingly and saves the final df.
    Windowing is based on the dimension determined from the positive training dataset.
    Args:
        seqarray_clean: df of Sequences and IDs
        dimension_positive: int, dimension determined from the positive training dataset
        stepsize: int, stepsize for the sliding window approach
    Returns:
        seqarray_final: final df of Sequences, IDs, WindowPos saved as csv
    """

    def __init__(
        self,
        seqarray_clean,
        dimension_positive,
        stepsize,
    ):
        # init args
        self.seqarray_clean = seqarray_clean
        self.dimension_positive = dimension_positive
        self.stepsize = stepsize

        # call sliding window and multiplier
        self.sliding = self._sliding_window(
            self.seqarray_clean,
            self.dimension_positive,
            (self.dimension_positive - self.stepsize),
        )
        self.seqarray_final = self._multiplier(self.seqarray_clean, self.sliding)

        # call saver to finally save the df
        self._saver()

    def _sliding_window(self, seqarray, dimension, stepsize=1):
        """
        Produces sliding windows of a fixed length (dimension) over each entry in the df, with a set stepsize.
        If the final window doesn't fit within the full dimension,
        the last positions (counting backward, len = dimension) are added to the list of windows.
        Returns a series with all sliding window sequences.
        """
        # init start time and init lists
        start_time = time.time()
        seqarray_sliding = []
        self.end_window = []

        # iterate over all sequences
        for seq in seqarray["Sequences"]:
            # # debugging
            # print(seq)
            # print(len(seq))
            # get sequence length
            seqlen = len(seq)
            # create sliding windows
            if len(seq) > dimension:
                # create slices with stepsize
                slices = [
                    seq[i : i + dimension]
                    for i in range(0, len(seq) - dimension + 1, stepsize)
                ]
                # check if last slice fits, if not add the last dimension slice
                if len(seq) % dimension != 0:
                    slices.append(seq[-dimension:])
            # if sequence smaller than dimension, just add the full sequence (will be padded later)
            else:
                slices = [seq]
            # append to final list
            seqarray_sliding.append(slices)
            self.end_window.append(seqlen)
        # final print
        elapsed_time = time.time() - start_time
        print(f"\tDone sliding window\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return pd.Series(seqarray_sliding)

    def _multiplier(self, seqarray_full, sliding):
        """
        Multiplies the IDs, boundaries, categories corresponding to the number of additionally created windows,
        to have one sliding window sequence, with the corresponding IDS, boudnaries and Category.
        Additionally the window position of the windows are given in a new column.
        Returned is a final df with: Sequences, Categories, IDs, Boundaries, WindowPos.
        """

        # init lists, start time and category index for iteration
        sequences = []
        ids = []
        window_positions = []
        start_time = time.time()
        category_index = 0

        # iterate over all sliding windows
        for nested_list in sliding:
            # get current row info, ID and length of nested list
            current_row = seqarray_full.iloc[category_index]
            current_id = current_row["ID"]
            len_nested = len(nested_list)

            # iterate over all created windows for current sequence
            for i in range(len_nested):
                # get current sequence, append to lists
                seq = nested_list[i]
                sequences.append(seq)
                # append current ID to list
                ids.append(current_id)

                # Calculate WindowPos as string
                # if more than 1 window
                if i == len_nested - 1 and len_nested > 1:
                    # get last window start and end
                    last_window_start = (
                        self.end_window[category_index] - self.dimension_positive
                    )
                    # get last window end
                    last_window_end = self.end_window[category_index]
                    # make string for window_pos column
                    window_pos = f"{last_window_start}-{last_window_end}"
                # for single window sequences, window pos is 0 and dimension_positive
                else:
                    # calculate start and end for window_pos string
                    start = i * self.dimension_positive - (
                        self.stepsize * i if i > 0 else 0
                    )
                    end = (i + 1) * self.dimension_positive - (
                        self.stepsize * i if i > 0 else 0
                    )
                    window_pos = f"{start}-{end}"

                # append window_pos to list
                window_positions.append(window_pos)

            # increment category index for next sequence
            category_index += 1

            # print progress every 10000 iterations
            if category_index % 10000 == 0:
                print(
                    "Multiplication iteration:", category_index, "/", len(seqarray_full)
                )

        # Convert once to DataFrame at the end
        sliding_df = pd.DataFrame(
            {
                "Sequences": sequences,
                "ID": ids,
                "WindowPos": window_positions,
            }
        )

        # final print
        elapsed_time = time.time() - start_time
        print(f"\t Done multiplying\n\t Elapsed Time: {elapsed_time:.4f} seconds")
        return sliding_df

    def _saver(self):
        """
        Saves the final df in a .csv file. Name of file is hardcoded
        """
        # start time
        start_time = time.time()
        # only if the script is run directly save it
        if __name__ == "__main__":
            self.seqarray_final.to_csv(FINAL_OUT, index=False)
        # final print
        elapsed_time = time.time() - start_time
        print(f"\tDone saving\n\tElapsed Time: {elapsed_time:.4f} seconds")


# -------------------------
# Main
# -------------------------


def main():
    """
    Main function to call the classes and create the final dataset for model training.
    """

    # Init class
    print("Loading in")
    fasta = DomainProcessing(INPUT_FASTA)
    # Load dataset and analyze it on length distribution, target dimension,etc
    seqarray_clean_rnd_trembl = fasta.load_in_Dataset()

    # Creation of Dataset by sliding window, multiply IDs, Boundaries, Categories accordingly and saving it
    print("Starting data creation")
    DatabaseCreater(
        seqarray_clean_rnd_trembl,
        dimension_positive,
        10,
    )
    print("All done creating Training dataset")


#################################
if __name__ == "__main__":
    main()
