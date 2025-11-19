###################################################################################################################################
"""
Script that prepares the inputed fasta file for normal usage purpuses.
This scipt creates a final dataset ready for predicting with the model.
the input length is determined by the positive datasets 0.65 quantile (previously done in Dataset_preprocess_TRAIN_v2.py)
all sequences above this length are cut into the targeted length using a sliding window apporoach with overlap
all sequences below this length are padded with 'X' to the targeted length (x=unknown amino acid)
the final dataset is saved as a csv for further use
df consists of sequence, ID, window_pos
"""
###################################################################################################################################

import re
import time

import numpy as np
import pandas as pd
from Bio import SeqIO
from Dataset_preprocess_TRAIN import dimension_positive


class DomainProcessing:
    """
    Class processing the load in and creation of all dataframes needed. 
    As well as the determination of the dimension length of the target domain.
    """
    def __init__(self, domain_path):
        self.domain_path = domain_path
        self.boundaries_all = []
        self.sequence_all = []
        self.id_all = []
        self._opener()

    def _opener(self):
        """
        Opens the .csv file and creates of type IDs, Sequence and found Boundaries by Interproscan
        """
        start_time = time.time()
        with open(self.domain_path, "r") as file:
            for record in SeqIO.parse(file, "fasta"):
                # print(record.id)
                # print(record.seq)

                self.sequence_all.append(str(record.seq))
                self.id_all.append(str(record.id))

                # print(self.sequence_all)
                # print(self.id_all)
            elapsed_time = time.time() - start_time
            print(f"\tDone opening\n\tElapsed Time: {elapsed_time:.4f} seconds")

    def seq_array_returner(self):
        """
        Returns a df of the form Sequences and IDs
        """
        start_time = time.time()
        self.sequence_all = pd.DataFrame(self.sequence_all, self.id_all)
        self.sequence_all.columns = ["Sequences", "ID"]
        # print(self.sequence_all)
        elapsed_time = time.time() - start_time
        print(f"\tDone Returning\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return self.sequence_all

    def len_finder(self):
        """
        Function to determine the Sequence length of every sequence
        """
        return [len(seq) for seq in self.sequence_all]

    def distribution_finder(self, seqlen):
        """
        Finds the distribution of sequence lengths of the corresponding domain, options for shapiro test.          
        """
        start_time = time.time()
        seqarraylen = np.array(seqlen)
        # shapiro = stats.shapiro(seqarraylen)
        # return shapiro
        seqarraylen_clean = seqarraylen  # [(seqarraylen>=np.quantile(seqarraylen,0.125/2)) & (seqarraylen<=np.quantile(seqarraylen,0.875))]
        # print(seqarraylen_clean)

        seqarray = pd.DataFrame({"ID": self.id_all, "Sequences": self.sequence_all})
        # print(seqarray)
        seqarray_clean = seqarray  # [(np.char.str_len(seqarray)>=np.quantile(seqarraylen,0.125/2)) & (np.char.str_len(seqarray)<=np.quantile(seqarraylen,0.875))]

        # shapiro = stats.shapiro(seqarraylen_clean)
        shapiro = None
        elapsed_time = time.time() - start_time
        print(
            f"\tDone finding distribution\n\tElapsed Time: {elapsed_time:.4f} seconds"
        )
        return seqarray_clean, seqarraylen_clean, shapiro

    def dimension_finder(self, seqarray_len):
        """
        Finds the dimension for the target domain by using the 0.65 quantile of all seqlengths
        """
        start_time = time.time()
        # print(seqarray_len)
        seqarray_len_clean = int(np.quantile(seqarray_len, 0.65))
        elapsed_time = time.time() - start_time
        print(f"\tDone finding dimension\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_len_clean

    def load_in_Dataset(self):
        """
        loads in the Trembl database and performs the functions len_finder() 
        and distribution_finder() on this dataset.
        Returns the seqarray with the IDs and Sequences as well as a list of boundaries for this dataset
        """
        start_time = time.time()
        seqlen_rnd = self.len_finder()
        (
            seqarray_clean,
            seqarraylen_clean,
            normaltest,
        ) = self.distribution_finder(seqlen_rnd)
        # print(seqarray_clean_rnd_sprot)
        elapsed_time = time.time() - start_time
        print(f"\tDone loading Trembl\n\tElapsed Time: {elapsed_time:.4f} seconds")
        # print(seqarray_clean.head(50))
        return seqarray_clean


class databaseCreater:
    """
    Class for creating the actual file for model training. 
    Used as input are the target domain df, random other domain df, sprot and trembl df, 
    dimension of the target domain, stepsize (default no overlap), and all boudnaries of all domains. 
    All variables are created with the class DomainProcessing
    """
    def __init__(
        self,
        seqarray_clean,
        dimension_positive,
        stepsize,
    ):
        ##################################################################################

        self.seqarray_clean = seqarray_clean
        self.dimension_positive = dimension_positive
        self.stepsize = stepsize

        ##################################################################################

        self.sliding = self._sliding_window(
            self.seqarray_clean,
            self.dimension_positive,
            (self.dimension_positive - self.stepsize),
        )

        self.seqarray_final = self._multiplier(self.seqarray_clean, self.sliding)

        self._saver()

    def _sliding_window(self, seqarray, dimension, stepsize=1):
        """
        Produces sliding windows of a fixed length (dimension) over each entry in the df, with a set stepsize.
        If the final window doesn't fit within the full dimension, 
        the last positions (counting backward, len = dimension) are added to the list of windows.
        Returns a series with all sliding window sequences.
        """
        start_time = time.time()
        seqarray_sliding = []
        self.end_window = []

        for seq in seqarray["Sequences"]:
            # print(seq)
            # print(len(seq))
            seqlen = len(seq)
            if len(seq) > dimension:
                slices = [
                    seq[i : i + dimension]
                    for i in range(0, len(seq) - dimension + 1, stepsize)
                ]
                if len(seq) % dimension != 0:
                    slices.append(seq[-dimension:])
            else:
                slices = [seq]
            seqarray_sliding.append(slices)

            self.end_window.append(seqlen)

        elapsed_time = time.time() - start_time
        print(f"\tDone sliding window\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return pd.Series(seqarray_sliding)

    def _multiplier(self, seqarray_full, sliding):
        """
        Multiplies the IDs, boundaries, categories corresponding to the number of additionally created windows, 
        to have one sliding widnow sequence, with the corresponding IDS, boudnaries and Category. 
        Additionally the window position of the windows are given in a new column. 
        Returned is a final df with: Sequences, Categories, IDs, Boundaries, WindowPos.
        """
        start_time = time.time()

        # Predefine lists for better performance
        sequences = []
        ids = []
        window_positions = []

        category_index = 0

        for nested_list in sliding:
            current_row = seqarray_full.iloc[category_index]
            current_id = current_row["ID"]


            len_nested = len(nested_list)

            for i in range(len_nested):
                seq = nested_list[i]
                sequences.append(seq)
                ids.append(current_id)

            

                # Calculate WindowPos as string
                if i == len_nested - 1 and len_nested > 1:
                    # last window gets special end_window value
                    last_window_start = (
                        self.end_window[category_index] - self.dimension_positive
                    )
                    last_window_end = self.end_window[category_index]
                    window_pos = f"{last_window_start}-{last_window_end}"
                else:
                    start = i * self.dimension_positive - (
                        self.stepsize * i if i > 0 else 0
                    )
                    end = (i + 1) * self.dimension_positive - (
                        self.stepsize * i if i > 0 else 0
                    )
                    window_pos = f"{start}-{end}"

                window_positions.append(window_pos)

            category_index += 1

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

        elapsed_time = time.time() - start_time
        print(f"\t Done multiplying\n\t Elapsed Time: {elapsed_time:.4f} seconds")
        return sliding_df

    def _saver(self):
        """
        Saves the final df in a .csv file. Name of file is hardcoded
        """
        start_time = time.time()
        # print("Final array:", self.seqarray_final)
        if __name__ == "__main__":
            self.seqarray_final.to_csv("TESTESTESTSS.csv", index=False)
        elapsed_time = time.time() - start_time
        print(f"\tDone saving\n\tElapsed Time: {elapsed_time:.4f} seconds")


##################################################################################################################################################


########### FOR CREATING TRAINING DATASET, FOR TRAINING THE MODEL ###########

if __name__ == "__main__":


    print("Loading in")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00177.fa"
    )
    seqarray_clean_rnd_trembl = fasta.load_in_Dataset()

    ################### Data creation ########################
    print("Starting data creation")
    dataset = databaseCreater(
        seqarray_clean_rnd_trembl,
        dimension_positive,
        10,
    )
    ##############################################################
    print("All done creating Training dataset")

