###################################################################################################################################
"""
ADDED SECOND POSITVE DOMAIN FOR A CLASSIFCIATION

This scipt creates a final dataset ready for training the model.
all input files need to be preprocessed annotated by InterproScan & striped by sequence-fairies-extractDomains
it uses one positive Domain file that is targeted to be classified, negative domains (4 in this test case)
and random protein sequences from swissprot and trembl (2.5 M)
the input length is determined by the positive datasets 0.65 quantile
all sequences above this length are cut into the targeted length using a sliding window apporoach with overlap
all sequences below this length are padded with 'X' to the targeted length (x=unknown amino acid)
the final dataset is saved as a numpy array for further use
"""
###################################################################################################################################

import re
import time

import numpy as np
import pandas as pd
from Bio import SeqIO



ENDFILENAME= "DataTrainSwissPro_esm_10d_Thiolase"



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

                # Remove the numeric range after the last underscore
                cleaned_id = re.sub(r"_\d+[-\d]+$", "", record.id)

                self.sequence_all.append(str(record.seq))
                self.id_all.append(str(cleaned_id))

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
        print(self.sequence_all)
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
        print(seqarray)
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

    def _load_in_SwissProt(self):
        """
        loads in the SwissProt database and performs the functions len_finder()
        and distribution_finder() on this dataset.
        Returns the seqarray with the IDs and Sequences as well as a list of boundaries for this dataset
        """
        start_time = time.time()
        seqlen_rnd_sprot = self.len_finder()
        seqarray_clean_rnd_sprot, seqarraylen_clean_rnd_sprot, normaltest_rnd_sprot = (
            self.distribution_finder(seqlen_rnd_sprot)
        )
        # print(seqarray_clean_rnd_sprot)
        elapsed_time = time.time() - start_time
        print(f"\tDone loading SwissProt\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_clean_rnd_sprot

    def _load_in_Trembl(self):
        """
        loads in the Trembl database and performs the functions len_finder()
        and distribution_finder() on this dataset.
        Returns the seqarray with the IDs and Sequences as well as a list of boundaries for this dataset
        """
        start_time = time.time()
        seqlen_rnd_trembl = self.len_finder()
        (
            seqarray_clean_rnd_trembl,
            seqarraylen_clean_rnd_trembl,
            normaltest_rnd_trembl,
        ) = self.distribution_finder(seqlen_rnd_trembl)
        # print(seqarray_clean_rnd_sprot)
        elapsed_time = time.time() - start_time
        print(f"\tDone loading Trembl\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_clean_rnd_trembl


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
        ##################################################################################

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

        ##################################################################################

        self.seqarray_full = self._concat_double_delete()

        # self.sliding = self._sliding_window(
        #     self.seqarray_full,
        #     self.dimension_positive,
        #     (self.dimension_positive - self.stepsize),
        # )

        # self.seqarray_final = self._multiplier(self.seqarray_full, self.sliding)

        self._saver()

    def _concat_double_delete(self):
        """
        Concatinates and deletes identical entries to one df.
        Categories are added: 0 = target domain, 1 = rnd domains, 2 = rnd protein sequences
        Returns the full df with all sequences
        """
        start_time = time.time()
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

        seq_labels_negative_domains = pd.concat(
            (
                self.seqarray_clean_PF00079,
                self.seqarray_clean_PF00080,
                self.seqarray_clean_PF00118,
                self.seqarray_clean_PF00162,
            )
        )
        seq_labels_negative_domains = seq_labels_negative_domains[1:]
        seq_labels_negative_domains.loc[:, "categories"] = 10

        seqarray_clean_rnd_all = self.seqarray_clean_rnd_all[1:]

        seqarray_clean_rnd_all.loc[:, "categories"] = 11
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

        seqarray_clean_rnd_without_double_domains = seqarray_clean_rnd_all.loc[
            ~seqarray_clean_rnd_all["Sequences"].isin(
                seq_labels_all_domains["Sequences"]
            )
        ]

        seqarray_full = pd.concat(
            [seqarray_clean_rnd_without_double_domains, seq_labels_all_domains]
        )
        ratio_positive = len(seqarray_clean1) / len(seqarray_full)
        print("ratio positive:", ratio_positive)

        ratio_negative_domains = (
            len(seqarray_clean_PF00079)
            + len(seqarray_clean_PF00080)
            + len(seqarray_clean_PF00118)
            + len(seqarray_clean_PF00162)
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
        start_time = time.time()
        print(seqarray)
        seqarray_sliding = []
        for seq in seqarray["Sequences"]:
            seq_slice = []
            if len(seq) > dimension:
                seq_slice = [
                    seq[i : i + dimension]
                    for i in range(0, len(seq), stepsize)
                    if len(seq[i : i + dimension]) == dimension
                ]
                if len(seq) % dimension != 0:
                    if seq[-dimension:] not in seq_slice:
                        seq_slice.append(seq[-dimension:])
                seqarray_sliding.append(seq_slice)
            else:
                seqarray_sliding.append([seq])

        seqarray_sliding = pd.Series(seqarray_sliding)
        # print(seqarray_sliding)
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
        start_time = time.time()

        sequences = []
        categories = []
        ids = []

        category_index = 0

        for nested_list in sliding:
            # nested_list is expected to be a list of sequences (strings)
            current_category = seqarray_full.iloc[category_index]["categories"]
            current_id = seqarray_full.iloc[category_index]["ID"]

            for seq in nested_list:
                sequences.append(seq)
                categories.append(current_category)
                ids.append(current_id)

            category_index += 1

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

        elapsed_time = time.time() - start_time
        print(f"\t Done multiplying in {elapsed_time:.2f} seconds")
        print(sliding_df)

        return sliding_df

    def _saver(self):
        """
        Saves the final df in a .csv file. Name of file is hardcoded
        """
        start_time = time.time()
        print("Final array:", self.seqarray_full)
        self.seqarray_full.to_csv(f"./Dataframes/{ENDFILENAME}.csv", index=False)
        elapsed_time = time.time() - start_time
        print(f"\tDone saving\n\tElapsed Time: {elapsed_time:.4f} seconds")


##################################################################################################################################################


########### FOR CREATING TRAINING DATASET, FOR TRAINING THE MODEL ###########

# positive Domain PF00108
print("Loading positive domain PF00108")
fasta = DomainProcessing("/global/research/students/sapelt/Masters/Thiolase_set/domains_PF00108.fa")
seqarray_clean1, seqarraylen_clean1, normaltest = fasta.distribution_finder(
    fasta.len_finder()
)
dimension_positive1 = fasta.dimension_finder(seqarraylen_clean1)
print("targeted dimension", dimension_positive1)


# 2nd positive Domain PF00109
print("Loading positive domain PF00109")
fasta = DomainProcessing("/global/research/students/sapelt/Masters/Thiolase_set/domains_PF00109.fa")
seqarray_clean2, seqarraylen_clean2, normaltest = fasta.distribution_finder(
    fasta.len_finder()
)
dimension_positive2 = fasta.dimension_finder(seqarraylen_clean2)
print("targeted dimension", dimension_positive2)


# 3rd positive Domain PF00195
print("Loading positive domain PF00195")
fasta = DomainProcessing("/global/research/students/sapelt/Masters/Thiolase_set/domains_PF00195.fa")
seqarray_clean3, seqarraylen_clean3, normaltest = fasta.distribution_finder(
    fasta.len_finder()
)
dimension_positive3 = fasta.dimension_finder(seqarraylen_clean3)
print("targeted dimension", dimension_positive3)

# 4th positive Domain PF01154
print("Loading positive domain PF01154")
fasta = DomainProcessing("/global/research/students/sapelt/Masters/Thiolase_set/domains_PF01154.fa")
seqarray_clean4, seqarraylen_clean4, normaltest = fasta.distribution_finder(
    fasta.len_finder()
)
dimension_positive4 = fasta.dimension_finder(seqarraylen_clean4)
print("targeted dimension", dimension_positive4)

# 5th positive Domain PF02797
print("Loading positive domain PF02797")
fasta = DomainProcessing("/global/research/students/sapelt/Masters/Thiolase_set/domains_PF02797.fa")
seqarray_clean5, seqarraylen_clean5, normaltest = fasta.distribution_finder(
    fasta.len_finder()
)
dimension_positive5 = fasta.dimension_finder(seqarraylen_clean5)
print("targeted dimension", dimension_positive5)


# 6th positive Domain PF02801
print("Loading positive domain PF02801")
fasta = DomainProcessing("/global/research/students/sapelt/Masters/Thiolase_set/domains_PF02801.fa")
seqarray_clean6, seqarraylen_clean6, normaltest = fasta.distribution_finder(
    fasta.len_finder()
)
dimension_positive6 = fasta.dimension_finder(seqarraylen_clean6)
print("targeted dimension", dimension_positive6)

# 7th positive Domain PF02803
print("Loading positive domain PF02803")
fasta = DomainProcessing("/global/research/students/sapelt/Masters/Thiolase_set/domains_PF02803.fa")
seqarray_clean7, seqarraylen_clean7, normaltest = fasta.distribution_finder(
    fasta.len_finder()
)
dimension_positive7 = fasta.dimension_finder(seqarraylen_clean7)
print("targeted dimension", dimension_positive7)


# 8th positive Domain PF07451
print("Loading positive domain PF07451")
fasta = DomainProcessing("/global/research/students/sapelt/Masters/Thiolase_set/domains_PF07451.fa")
seqarray_clean8, seqarraylen_clean8, normaltest = fasta.distribution_finder(
    fasta.len_finder()
)
dimension_positive8 = fasta.dimension_finder(seqarraylen_clean8)
print("targeted dimension", dimension_positive8)

# 9th positive Domain PF08392
print("Loading positive domain PF08392")
fasta = DomainProcessing("/global/research/students/sapelt/Masters/Thiolase_set/domains_PF08392.fa")
seqarray_clean9, seqarraylen_clean9, normaltest = fasta.distribution_finder(
    fasta.len_finder()
)
dimension_positive9 = fasta.dimension_finder(seqarraylen_clean9)
print("targeted dimension", dimension_positive9)

# 10th positive Domain PF08540
print("Loading positive domain PF08540")
fasta = DomainProcessing("/global/research/students/sapelt/Masters/Thiolase_set/domains_PF08540.fa")
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

    ################### Data creation ########################
    print("Starting data creation")
    dataset = databaseCreater(
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
    ##############################################################
    print("All done creating Training dataset")
