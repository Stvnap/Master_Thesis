###################################################################################################################################
# This scipt creates a final dataset ready for training the model.
# all input files need to be preprocessed annotated by InterproScan & striped by sequence-fairies-extractDomains
# it uses one positive Domain file that is targeted to be classified, negative domains (4 in this test case)
# and random protein sequences from swissprot and trembl (2.5 M)
# the input length is determined by the positive datasets 0.65 quantile
# all sequences above this length are cut into the targeted length using a sliding window apporoach with overlap
# all sequences below this length are padded with 'X' to the targeted length (x=unknown amino acid)
# the final dataset is saved as a numpy array for further use
###################################################################################################################################

import re
import time

import numpy as np
import pandas as pd
from Bio import SeqIO


class DomainProcessing:
    def __init__(self, domain_path):
        self.domain_path = domain_path
        self.boundaries_all = []
        self.sequence_all = []
        self.id_all = []
        self._opener()

        # if (
        #     domain_path
        #     == "/global/research/students/sapelt/Masters/alluniprot/sprot_domains.fa"
        # ):
        #     self._load_in_SwissProt()

        # if (
        #     domain_path
        #     == "/global/research/students/sapelt/Masters/alluniprot/trembl_domains.fa"
        # ):
        #     self._load_in_Trembl()

    def _opener(self):
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
        start_time = time.time()
        self.sequence_all = pd.DataFrame(self.sequence_all, self.id_all)
        self.sequence_all.columns = ["Sequences", "ID"]
        print(self.sequence_all)
        elapsed_time = time.time() - start_time
        print(f"\tDone Returning\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return self.sequence_all

    def len_finder(self):
        return [len(seq) for seq in self.sequence_all]

    def distribution_finder_and_cleaner(self, seqlen):
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
        start_time = time.time()
        # print(seqarray_len)
        seqarray_len_clean = int(np.quantile(seqarray_len, 0.65))
        elapsed_time = time.time() - start_time
        print(f"\tDone finding dimension\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_len_clean

    def _load_in_SwissProt(self):
        start_time = time.time()
        seqlen_rnd_sprot = self.len_finder()
        seqarray_clean_rnd_sprot, seqarraylen_clean_rnd_sprot, normaltest_rnd_sprot = (
            self.distribution_finder_and_cleaner(seqlen_rnd_sprot)
        )
        # print(seqarray_clean_rnd_sprot)
        elapsed_time = time.time() - start_time
        print(f"\tDone loading SwissProt\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_clean_rnd_sprot

    def _load_in_Trembl(self):
        start_time = time.time()
        seqlen_rnd_trembl = self.len_finder()
        (
            seqarray_clean_rnd_trembl,
            seqarraylen_clean_rnd_trembl,
            normaltest_rnd_trembl,
        ) = self.distribution_finder_and_cleaner(seqlen_rnd_trembl)
        # print(seqarray_clean_rnd_sprot)
        elapsed_time = time.time() - start_time
        print(f"\tDone loading Trembl\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_clean_rnd_trembl


class databaseCreater:
    def __init__(
        self,
        seqarray_clean,
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
        seqarray_clean_rnd_sprot,
        seqarray_clean_rnd_trembl,
        dimension_positive,
        stepsize,
    ):
        self.seqarray_clean = seqarray_clean
        self.seqarray_clean_PF00079 = seqarray_clean_PF00079
        self.seqarray_clean_PF00080 = seqarray_clean_PF00080
        self.seqarray_clean_PF00118 = seqarray_clean_PF00118
        self.seqarray_clean_PF00162 = seqarray_clean_PF00162
        self.seqarray_clean_rnd_sprot = seqarray_clean_rnd_sprot
        # self.seqarray_clean_rnd_all = self.seqarray_clean_rnd_sprot
        self.seqarray_clean_rnd_trembl = seqarray_clean_rnd_trembl
        self.seqarray_clean_rnd_all = pd.concat(
            [self.seqarray_clean_rnd_sprot, self.seqarray_clean_rnd_trembl]
        )
        self.dimension_positive = dimension_positive
        self.stepsize = stepsize
        self.seqarray_full = self._concat_double_delete()

        self.sliding = self._sliding_window(
            self.seqarray_full,
            self.dimension_positive,
            (self.dimension_positive - self.stepsize),
        )

        print(self.seqarray_full)

        self.seqarray_final = self._multiplier(self.seqarray_full, self.sliding)

        self._saver()

    def _concat_double_delete(self):
        start_time = time.time()
        seq_labels_positive = self.seqarray_clean[1:]
        seq_labels_positive.loc[:, "categories"] = 0
        seq_labels_negative_domains = pd.concat(
            (
                self.seqarray_clean_PF00079,
                self.seqarray_clean_PF00080,
                self.seqarray_clean_PF00118,
                self.seqarray_clean_PF00162,
            )
        )
        seq_labels_negative_domains = seq_labels_negative_domains[1:]
        seq_labels_negative_domains.loc[:, "categories"] = 1

        seqarray_clean_rnd_all = self.seqarray_clean_rnd_all[1:]

        seqarray_clean_rnd_all.loc[:, "categories"] = 2

        # print(seq_labels_negative_domains)
        # print(seq_labels_positive)

        seq_labels_all_domains = pd.concat(
            [seq_labels_positive, seq_labels_negative_domains]
        )

        seqarray_clean_rnd_without_double_domains = seqarray_clean_rnd_all.loc[
            ~seqarray_clean_rnd_all["Sequences"].isin(
                seq_labels_all_domains["Sequences"]
            )
        ]

        seqarray_full = pd.concat(
            [seqarray_clean_rnd_without_double_domains, seq_labels_all_domains]
        )
        ratio_positive = len(seqarray_clean) / len(seqarray_full)
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
        start_time = time.time()

        sequences = []
        categories = []

        category_index = 0

        for nested_list in sliding:
            # nested_list is expected to be a list of sequences (strings)
            current_category = seqarray_full.iloc[category_index]["categories"]

            for seq in nested_list:
                sequences.append(seq)
                categories.append(current_category)

            category_index += 1

            if category_index % 10000 == 0:
                print(f"Iteration: {category_index} / {len(seqarray_full)}")

        # Create DataFrame once
        sliding_df = pd.DataFrame({
            "Sequences": sequences,
            "categories": categories
        })

        elapsed_time = time.time() - start_time
        print(f"\t Done multiplying in {elapsed_time:.2f} seconds")
        print(sliding_df)

        return sliding_df

    def _saver(self):
        start_time = time.time()
        print("Final array:", self.seqarray_final)
        self.seqarray_final.to_csv("DataTrainALL.csv", index=False)
        elapsed_time = time.time() - start_time
        print(f"\tDone saving\n\tElapsed Time: {elapsed_time:.4f} seconds")


##################################################################################################################################################


########### FOR CREATING TRAINING DATASET, FOR TRAINING THE MODEL ###########

if __name__ == "__main__":
    # positive Domain PF00177
    print("Loading positive domain PF00177")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00177.fa"
    )
    seqarray_clean, seqarraylen_clean, normaltest = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    dimension_positive = fasta.dimension_finder(seqarraylen_clean)
    # print("targeted dimension", dimension_positive)

    # negative Domains:
    print("Loading negative PF00079")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00079.fa"
    )
    seqarray_clean_PF00079, seqarraylen_clean_PF00079, normaltest_PF00079 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    print("Loading negative PF00080")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00080.fa"
    )
    seqarray_clean_PF00080, seqarraylen_clean_PF00080, normaltest_PF00080 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    print("Loading negative PF00118")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00118.fa"
    )
    seqarray_clean_PF00118, seqarraylen_clean_PF00118, normaltest_PF00118 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    print("Loading negative PF00162")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00162.fa"
    )
    seqarray_clean_PF00162, seqarraylen_clean_PF00162, normaltest_PF00162 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )

    # load in swissprot and trembl
    print("Loading swissprot")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_uniprot_sprot.fa"
    )
    seqarray_clean_rnd_sprot = fasta._load_in_SwissProt()

    print("Loading trembl")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_uniprot_trembl.fa"
    )
    seqarray_clean_rnd_trembl = fasta._load_in_Trembl()

    ################### Data creation ########################
    print("Starting data creation")
    dataset = databaseCreater(
        seqarray_clean,
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
        seqarray_clean_rnd_sprot,
        seqarray_clean_rnd_trembl,
        dimension_positive,
        10,
    )
    ##############################################################
    print("All done creating Training dataset")

