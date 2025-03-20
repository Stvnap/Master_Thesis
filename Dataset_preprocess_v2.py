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

import numpy as np
import pandas as pd
from scipy import stats


class DomainProcessing:
    def __init__(self, domain_path):
        self.domain_path = domain_path
        self.boundaries_all = []
        self.sequence_all = []
        self.id_all = []
        self._opener()

        if (
            domain_path
            == "/global/research/students/sapelt/Masters/alluniprot/sprot_domains.fa"
        ):
            self._load_in_SwissProt()

    def _opener(self):
        with open(self.domain_path, "r") as file:
            sequence = []

            for line in file:
                line = line.strip()
                id_whole = line
                # print(line)
                # print('\n')
                if line.startswith(">"):
                    sequence_joined = "".join(sequence)
                    self.sequence_all.append(sequence_joined)

                    try:
                        boundaries = re.search(r"_(\d+-\d+)", line)
                        boundaries = boundaries.group(1)
                        self.boundaries_all.append(boundaries)
                    except:
                        continue

                    self.id_all.append(id_whole)

                    self.id_whole = line
                    # print(id_whole)

                    sequence = []
                # print(boundaries_all[0])
                else:
                    sequence.append(line)
            # print(id_all)

            # print(sequence_all)

            # print(len(id_all))
            # print(len(sequence_all))
            # print(len(boundaries_all))

    def seq_array_returner(self):
        self.sequence_all = pd.DataFrame(self.sequence_all)
        self.sequence_all.columns = ["Sequences"]
        return self.sequence_all

    def len_finder(self):
        return [len(seq) for seq in self.sequence_all]

    def distribution_finder_and_cleaner(self, seqlen):
        seqarraylen = np.array(seqlen)
        shapiro = stats.shapiro(seqarraylen)
        # return shapiro
        seqarraylen_clean = seqarraylen  # [(seqarraylen>=np.quantile(seqarraylen,0.125/2)) & (seqarraylen<=np.quantile(seqarraylen,0.875))]
        # print(seqarraylen_clean)

        seqarray = pd.DataFrame(self.sequence_all)
        seqarray.columns = ["Sequences"]
        seqarray_clean = seqarray  # [(np.char.str_len(seqarray)>=np.quantile(seqarraylen,0.125/2)) & (np.char.str_len(seqarray)<=np.quantile(seqarraylen,0.875))]

        shapiro = stats.shapiro(seqarraylen_clean)

        return seqarray_clean, seqarraylen_clean, shapiro

    def dimension_finder(self, seqarray_len):
        # print(seqarray_len)
        seqarray_len_clean = int(np.quantile(seqarray_len, 0.65))
        return seqarray_len_clean

    def _load_in_SwissProt(self):
        seqlen_rnd_sprot = self.len_finder()
        seqarray_clean_rnd_sprot, seqarraylen_clean_rnd_sprot, normaltest_rnd_sprot = (
            self.distribution_finder_and_cleaner(seqlen_rnd_sprot)
        )
        # print(seqarray_clean_rnd_sprot)

        return seqarray_clean_rnd_sprot


class databaseCreater:
    def __init__(
        self,
        seqarray_clean,
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
        seqarray_clean_rnd_sprot,
        dimension_positive,
        stepsize,
    ):
        self.seqarray_clean = seqarray_clean
        self.seqarray_clean_PF00079 = seqarray_clean_PF00079
        self.seqarray_clean_PF00080 = seqarray_clean_PF00080
        self.seqarray_clean_PF00118 = seqarray_clean_PF00118
        self.seqarray_clean_PF00162 = seqarray_clean_PF00162
        self.seqarray_clean_rnd_sprot = seqarray_clean_rnd_sprot
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
        seq_labels_positive = self.seqarray_clean[1:]
        seq_labels_positive["categories"] = 0
        seq_labels_negative_domains = pd.concat(
            (
                self.seqarray_clean_PF00079,
                self.seqarray_clean_PF00080,
                self.seqarray_clean_PF00118,
                self.seqarray_clean_PF00162,
            )
        )
        seq_labels_negative_domains = seq_labels_negative_domains[1:]
        seq_labels_negative_domains["categories"] = 1

        seqarray_clean_rnd_sprot = self.seqarray_clean_rnd_sprot[1:]

        seqarray_clean_rnd_sprot["categories"] = 2

        # print(seq_labels_negative_domains)
        # print(seq_labels_positive)

        seq_labels_all_domains = pd.concat(
            [seq_labels_positive, seq_labels_negative_domains]
        )

        seqarray_clean_rnd_without_double_domains = seqarray_clean_rnd_sprot[
            ~self.seqarray_clean_rnd_sprot["Sequences"].isin(
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

        return seqarray_full

    def _sliding_window(self, seqarray, dimension, stepsize=1):
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
        return seqarray_sliding

    def _multiplier(self, seqarray_full, sliding):
        result_list = []  # List to collect DataFrames
        category_index = 0
        for nested_list in sliding:
            nested_list = pd.DataFrame(nested_list, columns=["Sequences"])

            categories = seqarray_full.iloc[category_index]["categories"]
            nested_list["categories"] = categories

            category_index += 1

            # Append the DataFrame to the result list instead of concatenating
            temp_df = pd.DataFrame(
                {
                    "Sequences": nested_list["Sequences"],
                    "categories": nested_list["categories"],
                }
            )
            result_list.append(temp_df)

            if category_index % 1000 == 0:
                print("iteration:", category_index)

        # Concatenate all DataFrames in the result list at once
        sliding_df = pd.concat(result_list, ignore_index=True)
        print(sliding_df)
        return sliding_df

    def _saver(self):
        print("Final array:", self.seqarray_final)
        self.seqarray_final.to_csv("dataSwissProt_NoOvelap.csv", index=False)


##################################################################################################################################################


if __name__ == "__main__":
    # positive Domain PF00177
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00177.fa"
    )
    seqarray_clean, seqarraylen_clean, normaltest = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    dimension_positive = fasta.dimension_finder(seqarraylen_clean)
    # print("targeted dimension", dimension_positive)

    # negative Domains:
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00079.fa"
    )
    seqarray_clean_PF00079, seqarraylen_clean_PF00079, normaltest_PF00079 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00080.fa"
    )
    seqarray_clean_PF00080, seqarraylen_clean_PF00080, normaltest_PF00080 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00118.fa"
    )
    seqarray_clean_PF00118, seqarraylen_clean_PF00118, normaltest_PF00118 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00162.fa"
    )
    seqarray_clean_PF00162, seqarraylen_clean_PF00162, normaltest_PF00162 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )

    # load in swissprot and trembl
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/alluniprot/sprot_domains.fa"
    )
    seqarray_clean_rnd_sprot = fasta._load_in_SwissProt()

    ################### Data creation ########################
    dataset = databaseCreater(
        seqarray_clean,
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
        seqarray_clean_rnd_sprot,
        dimension_positive,
        0,
    )
