###################################################################################################################################
"""
This scipt creates a final dataset ready for EVALUATION the model's performance.
all input files need to be preprocessed annotated by InterproScan & striped by sequence-fairies-extractDomains
it uses one positive Domain file that is targeted to be classified, negative domains (4 in this test case)
and random protein sequences from swissprot and trembl (2.5 M)
the input length is determined by the positive datasets 0.65 quantile
all sequences above this length are cut into the targeted length using a sliding window apporoach with overlap
all sequences below this length are padded with 'X' to the targeted length (x=unknown amino acid)
the final dataset is saved as a numpy array for further use
!!!!    THE RAW SEQUENCE FILES NEED TO BE NAMED raw(domainID).fasta, and the domain files domains_(domainID).fa     !!!!
"""
###################################################################################################################################

import os
import re
import time

import numpy as np
import pandas as pd
from Bio import SeqIO


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
            records = list(SeqIO.parse(file, "fasta"))

        # Process sequences and cleaned IDs using list comprehensions
        self.sequence_all = [str(record.seq) for record in records]
        self.id_all = [re.sub(r"_\d+[-\d]+$", "", record.id) for record in records]
        # for eval purpuses
        print(self.domain_path)
        if "domain" not in os.path.basename(self.domain_path):
            self.domain_path = os.path.basename(self.domain_path)

            domain_path2 = os.path.join(
                "/global/research/students/sapelt/Masters/",
                self.domain_path.replace("raw", "domains_"),
            )
            domain_path2 = domain_path2.replace("fasta", "fa")
            print(domain_path2)
            with open(domain_path2, "r") as file:
                records = list(SeqIO.parse(file, "fasta"))
                # print(records)
                self.boundaries_all = [
                    re.sub(r"^.*_([0-9]+)-([0-9]+)$", r"\1-\2", record.id)
                    for record in records
                ]
                self.id_all_from_boundaries = [
                    re.sub(r"_\d+[-\d]+$", "", record.id) for record in records
                ]

                boundary_array = pd.DataFrame(
                    {
                        "ID": self.id_all_from_boundaries,
                        "Boundaries": self.boundaries_all,
                    }
                )

                print(boundary_array)

                # Create a new list to store the updated boundaries
                updated_boundaries = []
                previous_id = None
                current_boundaries = ""

                for i in range(len(boundary_array)):
                    current_id = boundary_array.loc[i, "ID"]
                    if current_id == previous_id:
                        # Append current boundaries to the previous boundaries
                        current_boundaries += "," + boundary_array.loc[i, "Boundaries"]
                    else:
                        # If the ID changes, store the previous boundaries
                        if previous_id is not None:
                            updated_boundaries.append(
                                {"ID": previous_id, "Boundaries": current_boundaries}
                            )
                        # Start a new group
                        previous_id = current_id
                        current_boundaries = boundary_array.loc[i, "Boundaries"]

                # Add the last group
                if previous_id is not None:
                    updated_boundaries.append(
                        {"ID": previous_id, "Boundaries": current_boundaries}
                    )

                # Convert the updated boundaries back to a DataFrame
                boundary_array = pd.DataFrame(updated_boundaries)

        sequence_all = pd.DataFrame({"ID": self.id_all, "Sequences": self.sequence_all})

        self.merged_df = pd.merge(boundary_array, sequence_all, on="ID", how="outer")

        print("MERGED", self.merged_df)
        print("length merged", len(self.merged_df))

        elapsed_time = time.time() - start_time
        print(f"\tDone opening\n\tElapsed Time: {elapsed_time:.4f} seconds")

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
        seqarray = self.merged_df

        print(seqarray)
        seqarray_clean = seqarray  # [(np.char.str_len(seqarray)>=np.quantile(seqarraylen,0.125/2)) & (np.char.str_len(seqarray)<=np.quantile(seqarraylen,0.875))]

        # shapiro = stats.shapiro(seqarraylen_clean)
        shapiro = None
        elapsed_time = time.time() - start_time
        print(
            f"\tDone finding distribution\n\tElapsed Time: {elapsed_time:.4f} seconds"
        )
        return seqarray_clean, self.boundaries_all

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
        seqarray_clean_rnd_sprot, boundaries_all = self.distribution_finder(
            seqlen_rnd_sprot
        )
        # print(seqarray_clean_rnd_sprot)
        elapsed_time = time.time() - start_time
        print(f"\tDone loading SwissProt\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_clean_rnd_sprot, boundaries_all

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
            boundaries_all,
        ) = self.distribution_finder(seqlen_rnd_trembl)
        # print(seqarray_clean_rnd_sprot)
        elapsed_time = time.time() - start_time
        print(f"\tDone loading Trembl\n\tElapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_clean_rnd_trembl, boundaries_all


class databaseCreater:
    """
    Class for creating the actual file for model evaluation. 
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
        boundaries_all,
    ):
        ##################################################################################
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
        self.boundaries_all = boundaries_all

        ##################################################################################

        self.seqarray_full = self._concat_double_delete()

        print("after concat", self.seqarray_full)
        print(len(self.boundaries_all), len(self.seqarray_full))

        self.sliding = self._sliding_window(
            self.seqarray_full,
            self.dimension_positive,
            (self.dimension_positive - self.stepsize),
        )

        print("after sliding", self.sliding)

        self.seqarray_multiplied = self._multiplier(self.seqarray_full, self.sliding)

        self.seqarray_final = self._overlapCalculater(self.seqarray_multiplied)

        self._saver()

    def _concat_double_delete(self):
        """
        Concatinates and deletes identical entries to one df. 
        Categories are added: 0 = target domain, 1 = rnd domains, 2 = rnd protein sequences
        Returns the full df with all sequences
        """
        start_time = time.time()
        seq_labels_positive1= self.seqarray_clean1[1:]
        seq_labels_positive2= self.seqarray_clean2[1:]


        seq_labels_positive1.loc[:, "categories"] = 0
        seq_labels_positive2.loc[:, "categories"] = 1
        seq_labels_negative_domains = pd.concat(
            (
                self.seqarray_clean_PF00079,
                self.seqarray_clean_PF00080,
                self.seqarray_clean_PF00118,
                self.seqarray_clean_PF00162,
            )
        )
        seq_labels_negative_domains = seq_labels_negative_domains[1:]
        seq_labels_negative_domains.loc[:, "categories"] = 2

        seqarray_clean_rnd_all = self.seqarray_clean_rnd_all[1:]

        seqarray_clean_rnd_all.loc[:, "categories"] = 3
        seq_labels_all_domains = pd.concat(
            [seq_labels_positive1, seq_labels_positive2, seq_labels_negative_domains]
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
        categories = []
        ids = []
        boundaries_all = []
        window_positions = []

        category_index = 0

        for nested_list in sliding:
            current_row = seqarray_full.iloc[category_index]
            current_category = current_row["categories"]
            current_id = current_row["ID"]


            try:
                current_boundary = current_row["Boundaries"]
                # print("current boundary is list", current_boundary)
            except:
                current_boundary = None
                pass

            len_nested = len(nested_list)

            for i in range(len_nested):
                seq = nested_list[i]
                sequences.append(seq)
                categories.append(current_category)
                ids.append(current_id)
                try:
                    if current_boundary:
                        boundaries_all.append(current_boundary)
                except:
                    pass
            

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

        if boundaries_all:
            sliding_df = pd.DataFrame(
                {
                    "Sequences": sequences,
                    "categories": categories,
                    "ID": ids,
                    "Boundaries": boundaries_all,
                    "WindowPos": window_positions,
                }
            )
        else:
            sliding_df = pd.DataFrame(
                {
                    "Sequences": sequences,
                    "categories": categories,
                    "ID": ids,
                    "WindowPos": window_positions,
                }
            )

        elapsed_time = time.time() - start_time
        print(f"\t Done multiplying\n\t Elapsed Time: {elapsed_time:.4f} seconds")
        return sliding_df

    def _overlapCalculater(self, seqarray_multiplied,binary_threshold=None):
        """
        Calculates the maximum overlap percentage between the window and any of the boundaries.
        Stores the overlap percentage (0.0 to 1.0) in the 'overlap' column. 
        Returned is the same df as entered with the addition of the 'overlap' column
        """
        start_time = time.time()

        overlaps = []
        n = len(seqarray_multiplied)

        for idx in range(n):
            if idx % 100000 == 0 and idx > 0:
                print(f"Overlap check iteration: {idx}/{n}")

            try:
                row = seqarray_multiplied.iloc[idx]
                # parse window
                ws, we = map(int, row["WindowPos"].split("-"))
                window_length = we - ws

                # skip non-zero categories
                if row["categories"] != 0 and row["categories"] != 1:
                    overlaps.append(0.0)
                    continue

                max_overlap_pct = 0.0
                for br in row["Boundaries"].split(","):
                    bs, be = map(int, br.split("-"))
                    overlap = max(0, min(we, be) - max(ws, bs))
                    ref_len = min(window_length, be - bs)
                    if ref_len > 0:
                        max_overlap_pct = max(max_overlap_pct, overlap / ref_len)

                # decide what to store
                if binary_threshold is not None:
                    val = 1.0 if max_overlap_pct >= binary_threshold else 0.0
                else:
                    val = round(max_overlap_pct, 4)

                overlaps.append(val)

            except Exception:
                overlaps.append(0.0)

        seqarray_multiplied["overlap"] = overlaps
        elapsed_time = time.time() - start_time
        print(f"\tDone checking overlap in {elapsed_time:.2f}s")
        return seqarray_multiplied


    def _saver(self):
        """
        Saves the final df in a .csv file. Name of file is hardcoded
        """
        start_time = time.time()
        print("Final array:", self.seqarray_final)
        self.seqarray_final.to_csv(
            "DataEvalSwissProt2d.csv", index=False
        )  # hardcoded filename
        elapsed_time = time.time() - start_time
        print(f"\tDone saving\n\tElapsed Time: {elapsed_time:.4f} seconds")


##################################################################################################################################################

###### FOR CREATING FULL SEQUENCE DATASET, FOR EVALUTATING PERFORMANCE ######

if __name__ == "__main__":
    # positive Domain PF00177
    print("Loading positive domain PF00177")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00177.fasta"
    )
    seqarray_clean1, boundaries_allPF00177 = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive1 = fasta.dimension_finder(fasta.len_finder())
    print("targeted dimension", dimension_positive1)


    # 2nd positive Domain PF00210
    print("Loading positive domain PF00210")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00210.fasta"
    )
    seqarray_clean2, boundaries_allPF00195 = fasta.distribution_finder(
        fasta.len_finder()
    )
    dimension_positive2 = fasta.dimension_finder(fasta.len_finder())
    print("targeted dimension", dimension_positive1)


    dimension_positive = max(dimension_positive1, dimension_positive2)



    # negative Domains:
    print("Loading negative PF00079")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00079.fasta"
    )
    seqarray_clean_PF00079, boundaries_allPF00079 = fasta.distribution_finder(
        fasta.len_finder()
    )
    print("Loading negative PF00080")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00080.fasta"
    )
    seqarray_clean_PF00080, boundaries_allPF00080 = fasta.distribution_finder(
        fasta.len_finder()
    )
    print("Loading negative PF00118")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00118.fasta"
    )
    seqarray_clean_PF00118, boundaries_allPF00118 = fasta.distribution_finder(
        fasta.len_finder()
    )
    print("Loading negative PF00162")
    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00162.fasta"
    )
    seqarray_clean_PF00162, boundaries_allPF00162 = fasta.distribution_finder(
        fasta.len_finder()
    )

    # load in swissprot and trembl

    fasta = DomainProcessing(
        "/global/research/students/sapelt/Masters/rawuniprot_sprot.fasta"
    )
    seqarray_clean_rnd_sprot, boundaries_allSwissprot = fasta._load_in_SwissProt()

    # print("Loading trembl")
    # fasta = DomainProcessing(
    #     "/global/research/students/sapelt/Masters/rawuniprot_trembl.fasta"
    # )
    # seqarray_clean_rnd_trembl, boundaries_allTrembl = fasta._load_in_Trembl()

    boundaries_all = [
        boundaries_allPF00177,
        boundaries_allPF00195,
        boundaries_allPF00079,
        boundaries_allPF00080,
        boundaries_allPF00118,
        boundaries_allPF00162,
        boundaries_allSwissprot,
        # boundaries_allTrembl,
    ]
    boundaries_all = [item for sublist in boundaries_all for item in sublist]

    ################### Data creation ########################
    print("Starting data creation for SwissProt validation set")
    dataset = databaseCreater(
        seqarray_clean1,
        seqarray_clean2,
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
        seqarray_clean_rnd_sprot,
        # seqarray_clean_rnd_trembl,
        dimension_positive,  
        0,
        boundaries_all,
    )
    ##############################################################
    print("All done creating evaluation dataset with full sequences")
