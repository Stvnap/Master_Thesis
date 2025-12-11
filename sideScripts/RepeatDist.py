"""
RepeatDist.py

Analysis script to determine the distribution of repeat entries in the Pfam database.
Used to analyze the reason for low recall values in the 1000d model.
See if repeat regions are overrepresented in the low-recall classes compared to overall Pfam database.

Table of Contents:
=====================
1. RepeatAnalysis
"""

# ------------------
# Imports & Globals
# ------------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

PFAM_A_HMM_DAT = "/global/research/students/sapelt/Masters/Pfam-A.hmm.dat"
EVAL_FILE = "/global/research/students/sapelt/Masters/MasterThesis/shells/EVAL_1000.out"
PFAM_DICT_FILE = "/global/research/students/sapelt/Masters/pfamdict1000.txt"

THRESH_PRECISION = 0.05
THRESH_RECALL = 0.05


# ------------------
# Class
# ------------------


class RepeatAnalysis:
    """
    Class to perform repeat analysis on Pfam data and model evaluation data.
    Functions:
    - pfam_repeat_analysis: Analyzes the Pfam database for repeat entries.
    - model_repeat_analysis: Analyzes the evaluation output for low-recall and low-precision classes.
    - plotter: Plots the results of the analysis.
    """

    def __init__(self):
        # Initialize variables for repeat analysis
        self.pfam_data_list = []
        self.entry_count = 0
        self.repeat_count = 0
        self.all_ml = []
        self.number_of_found_repeats = 0
        self.ml_list = []
        self.pfamID_found = []

        # Run analyses directly
        self.pfam_repeat_analysis()
        self.model_repeat_analysis()
        self.plotter()

    def pfam_repeat_analysis(self):
        """
        Analyzes the Pfam database for repeat entries. Uses the Pfam-A.hmm.dat file for this and searches entries with repeat type.
        Also calculates average model length (ML) of all repeat entries.
        """
        with open(PFAM_A_HMM_DAT, "r") as f:
            # read
            pfam_data = f.read()
            # init vars
            self.entry_count = 0
            self.repeat_count = 0
            self.all_ml = []
            # split entries by '//\n'
            self.pfam_data_list = pfam_data.split("//\n")
            # loop through all entries
            for idx, entry in enumerate(self.pfam_data_list):
                # add to entry count
                self.entry_count += 1
                # loop through each line of entry
                for idx2, line in enumerate(entry.splitlines()):
                    # if it is repeat
                    if line.startswith("#=GF TP   Repeat"):
                        # add counter
                        self.repeat_count += 1
                        # get model length
                        ml = self.pfam_data_list[idx].splitlines()
                        ml = ml[6].split()[2]
                        # store ml
                        self.all_ml.append(int(ml))
                        # exit inner line loop
                        break

            # final prints
            print("Total entries:", self.entry_count)
            print("Total repeat entries:", self.repeat_count)
            print("Fraction of repeat entries:", self.repeat_count / self.entry_count)
            print("Average ML of repeat entries:", np.mean(self.all_ml), "\n")

    def model_repeat_analysis(self):
        """
        Looks up the Pfam IDs of low-recall and low-precision classes in the resulting Evaluation output.
        Checks how many of these IDs are of repeat type in the Pfam database.
        Saves the hmm model lengths (ML) of the found repeat entries for further analysis.
        """

        with open(EVAL_FILE, "r") as f:
            # read eval file
            hmm_data = f.read()
            # split lines and init lists
            hmm_lines = hmm_data.splitlines()
            prec_under = []
            rec_under = []

            # loop through lines of eval file to find low-precision and low-recall classes
            for line in hmm_lines:
                # start string for individual class results
                if line.startswith("Mean precision for"):
                    # get precision and recall values
                    prec = line.split()[5].rstrip(",")
                    rec = line.split()[8]
                    # check thresholds & corresponding class IDs
                    if float(prec) < THRESH_PRECISION:
                        prec_under.append(int(line.split()[4].rstrip(":")))
                    if float(rec) < THRESH_RECALL:
                        rec_under.append(int(line.split()[4].rstrip(":")))

        # Load pfamdict to map class numbers to Pfam IDs
        with open(PFAM_DICT_FILE, "r") as f:
            # read & init
            pfamdict_data = f.read()
            pfamdict = []
            # dummy for align 0 index
            pfamdict.append("DUMMY")
            # get lines
            pfamdict_lines = pfamdict_data.splitlines()
            # loop through lines and store Pfam IDs
            for line in pfamdict_lines:
                pfamdict.append(line.split()[0])

        # init lists to hold found Pfam IDs
        self.pfamID_found_model_rec = []
        self.pfam_found_model_prec = []

        # lookup Pfam IDs for low-recall and low-precision classes
        for rec in rec_under:
            self.pfamID_found_model_rec.append(pfamdict[int(rec)])
        for prec in prec_under:
            self.pfam_found_model_prec.append(pfamdict[int(prec)])

        # prints how many Pfam IDs found
        print(
            f"\nTotal Pfam IDs found in low-recall classes ({THRESH_RECALL}):",
            len(self.pfamID_found_model_rec),
        )
        print(
            f"Total Pfam IDs found in low-precision classes ({THRESH_PRECISION}):",
            len(self.pfam_found_model_prec),
        )

        # lookup in hmm.dat and check if type is actually repeat for found recall Pfam IDs
        # init counters and lists
        self.number_of_found_repeats_rec = 0
        self.ml_list_model_rec = []
        # loop through found Pfam IDs
        for pfamID in self.pfamID_found_model_rec:
            # loop through all entries in hmm.dat
            for idx, entry in enumerate(self.pfam_data_list):
                # loop through lines of entry
                for line in entry.splitlines():
                    # check for matching Pfam ID
                    if line.startswith(f"#=GF AC   {pfamID.split('.')[0]}"):
                        # get IDs for printing/debugging
                        id = line.split()[2]
                        id = id.split(".")[0]
                        # check if type is repeat
                        if entry.splitlines()[5].startswith("#=GF TP   Repeat"):
                            # print(f"ID {id} is of type {type} and is a repeat.")
                            # add counter and store ML
                            self.number_of_found_repeats_rec += 1
                            ml = entry.splitlines()[6].split()[2]
                            self.ml_list_model_rec.append(int(ml))

        # lookup in hmm.dat and check if type is actually repeat for found precision Pfam IDs
        # init counters and lists
        self.number_of_found_repeats_prec = 0
        self.ml_list_model_prec = []
        # loop through found Pfam IDs
        for pfamID in self.pfam_found_model_prec:
            # loop through all entries in hmm.dat
            for idx, entry in enumerate(self.pfam_data_list):
                # loop through lines of entry
                for line in entry.splitlines():
                    # check for matching Pfam ID
                    if line.startswith(f"#=GF AC   {pfamID.split('.')[0]}"):
                        # get IDs for printing/debugging
                        id = line.split()[2]
                        id = id.split(".")[0]
                        # check if type is repeat
                        if entry.splitlines()[5].startswith("#=GF TP   Repeat"):
                            # print(f"ID {id} is of type {type} and is a repeat.")
                            # add counter and store ML
                            self.number_of_found_repeats_prec += 1
                            ml = entry.splitlines()[6].split()[2]
                            self.ml_list_model_prec.append(int(ml))

        # final prints
        print("\nNumber of found repeats (recall):", self.number_of_found_repeats_rec)
        print("Total Pfam IDs searched (recall):", len(self.pfamID_found_model_rec))
        print("Total Pfam IDs searched (precision):", len(self.pfam_found_model_prec))
        print(
            "Fraction of found repeats (recall):",
            self.number_of_found_repeats_rec / len(self.pfamID_found_model_rec),
        )
        print(
            "Fraction of found repeats (precision):",
            self.number_of_found_repeats_prec / len(self.pfam_found_model_prec),
        )
        print("Average ML of found repeats (recall):", np.mean(self.ml_list_model_rec))
        print(
            "Average ML of found repeats (precision):", np.mean(self.ml_list_model_prec)
        )

    def plotter(self):
        """
        Plots the results of the repeat analysis.
        Creates bar plots comparing overall Pfam repeat fraction to found repeat fractions in low-recall and low-precision classes.
        Also creates violin plots showing the distribution of model lengths (ML) for all Pfam repeats and found repeats.
        """

        # Get colors
        colors = cm.get_cmap("tab10").colors

        # Create plot for Recall
        # get fractions for 2 bars
        fractions = [
            self.repeat_count / self.entry_count,
            self.number_of_found_repeats_rec / len(self.pfamID_found_model_rec),
        ]
        # set labels
        labels = [
            "Overall Pfam Repeat Fraction",
            f"Found Repeats Fraction < {THRESH_RECALL} recall",
        ]

        # build bar plot for recall
        plt.figure(figsize=(10, 6))
        plt.bar(labels, fractions, color=colors[:2], alpha=0.7, width=0.5)
        plt.ylabel("Fraction")
        plt.title("Comparison of Repeat Fractions in 1000d Model")
        plt.ylim(0, 1)
        # Add value labels on top of bars
        for i, v in enumerate(fractions):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig('/home/sapelt/Documents/Master/FINAL/Comparison of Repeat Fractions in 1000d Model - Recall.png', dpi=600)
        plt.show()

        # Create plot for Precision
        # get fractions for 2 bars
        fractions = [
            self.repeat_count / self.entry_count,
            self.number_of_found_repeats_prec / len(self.pfam_found_model_prec),
        ]
        # set labels
        labels = [
            "Overall Pfam Repeat Fraction",
            f"Found Repeats Fraction < {THRESH_PRECISION} precision",
        ]

        # build bar plot for precision
        plt.figure(figsize=(10, 6))
        plt.bar(labels, fractions, color=colors[:2], alpha=0.7, width=0.5)
        plt.ylabel("Fraction")
        plt.title("Comparison of Repeat Fractions in 1000d Model")
        plt.ylim(0, 1)
        # Add value labels on top of bars
        for i, v in enumerate(fractions):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig('/home/sapelt/Documents/Master/FINAL/Comparison of Repeat Fractions in 1000d Model - Precision.png', dpi=600)
        plt.show()

        # Create a violin plot showing both ML distributions (without outliers)
        def remove_outliers(data):
            """
            Function to remove outliers from a dataset using the IQR method.
            Args:
                data (list): List of numerical values.
            Returns:
                cleaned_data (list): List with outliers removed.
            """
            # get Q1 and Q3
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            # calculate IQR and bounds
            IQR = Q3 - Q1
            # set bounds standard 1.5*IQR
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return [x for x in data if lower_bound <= x <= upper_bound]

        # Remove outliers from all datasets
        all_ml_clean = remove_outliers(self.all_ml)
        ml_list_clean_rec = remove_outliers(self.ml_list_model_rec)
        ml_list_clean_prec = remove_outliers(self.ml_list_model_prec)

        # Create combined data with pfam data and model data
        ml_data_rec = [all_ml_clean, ml_list_clean_rec]
        ml_data_prec = [all_ml_clean, ml_list_clean_prec]

        # Create a violin plot showing both ML distributions (without outliers) for recall
        plt.figure(figsize=(10, 6))
        parts = plt.violinplot(
            ml_data_rec, positions=[1, 2], showmeans=True, showmedians=True
        )
        # Colors
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        for part in ['cmaxes', 'cmins', 'cbars','cmeans','cmedians']:
            if part in parts:
                parts[part].set_color('dimgray')

        # Add mean values as text labels
        means = [np.mean(data) for data in ml_data_rec]
        for i, mean_val in enumerate(means):
            y_pos = (plt.ylim()[0] + plt.ylim()[1]) / 2
            plt.text(
                i + 1,
                mean_val,
                f"{mean_val:.1f}",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
        ml_labels = ["All Pfam Repeats", f"Found Repeats < {THRESH_RECALL} recall"]
        plt.xticks([1, 2], ml_labels)
        plt.ylabel("Model Length (ML)")
        plt.title("Distribution of HHM Lengths in 1000d Model (Outliers Removed)")
        plt.tight_layout()
        plt.savefig('/home/sapelt/Documents/Master/FINAL/Distribution of HHM Lengths in 1000d Model - Recall.png', dpi=600)
        plt.show()

        # Create a violin plot showing both ML distributions (without outliers) for precision
        plt.figure(figsize=(10, 6))
        parts = plt.violinplot(
            ml_data_prec, positions=[1, 2], showmeans=True, showmedians=True
        )

        # Color the violin plots
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        for part in ['cmaxes', 'cmins', 'cbars','cmeans','cmedians']:
            if part in parts:
                parts[part].set_color('dimgray')

        # Add mean values as text labels
        means = [np.mean(data) for data in ml_data_prec]
        for i, mean_val in enumerate(means):
            y_pos = (plt.ylim()[0] + plt.ylim()[1]) / 2
            plt.text(
            i + 1,
            y_pos,
            f"{mean_val:.1f}",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
        ml_labels = [
            "All Pfam Repeats",
            f"Found Repeats < {THRESH_PRECISION} precision",
        ]
        plt.xticks([1, 2], ml_labels)
        plt.ylabel("Model Length (ML)")
        plt.title("Distribution of HHM Lengths in 1000d Model (Outliers Removed)")
        plt.tight_layout()
        plt.savefig('/home/sapelt/Documents/Master/FINAL/Distribution of HHM Lengths in 1000d Model - Precision.png', dpi=600)
        plt.show()


###################################
if __name__ == "__main__":
    RepeatAnalysis()
