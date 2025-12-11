"""
RandomAASeqGenerator.py

Script to generate the class 0 for the final max model.
We use artificially generated random amino acid sequences and sequences from uniprot without annotated domains by any domain database.
To not introduce any bias in the length distribution of the generated sequences we draw lengths from the original domain length distribution and generate sequences to create a homogenetic length distribution up to 1000 residues.

Table of Contents:
=========================
RandomAASeqGenerator
AminoAciderWeights()
Main()
"""
# -------------------------
# Imports & Globals
# -------------------------

import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

NO_DOMAIN_PATH = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/NO_DOMAINS_SWISSPROT.csv"
MAX_LENGTH = 1000
N_ART_SEQ = 200000

# -------------------------
# Class
# -------------------------
class RandomAASeqGenerator:
    """
    
    """
    def __init__(self, loop, original_lengths, max_length=1000):
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # Standard 20 amino acids
        # init parameters
        self.loop = loop
        self.AAList = []
        self.max_length = max_length
        self.amino_acid_probs = self._AminoAciderWeights()

        # Filter out sequences longer than max_length
        original_lengths = original_lengths[original_lengths <= max_length]

        # Get unique lengths and their counts
        unique_lengths, counts = np.unique(original_lengths, return_counts=True)

        # Find the maximum count to balance to
        max_count = np.max(counts)

        # Create a complete range from min to max length with all possible lengths
        min_length = unique_lengths.min()
        # Create a range of all possible lengths from min to max length
        all_possible_lengths = np.arange(min_length, max_length + 1)

        # Create a balanced distribution
        balanced_lengths = []
        for length in all_possible_lengths:
            if length in unique_lengths:
                # Get current count for this length
                current_count = counts[unique_lengths == length][0]
            else:
                # Length doesn't exist in original data
                current_count = 0

            # Add (max_count - current_count) instances of this length
            missing_count = max_count - current_count
            balanced_lengths.extend([length] * missing_count)

        # lenghts to be drawn from
        self.lengths = np.array(balanced_lengths)

        print(f"Original data: {len(original_lengths)} sequences")
        print(
            f"Length range in original data: {unique_lengths.min()} - {unique_lengths.max()}"
        )
        print(f"Max count for any length: {max_count}")
        print(f"Sequences to generate for balancing: {len(self.lengths)}")
    
    
    def _AminoAciderWeights(self):
        """
        Calculate amino acid probabilities from the NO_DOMAIN_PATH CSV file.
        """
        # Read sequences from CSV
        df = pd.read_csv(NO_DOMAIN_PATH)

        # concatenate all sequences into a single string
        amino_acid_sequence = df["Sequence"].tolist()
        amino_acid_string = "".join(amino_acid_sequence)
        # count occurrences of each amino acid
        amino_acid_counts = {
            aa: amino_acid_string.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"
        }
        # Calculate total count of all amino acids
        total_count = sum(amino_acid_counts.values())

        # calculate weights
        amino_acid_probabilities = {
            aa: count / total_count for aa, count in amino_acid_counts.items()
        }

        # check if probabilities sum to 1 else raise error
        prob_sum = sum(amino_acid_probabilities.values())
        print(f"Sum of probabilities: {prob_sum}")
        if not np.isclose(prob_sum, 1.0):
            raise ValueError(f"Amino acid probabilities do not sum to 1, but to {prob_sum}")

        # final print
        print("Amino Acid Probabilities:")
        for aa, prob in amino_acid_probabilities.items():
            print(f"{aa}: {prob:.4f}")

        return amino_acid_probabilities


    def get_random_length(self):
        # Draw from the distribution with replacement
        return np.random.choice(self.lengths)

    def generate_sequence(self):
        """
        Generate random amino acid sequences
        """
        for _ in range(self.loop):
            # get random length for the sequence
            length = self.get_random_length()
            # Skip if length is non-positive
            if length <= 0:
                continue
            # Generate random sequence based on weighed amino acid probabilities
            self.AAList.append(
                "".join(
                    random.choices(
                        self.amino_acids,
                        weights=[self.amino_acid_probs[aa] for aa in self.amino_acids],
                        k=length,
                    )
                )
            )
        return self.AAList


# -------------------------
# Functions
# -------------------------

def length_calculter(path):
    df = pd.read_csv(path)
    legths = df["Sequence"].apply(len)
    return legths

def plotter_saver(random_sequence, lengths):
    """
    Plots the length distribution comparison between original and generated sequences in a stacked histogram.
    """
    # Extract lengths from generated sequences
    generated_lengths = [len(seq) for seq in random_sequence]

    # Filter lengths to be within MAX_LENGTH
    lengths = lengths[lengths <= MAX_LENGTH]

    # Create DataFrame for seaborn - CHANGED ORDER HERE
    length_data = pd.DataFrame({
        'Length': np.concatenate([generated_lengths, lengths]),  # Generated first, then original
        'Type': ['Generated Sequences'] * len(generated_lengths) + ['Original Sequences'] * len(lengths)  # Generated first, then original
    })

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=length_data,
        x='Length',
        hue='Type',
        bins=100,
        alpha=0.7,
        edgecolor="black",
        multiple="stack",
        palette={'Original Sequences': sns.color_palette("tab10")[0], 
                'Generated Sequences': sns.color_palette("tab10")[1]},
        hue_order=['Generated Sequences', 'Original Sequences']  # Added this to control stacking order
    )
    
    plt.title("Length Distribution Comparison: Original vs Generated Sequences")
    plt.xlabel("Domain Length in Residues")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/domain_length_comparison.png", dpi=600
    )
    plt.show()
    
    # Print summary
    print(f"Original sequences: {len(lengths)}")
    print(f"Generated {len(random_sequence)} Random Amino Acid Sequences")
    print(f"Original length range: {min(lengths)} - {max(lengths)}")
    print(
        f"Generated length range: {min(generated_lengths)} - {max(generated_lengths)}"
    )
    print(f"Total numbers of all sequences: {len(lengths) + len(random_sequence)}")

    # save generated sequences to one file as csv
    pd.DataFrame(random_sequence, columns=["Sequence"]).to_csv(
        "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/random_AA_sequences.csv",
        index=False,
    )

# -------------------------
# Main
# -------------------------

def main():
    # get lengths from original data
    lengths = length_calculter(NO_DOMAIN_PATH)

    # generate random sequences based on lengths
    generator = RandomAASeqGenerator(
        loop=N_ART_SEQ,
        original_lengths=lengths,
        max_length=MAX_LENGTH,
    )
    random_sequence = generator.generate_sequence()

    # plot and save results
    plotter_saver(random_sequence, lengths)

#################################################
if __name__ == "__main__":
    main()


