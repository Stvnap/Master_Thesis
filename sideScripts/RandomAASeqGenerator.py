
import random
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class RandomAASeqGenerator:
    def __init__(self, loop, length_distribution_file=None, max_length=1000,amino_acid_probs=None):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # Standard 20 amino acids
        self.loop = loop
        self.AAList = []
        self.max_length = max_length
        self.amino_acid_probs = amino_acid_probs

        # Load length distribution if provided
        df = pd.read_csv(length_distribution_file)
        # Get the lengths from the original data
        original_lengths = df['Length'].values.astype(int)
        
        # Filter out sequences longer than max_length
        original_lengths = original_lengths[original_lengths <= max_length]
        
        # Get unique lengths and their counts
        unique_lengths, counts = np.unique(original_lengths, return_counts=True)
        
        # Find the maximum count (target for even distribution)
        max_count = np.max(counts)
        
        # Create a complete range from min to max length
        min_length = unique_lengths.min()
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
        
        self.lengths = np.array(balanced_lengths)
        
        print(f"Original data: {len(original_lengths)} sequences")
        print(f"Length range in original data: {unique_lengths.min()} - {unique_lengths.max()}")
        print(f"Max count for any length: {max_count}")
        print(f"Sequences to generate for balancing: {len(self.lengths)}")

    def get_random_length(self):
        # Draw from the distribution with replacement
        return np.random.choice(self.lengths)

    def generate_sequence(self):
        for _ in range(self.loop):
            length = self.get_random_length()
            if length <= 0:
                continue
            self.AAList.append(''.join(random.choices(self.amino_acids, weights=[self.amino_acid_probs[aa] for aa in self.amino_acids], k=length)))
        return self.AAList


#################

def AminoAciderWeights():
    df = pd.read_csv("/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/NO_DOMAINS_SWISSPROT.csv")

    amino_acid_sequence = df['Sequence'].tolist()
    amino_acid_string = ''.join(amino_acid_sequence)
    amino_acid_counts = {aa: amino_acid_string.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    total_count = sum(amino_acid_counts.values())

    # calculate weights
    amino_acid_probabilities = {aa: count / total_count for aa, count in amino_acid_counts.items()}

    # check if probabilities sum to 1
    prob_sum = sum(amino_acid_probabilities.values())
    print(f"Sum of probabilities: {prob_sum}")
    if not np.isclose(prob_sum, 1.0):

        raise ValueError(f"Amino acid probabilities do not sum to 1, but to {prob_sum}")    

    print("Amino Acid Probabilities:")
    for aa, prob in amino_acid_probabilities.items():
        print(f"{aa}: {prob:.4f}")

    return amino_acid_probabilities


if __name__ == "__main__":

    amino_acid_probs = AminoAciderWeights()


    length_distribution_file='/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/domain_lengths.csv'
    
    # Example usage with distribution file
    generator = RandomAASeqGenerator(
        loop=200000, 
        length_distribution_file=length_distribution_file,
        max_length=1000,
        amino_acid_probs=amino_acid_probs

    )
    random_sequence = generator.generate_sequence()

    # Extract lengths from generated sequences
    generated_lengths = [len(seq) for seq in random_sequence]
    
    # Load original data for comparison
    # Load length distribution if provided
    df = pd.read_csv(length_distribution_file)
    # Get the lengths from the original data
    original_lengths = df['Length'].values.astype(int)
    original_lengths = original_lengths[original_lengths <= 1000]


    # Create comparison plot
    plt.figure(figsize=(8, 5))
    
    # Create stacked histogram
    plt.hist([original_lengths, generated_lengths], 
             color=[cm.get_cmap('tab10').colors[0], cm.get_cmap('tab10').colors[1]],alpha=0.7, 
             edgecolor='black', bins=100, label=['Original Sequencesz', 'Generated Sequences'],
             stacked=True,density=False)
    
    plt.title('Length Distribution Comparison: Original vs Generated Sequences')
    plt.xlabel('Domain Length in residues')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/domain_length_comparison.png')
    plt.show()

    print(f"Original sequences: {len(original_lengths)}")
    print(f"Generated {len(random_sequence)} Random Amino Acid Sequences")
    print(f"Original length range: {min(original_lengths)} - {max(original_lengths)}")
    print(f"Generated length range: {min(generated_lengths)} - {max(generated_lengths)}")
    print(f"Total numbers of all sequences: {len(original_lengths) + len(random_sequence)}")

    # save generated sequences to one file as csv   
    pd.DataFrame(random_sequence, columns=['Sequence']).to_csv(
        '/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/random_AA_sequences.csv', 
        index=False
    )