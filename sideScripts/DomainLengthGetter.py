import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt



class DomainLengthGetter:
    def __init__(self, df):
        self.df = df

    def get_domain_lengths(self):
        # Calculate domain lengths
        self.length = self.df['Sequence'].str.len()
        return self.length
    
if __name__ == "__main__":

    save_path = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/domain_lengths.csv"

    # load in dataframe
    file_name = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/NO_DOMAINS_SWISSPROT.csv"
    # Total number of lines (excluding header)

    df = pd.read_csv(file_name)

    getter = DomainLengthGetter(df)
    lengths = getter.get_domain_lengths()
    # save lengths to a file with a clear column name
    pd.DataFrame({'Length': lengths}).to_csv(save_path, index=False)
    print("Domain lengths saved to '/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/tempTest/domain_lengths.csv'")


    # fig of length distribution
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=100 ,color=cm.get_cmap('tab10').colors[0], alpha=0.7, density=False)
    plt.title('Length Distribution of non Pfam Sequences from SwissProt')
    plt.xlabel('Domain Length in residues')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/domain_length_distribution.png')

