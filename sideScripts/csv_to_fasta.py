########## csv to fasta formater ###################

import pandas as pd

# Load the CSV file
df = pd.read_csv('/global/students/research/sapelt/Masters/MasterThesis/Dataframes/v3/FoundEntriesSwissProteins.csv') 

# Create a FASTA file
with open('/global/students/research/sapelt/Masters/MasterThesis/Dataframes/v3/FoundEntriesSwissProteins.fasta', 'w') as fasta_file:
    for index, row in df.iterrows():
        identifier = row['id']  # or use 'categories', or both if needed
        sequence = row['Sequence']
        
        # Write in FASTA format
        fasta_file.write(f">{identifier}\n")
        
        # Optional: wrap the sequence every 60 characters (FASTA convention)
        for i in range(0, len(sequence), 60):
            fasta_file.write(sequence[i:i+60] + '\n')