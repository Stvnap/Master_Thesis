import csv
from Bio import SeqIO

def fasta_to_csv(fasta_path, csv_path, include_description=False):
    """
    Converts a FASTA file to a CSV with columns: id, sequence[, description].
    
    Args:
        fasta_path (str): Path to input FASTA file.
        csv_path (str): Path to output CSV file.
        include_description (bool): Whether to include the full FASTA header as a column.
    """
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'Sequence', 'taxid']
        if include_description:
            fieldnames.append('description')
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for record in SeqIO.parse(fasta_path, "fasta"):
            taxid_val = ''
            if "OX=" in record.description:
                # Extract the taxid from the description
                try:
                    taxid_val = record.description.split("OX=")[1].split(" ")[0]
                except IndexError:
                    taxid_val = '' # Handle cases where split fails

            if '|' in record.id:
                parts = record.id.split('|')
                if len(parts) >= 2:
                    record.id = parts[1]

            row = {
                'ID': record.id,
                'Sequence': str(record.seq),
                'taxid': taxid_val
            }
            if include_description:
                row['description'] = record.description
            writer.writerow(row)

    # print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    fasta_to_csv("/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/NO_DOMAINS_SWISSPROT.fasta", "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/NO_DOMAINS_SWISSPROT.csv", include_description=False)
