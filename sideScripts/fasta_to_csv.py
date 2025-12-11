"""
fasta_to_csv.py

This script converts a FASTA file into a CSV format. Each entry in the FASTA file is transformed into a row in the CSV, with columns for the sequence ID, sequence, and optionally the full description.

Functions:
=========================
fasta_to_csv()

"""

# -------------------------
# Imports and Globals
# -------------------------
import csv

from Bio import SeqIO

FASTA = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/NO_DOMAINS_SWISSPROT.fasta"
CSV = "/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/NO_DOMAINS_SWISSPROT.csv"
INCLUDE_DESCRIPTION = False


# -------------------------
# Function
# -------------------------
def fasta_to_csv(fasta_path, csv_path, include_description=False):
    """
    Converts a FASTA file to a CSV with columns: id, sequence[, description].

    Args:
        fasta_path (str): Path to input FASTA file.
        csv_path (str): Path to output CSV file.
        include_description (bool): Whether to include the full FASTA header as a column.
    """
    # Open the CSV file for writing
    with open(csv_path, "w", newline="") as csvfile:
        # Define CSV columns
        fieldnames = ["ID", "Sequence", "taxid"]
        # Add description column if needed
        if include_description:
            fieldnames.append("description")

        # Create CSV writer
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write header
        writer.writeheader()

        # Parse the FASTA file and write each record to the CSV
        for record in SeqIO.parse(fasta_path, "fasta"):
            # empty taxid string by default
            taxid_val = ""
            # Extract taxid if present in description
            if "OX=" in record.description:
                # trying default parsing
                try:
                    taxid_val = record.description.split("OX=")[1].split(" ")[0]
                # fallback parsing case
                except IndexError:
                    taxid_val = ""

            # Clean ID if it contains pipe characters
            if "|" in record.id:
                parts = record.id.split("|")
                # take second part as ID as its usually the ID
                if len(parts) >= 2:
                    record.id = parts[1]

            # Write row to CSV
            row = {"ID": record.id, "Sequence": str(record.seq), "taxid": taxid_val}
            # Add description if set
            if include_description:
                row["description"] = record.description
            # Write the row to the CSV file
            writer.writerow(row)

    # Final message
    print(f"Saved CSV to: {csv_path}")


#######################################
if __name__ == "__main__":
    fasta_to_csv(FASTA, CSV, include_description=INCLUDE_DESCRIPTION)
