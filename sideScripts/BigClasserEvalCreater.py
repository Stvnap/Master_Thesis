import os 
import pandas as pd

CSV_PATH = "/global/students/research/sapelt/Masters/MasterThesis/Dataframes/v3/RemainingEntriesCompleteProteins.csv"
START_COL = "start"
END_COL = "end"
ID_COL = "id"
PFAM_COL = "Pfam_id"
SEQUENCE_COL = "Sequence"

pd.set_option('display.max_rows', None)     # Show all rows in DataFrame



# -------------------------
# Helper Functions
# -------------------------

def load_csv():
    """
    Load the CSV file and return a DataFrame.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")
    
    print(f"Loading CSV from {CSV_PATH}")
    return pd.read_csv(CSV_PATH, usecols=[START_COL, END_COL, ID_COL, PFAM_COL, SEQUENCE_COL],index_col=False)

def ReDefiner(df):


    # Sort by ID and START (in case it's not already sorted)
    df = df.sort_values(by=[ID_COL, START_COL]).reset_index(drop=True)

    result = []
    grouped = df.groupby(ID_COL)

    for id_val, group in grouped:
        group = group.reset_index(drop=True)
        num_rows = len(group)

        for i, row in group.iterrows():
            sequence = row[SEQUENCE_COL]
            new_seq = ""

            if num_rows == 1:
                # Only one entry for this ID → keep entire sequence
                new_seq = sequence
            else:
                if i == 0:
                    # First entry: cut from 0 to start of next domain
                    next_start = group.loc[i + 1, START_COL]
                    new_seq = sequence[0:next_start]
                elif i < num_rows - 1:
                    # Middle entries: cut from end of previous to start of next
                    prev_end = group.loc[i - 1, END_COL]
                    next_start = group.loc[i + 1, START_COL]
                    new_seq = sequence[prev_end:next_start]
                else:
                    # Last entry: cut from end of previous to end of sequence
                    prev_end = group.loc[i - 1, END_COL]
                    new_seq = sequence[prev_end:]

            # Store trimmed row
            trimmed = row.copy()
            trimmed[SEQUENCE_COL] = new_seq
            result.append(trimmed)

    # Create final DataFrame
    trimmed_df = pd.DataFrame(result)
    print("Data trimmed")
    return trimmed_df


def save_to_csv(df, output_path):
    """
    Save the DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"DataFrame saved to {output_path}")

# -------------------------
# MAIN 
# -------------------------
def main():
    df = load_csv()
    print(f"Loaded DataFrame with {len(df)} entries")
    
    # Display the first few rows of the DataFrame
    # print(df.head(15))

    trimmed_df = ReDefiner(df)

    # print("\n",trimmed_df.head(15))

    # for row in trimmed_df.itertuples(index=False):
    #     print(len(row.Sequence))

    output_path = "/global/students/research/sapelt/Masters/MasterThesis/Dataframes/v3/RemainingEntriesCompleteProteins_Eval.csv"
    save_to_csv(trimmed_df, output_path)


########################################################################################################################
if __name__ == "__main__":
    main()  