import pandas as pd
import random


LARGECSV = "/scratch/tmp/sapelt/Master_Thesis/Dataframes/v3/RemainingEntriesCompleteProteins.csv"
SMALLCSV = "/scratch/tmp/sapelt/Master_Thesis/Dataframes/v3/Unknown_class_train.csv"
OUTPUT = "/scratch/tmp/sapelt/Master_Thesis/Dataframes/v3/RemainingEntriesCompleteProteins_MAX.csv"

import pandas as pd



# Load both files
df_large = pd.read_csv(LARGECSV)
df_small = pd.read_csv(SMALLCSV)

# Get the union of all columns
all_columns = list(df_large.columns) + [c for c in df_small.columns if c not in df_large.columns]


print(all_columns)

# Reindex both DataFrames to have the same columns (missing ones get filled with empty string)
df_large = df_large.reindex(columns=all_columns, fill_value="")
df_small = df_small.reindex(columns=all_columns, fill_value="")

# Concatenate (append)
combined = pd.concat([df_large, df_small], ignore_index=True)

# Free up memory
del df_large, df_small

# Shuffle the combined DataFrame
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)


# Save to new file
combined.to_csv(OUTPUT, index=False)

print(f"Combined file written to {OUTPUT}")