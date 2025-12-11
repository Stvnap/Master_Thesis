"""
DataAllCreater_part1.py


File for creating the whole Training Dataset in tf.Dataset format to be used for training
INFOS:
Use enough RAM for this script

Table of Contents:
=========================
1. _sequence_to_int
2. _padder
3. _labler
4. _labler2d
5. splitter2
6. make_dataset

"""

# -------------------------
# Imports & Globals
# -------------------------

import time

import polars as pl
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(  # Disable GPU, crashes on GPU
    [], "GPU"
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from Dataset_preprocess_TRAIN_2d import dimension_positive             # change here to 2d if u want to create a ds for multi categorial classification

DF_PATH = "./Dataframes/DataTrainALL.csv"
DIMENSION_POSITIVE = 148


# -------------------------
# Functions
# -------------------------


def _sequence_to_int(df):
    """
    Function to translate the sequences into a list of int
    Args:
        df: Polars DataFrame with column 'Sequences' containing amino acid sequences as strings
    Returns:
        Polars DataFrame with column 'Sequences' containing amino acid sequences as lists of integers
    """
    # start time
    start_time = time.time()

    # AA dictionary
    amino_acid_to_int = {
        "A": 1,  # Alanine
        "C": 2,  # Cysteine
        "D": 3,  # Aspartic Acid
        "E": 4,  # Glutamic Acid
        "F": 5,  # Phenylalanine
        "G": 6,  # Glycine
        "H": 7,  # Histidine
        "I": 8,  # Isoleucine
        "K": 9,  # Lysine
        "L": 10,  # Leucine
        "M": 11,  # Methionine
        "N": 12,  # Asparagine
        "P": 13,  # Proline
        "Q": 14,  # Glutamine
        "R": 15,  # Arginine
        "S": 16,  # Serine
        "T": 17,  # Threonine
        "V": 18,  # Valine
        "W": 19,  # Tryptophan
        "Y": 20,  # Tyrosine
        "X": 21,  # Unknown or special character            (21 for all other AA)
        "Z": 21,  # Glutamine or Glutamic Acid
        "B": 21,  # Asparagine or Aspartic Acid
        "U": 21,  # Selenocysteine
        "O": 21,  # Pyrrolysine
    }

    # drop nulls to avoid errors
    df = df.drop_nulls(subset=["Sequences"])

    def encode_sequence(seq):
        # functin to apply to each sequence
        return [amino_acid_to_int[amino_acid] for amino_acid in seq]

    # apply function
    df = df.with_columns(
        pl.col("Sequences")
        .map_elements(encode_sequence, return_dtype=pl.List(pl.Int16))
        .alias("Sequences")
    )

    # elapsed time and prints
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done encoding\nElapsed Time: {elapsed_time:.4f} seconds")

    return df


def _padder(df_int):
    """
    Pads the sequences to the target dimension with value 21 = unidentified aa
    Args:
        df_int: Polars DataFrame with column 'Sequences' containing amino acid sequences as lists
    Returns:
        Polars DataFrame with column 'Sequences' containing padded amino acid sequences as lists
    """
    # start time
    start_time = time.time()

    # get sequences as list
    sequences = df_int["Sequences"].to_list()

    # pad sequences with keras pad_sequences
    padded = pad_sequences(
        sequences,
        maxlen=DIMENSION_POSITIVE,  # set to dimension_positive
        padding="post",  # pad at the end
        truncating="post",  # truncate at the end
        value=21,  # pad with 21 = unidentified aa
    )

    # put back into df
    df_int = df_int.with_columns(pl.lit(padded).alias("Sequences"))

    # elapsed time and prints
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done padding\nElapsed Time: {elapsed_time:.4f} seconds")

    return df_int


def _labler(padded):
    """
    Creates a new column 'Labels' that translates the categories column to 1 = target domain, 0 = all other. Used for 1 domain classification
    Args:
        padded: Polars DataFrame with column 'categories' containing category labels
    Returns:
        Polars DataFrame with added 'Labels' column
    """
    # start time
    start_time = time.time()

    # create Labels column. 1 if category 0 (positive domain), else 0 (negative)
    padded = padded.with_columns(
        pl.when(pl.col("categories") == 0).then(1).otherwise(0).alias("Labels")
    )
    # drop categories column
    padded = padded.drop("categories")

    # elapsed time and prints
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done labeling\nElapsed Time: {elapsed_time:.4f} seconds")

    return padded


def _labler2d(padded):
    """
    Creates a new column 'Labels' that translates the categories column to 1 = target domain, 0 = all other. Used for 2 domain classification
    Args:
        padded: Polars DataFrame with column 'categories' containing category labels
    Returns:
        Polars DataFrame with added 'Labels' column
    """
    # start time
    start_time = time.time()

    # create Labels column. 1 if category 0 (positive domain), 2 if category 1 (second positive domain), else 0 (negative)
    padded = padded.with_columns(
        pl.when(pl.col("categories") == 0)
        .then(1)
        .when(pl.col("categories") == 1)
        .then(2)
        .otherwise(0)
        .alias("Labels")
    )

    # drop categories column
    padded = padded.drop("categories")

    # elapsed time and prints
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done labeling\nElapsed Time: {elapsed_time:.4f} seconds")

    return padded


def splitter2(padded_label):
    """
    Splits padded_label into train/val/test (60/20/20), grouping all rows with the same ID
    in the same split *and* stratifying by label.
    Writes each split to a parquet, and returns the three DataFrames.
    """
    start_time = time.time()

    # 1) Build an ID→label table (here taking the first label seen per ID)
    ids_df = (
        padded_label.select(["IDs", "Labels"])
        .unique(subset="IDs")  # keep one row per ID
        .rename({"Labels": "Label"})  # so column names don't collide
    )

    # Convert to Pandas for sklearn
    ids_pd = ids_df.to_pandas()
    id_values = ids_pd["IDs"].values
    label_vals = ids_pd["Label"].values

    # Split IDs → train vs temp (60% train, 40% temp), stratified by ID-label
    train_ids, temp_ids = train_test_split(
        id_values, test_size=0.4, stratify=label_vals, random_state=42
    )

    # From temp_ids split into val vs test (each half of the 40%), again stratified
    temp_labels = ids_pd.set_index("IDs").loc[temp_ids, "Label"].values
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Filter the original padded_label into three DataFrames
    train_df = padded_label.filter(pl.col("IDs").is_in(train_ids))
    val_df = padded_label.filter(pl.col("IDs").is_in(val_ids))
    test_df = padded_label.filter(pl.col("IDs").is_in(test_ids))

    # print shapes
    print(f"Train set shape:      {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape:       {test_df.shape}")

    # Write to parquet files
    train_df.write_parquet("trainsetALL.parquet")
    val_df.write_parquet("valsetALL.parquet")
    test_df.write_parquet("testsetALL.parquet")

    # elapsed time and prints
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done Splitting\nElapsed Time: {elapsed_time:.4f} seconds")

    return train_df, val_df, test_df


def make_dataset(
    df,
    save_path,
    shuffle_buffer: int = 1000,
):
    """
    Streams Sequences, Labels, and IDs out of the Polars DF one row at a time,
    pads to length dimension_positive, one‐hots to depth=21 (uint8), then shuffles & batches.
    Peak memory stays at ~batch_size×dimension_positive×21 elements.
    """

    def gen():
        """
        Generator reading each row of the df, yielding seq, lab, id
        """
        for seq, lab, id_ in zip(
            df["Sequences"].to_list(), df["Labels"].to_list(), df["ID"].to_list()
        ):
            yield seq, int(lab), id_

    # Predefine the shape of the ds and generate
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int32),  # seq (variable length)
            tf.TensorSpec(shape=(), dtype=tf.int32),  # label
            tf.TensorSpec(shape=(), dtype=tf.string),  # id
        ),
    )

    def pad_and_one_hot(seq, lab, id_):
        """
        Pad and optional onehot encode
        """
        seq = tf.pad(
            seq, [[0, DIMENSION_POSITIVE - tf.shape(seq)[0]]], constant_values=21
        )

        # oh = tf.one_hot(seq, depth=21, dtype=tf.uint8)
        # laboh = tf.one_hot(lab,depth=3,dtype=tf.uint8)

        return seq, lab, id_

    # apply to the ds
    ds = ds.map(pad_and_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle
    ds = ds.shuffle(shuffle_buffer)

    # Save
    if save_path:
        ds.save(save_path)
        print("tf saving done")

    # init lists to save as dataframe later
    inputs = []
    labels = []
    ids = []

    # loop through the dataset and save to lists
    for x, y, z in ds:
        inputs.append(x.numpy().flatten())
        labels.append(y.numpy().tolist())
        ids.append(z.numpy())

    # free memory
    del ds

    # create dataframe
    df = pl.DataFrame({"input": inputs, "label": labels, "id": ids})

    # flatten nested columns
    df_flat = df.with_columns(
        [
            # Convert integers in the "input" column to strings, then join them with commas
            pl.col("input")
            .map_elements(lambda lst: ",".join(map(str, lst)))
            .alias("input_str"),
            # Convert the binary 'id' column to a string
            pl.col("id").cast(pl.Utf8).alias("id_str"),
        ]
    )

    # Now drop the original nested column and rename the new columns
    df_flat = df_flat.drop("input")
    df_flat = df_flat.drop("id")
    df_flat = df_flat.rename({"input_str": "input", "id_str": "id"})

    # Write the DataFrame to a CSV file
    df_flat.write_csv("./saved_datasetNew.csv")

    # final prints
    print("Dataset built")
    print(df_flat.head())
    print(df_flat.schema)


def main():
    print("Starting data preparation...")

    # load in
    df = pl.read_csv(
        DF_PATH,
        schema_overrides={
            "Sequences": pl.Utf8,
            "categories": pl.Int8,
            "ID": pl.Utf8,
        },
    ).lazy()

    # encode
    df = _sequence_to_int(df)
    print("Done encoding")

    # pad
    df = _labler2d(df)
    print("Done labeling")

    # collect lazy df
    df = df.collect()

    # create dataset and save
    make_dataset(df, save_path="./saved_datasetNew")
    print("Data preparation completed.")


##################################################################################

if __name__ == "__main__":
    main()
