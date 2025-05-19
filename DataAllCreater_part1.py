###################################################################################################################################
"""
File for creating the whole Training Dataset in tf.Dataset format to be used for training

INFOS:
Use enough RAM for this script

"""
###################################################################################################################################

import time

import polars as pl
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(  # Disable GPU, crashes on GPU
    [], "GPU"
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from Dataset_preprocess_TRAIN import dimension_positive             # change here to 2d if u want to create a ds for multi categorial classification

###################################################################################################################################
dimension_positive=148

def _sequence_to_int(df):
    """
    Function to translate the sequences into a list of int
    Returns the translated df
    """
    start_time = time.time()

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

    df = df.drop_nulls(subset=["Sequences"])

    def encode_sequence(seq):
        return [amino_acid_to_int[amino_acid] for amino_acid in seq]

    df = df.with_columns(
        pl.col("Sequences")
        .map_elements(encode_sequence, return_dtype=pl.List(pl.Int16))
        .alias("Sequences")
    )
    # print(self.df)
    # print(type(self.df))
    # print(self.df['Sequences'][0])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done encoding\nElapsed Time: {elapsed_time:.4f} seconds")
    return df


def _padder(df_int):
    """
    Pads the sequences to the target dimension with value 21 = unidentified aa
    Returns the padded df
    """
    start_time = time.time()
    sequences = df_int["Sequences"].to_list()
    # print(type(sequences))
    # print(sequences[0:3])
    # print(self.target_dimension)
    padded = pad_sequences(
        sequences,
        maxlen=dimension_positive,
        padding="post",
        truncating="post",
        value=21,
    )
    # print(padded)
    df_int = df_int.with_columns(pl.lit(padded).alias("Sequences"))
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Done padding\nElapsed Time: {elapsed_time:.4f} seconds")
    print(df_int)

    return df_int


def _labler(padded):
    """
    Creates a new column 'Labels' that translates the categories column to 1 = target domain, 0 = all other
    Returns the df with added 'Labels' column
    """
    start_time = time.time()
    padded = padded.with_columns(
        pl.when(pl.col("categories") == 0).then(1).otherwise(0).alias("Labels")
    )
    padded_label = padded
    padded_label = padded_label.drop("categories")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done labeling\nElapsed Time: {elapsed_time:.4f} seconds")
    return padded_label

def _labler2d(padded):
    """
    Creates a new column 'Labels' that translates the categories column to 1 = target domain, 0 = all other
    Returns the df with added 'Labels' column
    """
    start_time = time.time()
    padded = padded.with_columns(
        pl.when(pl.col("categories") == 0).then(1)
        .when(pl.col("categories") == 1).then(2)
        .otherwise(0)
        .alias("Labels")
    )
    padded_label = padded
    padded_label = padded_label.drop("categories")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done labeling\nElapsed Time: {elapsed_time:.4f} seconds")
    return padded_label


def splitter2(padded_label):
    """
    Splits padded_label into train/val/test (60/20/20), grouping all rows with the same ID
    in the same split *and* stratifying by label.
    Writes each split to a parquet, and returns the three DataFrames.
    """
    t0 = time.time()

    # 1) Build an ID→label table (here taking the first label seen per ID)
    ids_df = (
        padded_label.select(["IDs", "Labels"])
        .unique(subset="IDs")  # keep one row per ID
        # .groupby("IDs").agg(pl.col("Labels").mode().first())  # if you want the mode instead
        .rename({"Labels": "Label"})  # so column names don't collide
    )
    # Convert to Pandas for sklearn
    ids_pd = ids_df.to_pandas()
    id_values = ids_pd["IDs"].values
    label_vals = ids_pd["Label"].values

    # 2) Split IDs → train vs temp (60% train, 40% temp), stratified by ID-label
    train_ids, temp_ids = train_test_split(
        id_values, test_size=0.4, stratify=label_vals, random_state=42
    )

    # 3) From temp_ids split into val vs test (each half of the 40%), again stratified
    temp_labels = ids_pd.set_index("IDs").loc[temp_ids, "Label"].values
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # 4) Filter the original padded_label into three DataFrames
    train_df = padded_label.filter(pl.col("IDs").is_in(train_ids))
    val_df = padded_label.filter(pl.col("IDs").is_in(val_ids))
    test_df = padded_label.filter(pl.col("IDs").is_in(test_ids))

    # 5) Report shapes
    print(f"Train set shape:      {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape:       {test_df.shape}")

    # 6) Write to parquet
    train_df.write_parquet("trainsetALL.parquet")
    val_df.write_parquet("valsetALL.parquet")
    test_df.write_parquet("testsetALL.parquet")

    print(f"Done splitting in {time.time() - t0:.3f}s")

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
        Generator reading each row of the df
        """
        for seq, lab, id_ in zip(
            df["Sequences"].to_list(), df["Labels"].to_list(), df["ID"].to_list()
        ):
            yield seq, int(lab), id_

    # Predefine the shape of the ds
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
        Pad and onehot encode
        """
        seq = tf.pad(seq, [[0, dimension_positive - tf.shape(seq)[0]]], constant_values=21)
        oh = tf.one_hot(seq, depth=21, dtype=tf.uint8)
        return oh, lab, id_

    # apply to the ds
    ds = ds.map(pad_and_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle
    ds = ds.shuffle(shuffle_buffer)

    # Save
    if save_path:
        ds.save(save_path)

    print("Dataset built")
    return ds


##################################################################################

if __name__ == "__main__":
    print("Starting data preparation...")
    df_path = "./DataTrainSwissProt.csv"

    # load in
    df = pl.read_csv(
        df_path,
        schema_overrides={
            "Sequences": pl.Utf8,
            "categories": pl.Int8,
            # "ID": pl.Utf8,
        },
        # n_rows=100000,
    ).lazy()


    df = df.with_columns([
    pl.lit("xyz").alias("ID")       ############### TEST PURPUSE ################
    ]) 

    print("Done loading")

    df = _sequence_to_int(df)
    print("Done encoding")

    df = _labler(df)
    print("Done labeling")

    df = df.collect()

    df = make_dataset(df, save_path="./saved_dataset")

    print("Data preparation completed.")
