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
tf.config.set_visible_devices(                                       # Disable GPU, crashes on GPU
    [], "GPU"
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

###################################################################################################################################


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
        maxlen=148,
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


def splitter2(padded_label):
    """
    Splits the whole df into three sets: train, val and test set.
    Returns these three df
    """
    start_time = time.time()

    train_df, temp_df = train_test_split(
        padded_label,
        test_size=0.4,
        stratify=padded_label["Labels"],
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["Labels"], random_state=42
    )

    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    train_df.write_parquet("trainsetALL.parquet")
    val_df.write_parquet("valsetALL.parquet")
    test_df.write_parquet("testsetALL.parquet")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done splitting\nElapsed Time: {elapsed_time:.4f} seconds")

    return train_df, val_df, test_df




def make_dataset(df, batch_size=512, shuffle_buffer=1000, save_path=None):
    """
    Builds a tf.data.Dataset that:
      1. Reads your integer sequences directly from the Polars DF.
      2. One‐hots them in‐batch with dtype=tf.uint8 (1 byte per entry).
      3. Shuffles, batches & prefetches.
      4. (Optionally) saves to disk in TFRecord style.

    Returns a tf.data.Dataset ready to be used for training
    """
    # 1) Pull out the raw numpy arrays (no Python-list-of-tensors!)
    seqs   = tf.constant(df["Sequences"].to_list(), dtype=tf.int32)  # shape=(N,148)
    labels = tf.constant(df["Labels"].to_list(),    dtype=tf.int32)

    # 2) Build the base dataset
    ds = tf.data.Dataset.from_tensor_slices((seqs, labels))

    # 3) One‐hot & cast in the map step
    ds = ds.map(
        lambda seq, lab: (tf.one_hot(seq, depth=21, dtype=tf.uint8), lab),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 4) Shuffle, batch, prefetch
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # 5) Optionally save it as a TFRecord-style directory
    if save_path:
        tf.data.Dataset.save(ds, save_path)

    print(f"Dataset built: {ds.cardinality().numpy()} batches of up to {batch_size}")
    return ds



##################################################################################

if __name__ == "__main__":
    print("Starting data preparation...")
    df_path = "./DataTrainALL.csv"

    df = pl.read_csv(
        df_path,
        schema_overrides={
            "Sequences": pl.Utf8,
            "categories": pl.Int8,
        },
    ).lazy()
    print("Done loading")

    df = _sequence_to_int(df)
    print("Done encoding")

    df = _labler(df)
    print("Done labeling")

    df = df.collect()

    df = _padder(df)
    print("Done padding")

    # train_dataset, val_dataset, test_dataset = splitter2(padded_label)
    # print("Done splitting")

    # train_df_onehot = _one_hot(train_dataset)
    # print("Done one hot train set")
    # val_df_onehot = _one_hot(val_dataset)
    # print("Done one hot val set")
    # test_df_onehot = _one_hot(test_dataset)
    # print("Done one hot test set")

    # df_onehot = _one_hot(df)

    # train_df_ready = _creater(train_dataset, train_df_onehot, "trainset")
    # print("Done creating train set")
    # val_df_ready = _creater(val_dataset, val_df_onehot, "valset")
    # print("Done creating val set")
    # test_df_ready = _creater(test_dataset, test_df_onehot, "testset")
    # print("Done creating test set")

    # df = _creater(df, df_onehot, "AllSet")


    df = make_dataset(df, batch_size=512, save_path='./saved_dataset')

    print("Data preparation completed.")
