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
import numpy
gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(  # Disable GPU, crashes on GPU
    [], "GPU"
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from Dataset_preprocess_TRAIN_2d import dimension_positive             # change here to 2d if u want to create a ds for multi categorial classification

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
        pl.when(pl.col("categories") == 0)
        .then(1)
        .when(pl.col("categories") == 1)
        .then(2)
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

    # train_df.write_parquet("trainsetALL.parquet")
    # val_df.write_parquet("valsetALL.parquet")
    # test_df.write_parquet("testsetALL.parquet")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done splitting\nElapsed Time: {elapsed_time:.4f} seconds")

    return train_df, val_df, test_df

def one_hot(_df):
    """
    Creates one hot tensors for further pipelining it into the model
    Returns a new tensor with only the one hot encoded sequences
    """
    start_time = time.time()
    with tf.device("/CPU:0"):
        df_one_hot = [
            tf.one_hot(int_sequence, 21) for int_sequence in _df["Sequences"]
        ]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done one hot\nElapsed Time: {elapsed_time:.4f} seconds")
        # print(len(df_one_hot))
        # print(len(self.padded))
        return df_one_hot

def creater(df, df_onehot, name):
    """
    Creates a tf.Dataset with the one_hot tensor and the df column 'Labels'
    Returned is this tf.Dataset aswell as it is saved as a directory
    """
    start_time = time.time()

    with tf.device("/CPU:0"):
        tensor_df = tf.data.Dataset.from_tensor_slices((df_onehot, df["Labels"]))

        tensor_df.shuffle(buffer_size=1000, seed=4213122)

        tensor_df.save(name)

        print(f"dataset size: {len(tensor_df)}")

        print(f"Creation completed in {time.time() - start_time:.2f} seconds")

        return tensor_df



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
        seq = tf.pad(
            seq, [[0, dimension_positive - tf.shape(seq)[0]]], constant_values=21
        )
        oh = tf.one_hot(seq, depth=21, dtype=tf.uint8)

        # laboh = tf.one_hot(lab,depth=3,dtype=tf.uint8)
        return oh, lab, id_

    # apply to the ds
    ds = ds.map(pad_and_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle
    ds = ds.shuffle(shuffle_buffer)

    # Save
    if save_path:
        ds.save(save_path)
        print("Saved")

    inputs = []
    labels = []
    ids = []

    for x, y, z in ds:
        inputs.append(x.numpy().flatten())
        labels.append(y.numpy().tolist())
        ids.append(z.numpy())

    # Annahme: Inputs sind flattenbar
    df = pl.DataFrame({
        'input': inputs,
        'label': labels,
        'id': ids
    })

    df.write_csv('./saved_dataset2dNew.csv')
    
    print("Dataset built")
    return ds


##################################################################################

if __name__ == "__main__":

    dimension=1

    print("Starting data preparation...")
    df_path = "./Dataframes/DataTrainPF00120.csv"

    # load in
    df = pl.read_csv(
        df_path,
        schema_overrides={
            "Sequences": pl.Utf8,
            "categories": pl.Int8,
            "ID": pl.Utf8,
        },
    ).lazy()

    print("Done loading")

    df = _sequence_to_int(df)
    print("Done encoding")

    if dimension == 1:
        df = df.collect()
        df= _padder(df)
        df = _labler(df)
        print("Done labeling")

        train_df, val_df, test_df=splitter2(df) # Split the dataset into train/val/test sets old approach used for the binary classification
        train_df_oh=one_hot(_df=train_df)
        val_df_oh=one_hot(_df=val_df)
        test_df_oh=one_hot(_df=test_df)

        creater(train_df, train_df_oh, name="./Datasets/PF00210/trainset")
        creater(val_df, val_df_oh, name="./Datasets/PF00210/valset")
        creater(test_df, test_df_oh, name="./Datasets/PF00210/testset")
        
        # this creates the ready to go datasets, so you can skip DataAllCreater_part2.py and use it directly for training
    else:
        df = _labler2d(df)
        df= df.collect()
        df = make_dataset(df, save_path="./Datasets/2dNew")





    print("Data preparation completed.")
