###################################################################################################################################

# File after Dataset_preprocessing.py
# This file is used to create a DNN model using the preprocessed dataset

###################################################################################################################################
# imports

import datetime
import os
import time

import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python import keras

###################################################################################################################################


class Starter:
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path, index_col=False)  # Open CSV file
        self.df_int = self._sequence_to_int()
        self.target_dimension = len(self.df["Sequences"].max())
        self.padded = self._padder()
        self.df_one_hot = self._one_hot()
        self.padded_label = self._labler()
        self.train_dataset, self.val_dataset, self.test_dataset = self._splitter()
        self.model = self._modeler()

    def _sequence_to_int(self):
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
            "X": 21,  # Unknown or special character                (21 for all other AA)
            "Z": 21,  # Glutamine (Q) or Glutamic acid (E)
            "B": 21,  # Asparagine (N) or Aspartic acid (D)
            "U": 21,  # Selenocysteine
        }

        def encode_sequence(seq):
            return [amino_acid_to_int[amino_acid] for amino_acid in seq]

        self.df["Sequences"] = self.df["Sequences"].apply(encode_sequence)

        # print(self.df)
        # print(type(self.df))
        # print(self.df['Sequences'][0])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done encoding\nElapsed Time: {elapsed_time:.4f} seconds")
        return self.df

    def _padder(self):
        start_time = time.time()
        sequences = self.df_int["Sequences"].tolist()
        # print(type(sequences))
        # print(sequences[0:3])
        # print(self.target_dimension)
        padded = pad_sequences(
            sequences,
            maxlen=self.target_dimension,
            padding="post",
            truncating="post",
            value=21,
        )
        # print(padded)
        self.df_int["Sequences"] = list(padded)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Done padding\nElapsed Time: {elapsed_time:.4f} seconds")
        # print(self.df_int)

        return self.df_int

    def _one_hot(self):
        start_time = time.time()
        df_one_hot = [
            tf.one_hot(int_sequence, 21) for int_sequence in self.padded["Sequences"]
        ]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done one hot\nElapsed Time: {elapsed_time:.4f} seconds")
        # print(len(df_one_hot))
        # print(len(self.padded))
        return df_one_hot

    def _labler(self):
        start_time = time.time()
        self.padded["Labels"] = self.padded["categories"].apply(
            lambda x: 0 if x == 0 else 1
        )
        self.padded_label = self.padded
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done labeling\nElapsed Time: {elapsed_time:.4f} seconds")
        return self.padded_label

    def _splitter(self):
        start_time = time.time()
        tensor_df = tf.data.Dataset.from_tensor_slices(
            (self.df_one_hot, self.padded_label["Labels"])
        )
        tensor_df = tensor_df.shuffle(
            buffer_size=218468, reshuffle_each_iteration=False
        )

        ### splitting ###

        # split sizes
        dataset_size = len(self.padded)
        train_size = int(0.70 * dataset_size)  # 70%
        val_size = int(0.15 * dataset_size)  # 15%
        test_size = dataset_size - train_size - val_size  # 15%

        # split the sets
        train_dataset = tensor_df.take(train_size)
        val_dataset = tensor_df.skip(train_size).take(val_size)
        test_dataset = tensor_df.skip(train_size + val_size)

        # Batch & prefetch
        train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done splitting\nElapsed Time: {elapsed_time:.4f} seconds")
        return train_dataset, val_dataset, test_dataset

    def _modeler(self):
        start_time = time.time()
        model = Sequential(
            [
                Flatten(input_shape=(self.target_dimension, 21)),
                Dense(21, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                Dropout(0.3),
                Dense(21, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                Dropout(0.3),
                Dense(
                    1,
                    activation="sigmoid",
                    kernel_regularizer=regularizers.l2(0.001),
                ),
            ]
        )

        print(model.summary())

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done modelling\nElapsed Time: {elapsed_time:.4f} seconds")

        return model

    def trainer(self):
        start_time = time.time()
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
        log_dir = os.path.join("logs", time.strftime("run_%Y_%m_%d-%H_%M_%S"))

        tensorboard_cb = TensorBoard(log_dir=log_dir)

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            f"run_{timestamp}.keras", save_best_only=True
        )

        early_stopping_cb = keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_loss"
        )
        history = self.model.fit(
            self.train_dataset,
            epochs=500,
            validation_data=self.val_dataset,
            callbacks=[tensorboard_cb, early_stopping_cb, checkpoint_cb, reduce_lr],
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done training\nElapsed Time: {elapsed_time:.4f} seconds")


###################################################################################################################################

if __name__ == "__main__":
    run = Starter("./datatest1.csv")
    # run.trainer()

    print(run.padded_label)

    # lists = run.padded_label['Labels'].tolist()
    # for entry in lists:
    #     if entry == 0:
    #         print('found')
    #     else:
    #         # print('NOT found')
    #         continue
