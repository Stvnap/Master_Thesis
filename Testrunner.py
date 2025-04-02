import datetime
import os
import time

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python import keras
import gc


BATCH_SIZE = 64

print(tf.keras.__version__)
print(tf.__version__)


class Predictor:
    def __init__(self,modelpath,df_path='./datatestSwissProt.csv', batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        self.df = pd.read_csv(df_path, index_col=False)  # Open CSV file
        self.df_int = self._sequence_to_int()
        self.target_dimension = len(self.df["Sequences"].max())
        self.padded = self._padder()
        self.padded_label = self._labler()
        self.train_dataset, self.val_dataset, self.test_dataset = self._loader()
        self.model=tf.keras.models.load_model(modelpath)        # self.model.summary()


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
            "O": 21,  # Pyrrolysin
        }
        self.df = self.df.dropna(subset=["Sequences"])
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
    

    def _loader(self):
        self.class_weights = compute_class_weight(
            "balanced",
            classes=np.unique(self.padded_label["Labels"]),
            y=self.padded_label["Labels"],
        )
        self.class_weight_dict = dict(enumerate(self.class_weights))

        train_dataset = tf.data.Dataset.load("trainset")
        val_dataset = tf.data.Dataset.load("valset")
        test_dataset = tf.data.Dataset.load("testset")

        train_dataset = train_dataset.batch(self.batch_size)
        val_dataset = val_dataset.batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)
        
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset

    def trainer(self):
        start_time = time.time()
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
        log_dir = os.path.join("logs", time.strftime("run_%Y_%m_%d-%H_%M_%S"))

        tensorboard_cb = TensorBoard(log_dir=log_dir)

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            f"models/run_{timestamp}.keras", save_best_only=True
        )

        early_stopping_cb = keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_loss"
        )
        self.history = self.model.fit(
            self.train_dataset,
            epochs=500,
            validation_data=self.val_dataset,
            callbacks=[tensorboard_cb, early_stopping_cb, checkpoint_cb, reduce_lr],
            class_weight=self.class_weight_dict,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done training\nElapsed Time: {elapsed_time:.4f} seconds")



    def predict(self):
        start_time = time.time()
        predictions = self.history.predict(self.train_dataset, batch_size=self.batch_size)
        predictions = np.argmax(predictions, axis=1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done predicting\nElapsed Time: {elapsed_time:.4f} seconds")
        return predictions

##################################################################################################################################################################
if __name__ == "__main__":

    prediction=Predictor("/global/research/students/sapelt/Masters/MasterThesis/bestmodels/2025_04_02-10_42/best_model.keras")

    prediction.trainer()





