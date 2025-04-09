import datetime
import json
import os
import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.saving import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python import keras

STRATEGY = tf.distribute.MirroredStrategy()
BATCH_SIZE = 256

print(tf.keras.__version__)
print(tf.__version__)


# with STRATEGY.scope():


class Predictor:
    def __init__(
        self,
        modelpath,
        weightpath,
        df_path="./datatestSwissProt.csv",
        batch_size=BATCH_SIZE,
        strategy=STRATEGY,
    ):
        self.strategy = strategy
        self.batch_size = batch_size
        self.df = pd.read_csv(df_path, index_col=False)  # Open CSV file
        self.df_int = self._sequence_to_int()
        self.target_dimension = len(self.df["Sequences"].max())
        self.padded = self._padder()
        self.padded_label = self._labler()
        self.train_dataset, self.val_dataset, self.test_dataset = self._loader()
        # self.weightpath = weightpath
        # self.json_path=modelpath
        # # self.model=tf.keras.models.load_model(modelpath)        # self.model.summary()
        # self.model_values= self.fromjson(self.json_path)
        # self.model= self.buildfromjson()

    def fromjson(self, json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
        # print(json_data)
        json_data = json_data["hyperparameters"]["values"]
        # print(json_data)

        return json_data

    def buildfromjson(self):
        if self.model_values["activation"] == "leaky_relu":
            activation = tf.keras.layers.LeakyReLU(negative_slope=0.2)
            kernel_init = "he_normal"
        elif self.model_values["activation"] == "sigmoid":
            activation = "sigmoid"
            kernel_init = "glorot_normal"
        elif self.model_values["activation"] == "elu":
            activation = "elu"
            kernel_init = "he_normal"
        else:
            pass

        if self.model_values["optimizer"] == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.model_values["lr"],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
            )
        else:
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.model_values["lr"],
                nesterov=True,  # , decay=1e-4 for power scheduling
            )

        model = Sequential()
        model.add(Input(shape=(self.target_dimension, 21)))
        model.add(Flatten())

        for i in range(self.model_values["n_hidden"]):
            if i <= 5:
                model.add(tf.keras.layers.Dropout(rate=self.model_values["drop_rate"]))

            model.add(
                Dense(
                    self.model_values["n_neurons"],
                    activation=activation,
                    kernel_initializer=kernel_init,
                    kernel_constraint=tf.keras.constraints.max_norm(1.0),
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.model_values["l2_reg"]
                    ),
                ),
            )

        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall", "AUC"],
        )

        model.load_weights(self.weightpath)

        model.summary()
        return model

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
        print(self.class_weight_dict)

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

        log_dir = os.path.join("logs", time.strftime("run_%Y_%m_%d-%H_%M"))

        tensorboard_cb = TensorBoard(log_dir=log_dir)

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            f"models/run_{timestamp}.keras", save_best_only=True
        )

        early_stopping_cb = keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_loss"
        )

        self.history = self.model.fit(
            self.train_dataset,
            epochs=1,
            validation_data=self.val_dataset,
            callbacks=[tensorboard_cb, early_stopping_cb, checkpoint_cb],
            class_weight=self.class_weight_dict,
        )
        self.model.save("models/my_model")
        self.model.save("models/my_model.keras")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done training\nElapsed Time: {elapsed_time:.4f} seconds")

    


##################################################################################################################################################################
if __name__ == "__main__":
    # prediction=Predictor("/global/research/students/sapelt/Masters/MasterThesis/bestmodels/2025_04_02-10_42/best_model.keras")

    prediction = Predictor(
        "/global/research/students/sapelt/Masters/MasterThesis/logshp/run_20250406_145252/trial_01/trial.json",
        "/global/research/students/sapelt/Masters/MasterThesis/logshp/run_20250406_145252/trial_01/checkpoint.weights.h5",
    )

    # prediction.trainer()

