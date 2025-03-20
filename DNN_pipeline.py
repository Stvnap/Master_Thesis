###################################################################################################################################

# File after Dataset_preprocessing.py
# This file is used to create a DNN model using the preprocessed dataset

# Tasks:
# Check datasplitting, reason for high overfitting/ accuracy in train & val set

###################################################################################################################################
# imports

import datetime
import os
import time

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python import keras

###################################################################################################################################
STRATEGY = tf.distribute.MirroredStrategy()


class HyperModel(kt.HyperModel):
    def __init__(self, target_dimension, strategy=STRATEGY):
        self.target_dimension = target_dimension
        self.strategy = strategy

    def build(self, hp):
        with self.strategy.scope():
            model = Sequential()
            model.add(Flatten(input_shape=(self.target_dimension, 21)))

            model.add(
                Dense(
                    units=hp.Int("units", min_value=32, max_value=512, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )

            if hp.Boolean("dropout"):
                model.add(Dropout(rate=hp.Float("dropout_rate", 0.1, 0.5, step=0.1)))

            model.add(
                Dense(
                    1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.001)
                )
            )

            learning_rate = hp.Float(
                "lr", min_value=1e-4, max_value=1e-2, sampling="log"
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy", "precision", "recall", "AUC"],
            )

            return model


class Starter:
    def __init__(self, df_path, strategy=STRATEGY):
        self.strategy = strategy
        self.df = pd.read_csv(df_path, index_col=False)  # Open CSV file
        self.df_int = self._sequence_to_int()
        self.target_dimension = len(self.df["Sequences"].max())
        self.padded = self._padder()
        self.df_one_hot = self._one_hot()
        self.padded_label = self._labler()
        # self.train_dataset, self.val_dataset, self.test_dataset = self._splitter()
        self.train_dataset, self.val_dataset, self.test_dataset = self._loader()

        # self.model = self._modeler(kt.HyperParameters())

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

    def _one_hot(self):
        start_time = time.time()
        with tf.device("/CPU:0"):
            df_one_hot = [
                tf.one_hot(int_sequence, 21)
                for int_sequence in self.padded["Sequences"]
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

        self.class_weights = compute_class_weight(
            "balanced",
            classes=np.unique(self.padded_label["Labels"]),
            y=self.padded_label["Labels"],
        )
        self.class_weight_dict = dict(enumerate(self.class_weights))

        tensor_df = tf.data.Dataset.from_tensor_slices(
            (self.df_one_hot, self.padded_label["Labels"])
        )

        features, labels = [], []
        for feature, label in tensor_df:
            features.append(feature.numpy())
            labels.append(label.numpy())

        features = np.array(features)
        labels = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, stratify=labels, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        train_dataset = train_dataset.shuffle(buffer_size=218468).batch(32)
        val_dataset = val_dataset.batch(32)
        test_dataset = test_dataset.batch(32)

        train_dataset.save("trainset")
        val_dataset.save("valset")
        test_dataset.save("testset")

        print(f"Train dataset size: {len(X_train)}")
        print(f"Validation dataset size: {len(X_val)}")
        print(f"Test dataset size: {len(X_test)}")

        print(f"Stratified split completed in {time.time() - start_time:.2f} seconds")

        return train_dataset, val_dataset, test_dataset

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

        return train_dataset, val_dataset, test_dataset

        # def _modeler(self, hp):
        start_time = time.time()

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            model = Sequential()

            # Flatten layer
            model.add(Flatten(input_shape=(self.target_dimension, 21)))

            # Dense layer with hyperparameters
            model.add(
                Dense(
                    units=hp.Int("units", min_value=5, max_value=505, step=25),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )

            # Conditionally add the Dropout layer
            if hp.Boolean("dropout"):
                model.add(Dropout(rate=0.25))

            # Output layer
            model.add(
                Dense(
                    1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.001)
                )
            )

            print(model.summary())

            learning_rate = hp.Float(
                "lr", min_value=1e-4, max_value=1e-2, sampling="log"
            )

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy", "precision", "recall", "AUC", "f1_score"],
            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done modelling\nElapsed Time: {elapsed_time:.4f} seconds")

        return model

    def trainer(self):
        start_time = time.time()
        with self.strategy.scope():
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
                class_weight=self.class_weight_dict,
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done training\nElapsed Time: {elapsed_time:.4f} seconds")

    def tuner(self):
        hypermodel = HyperModel(target_dimension=self.target_dimension)

        tuner = kt.RandomSearch(
            hypermodel=hypermodel,
            objective="val_AUC",
            max_trials=3,
            executions_per_trial=2,
            overwrite=True,
            directory="./logshp",
            project_name="test1",
        )

        tuner.search_space_summary()
        tuner.search(self.train_dataset, epochs=2, validation_data=self.val_dataset)


###################################################################################################################################

if __name__ == "__main__":
    print(tf.config.list_physical_devices("GPU"), "\n", "\n", "\n", "\n")
    with STRATEGY.scope():
        run = Starter(
            "/global/research/students/sapelt/Masters/MasterThesis/datatestSwissProt.csv"
        )

        # run = Starter("/global/research/students/sapelt/Masters/MasterThesis/datatest1.csv")

        run.tuner()
