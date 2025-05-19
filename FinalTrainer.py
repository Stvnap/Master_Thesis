###################################################################################################################################
"""
File after DNN_pipeline.py
This file is used to train the final model found via the HP search

INFOS:
the trial.json and checkpoint.weights.h5 are both needed to initialize the model sturcture correctly.

"""
###################################################################################################################################

import datetime
import json
import os
import time

import keras
import numpy as np
import polars as pl
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python import keras

###################################################################################################################################

STRATEGY = tf.distribute.MirroredStrategy()
BATCH_SIZE = 256

print(tf.keras.__version__)
print(tf.__version__)
print(pl.__version__)


class Testrunning:
    """
    Class to prepare the model and start the model training
    """

    def __init__(
        self,
        modelpath,
        weightpath,
        dimension,
        df_path="./DataTrainSwissProt.csv",  # the Dataset_path used to train the model
        batch_size=BATCH_SIZE,
        strategy=STRATEGY,
    ):
        self.strategy = strategy
        self.batch_size = batch_size
        self.dimension=dimension

        print("opening Train data")
        self.df = pl.read_csv(  # Open CSV file
            df_path,
            dtypes={
                "Sequences": pl.Utf8,
                "categories": pl.Int8,
            },
        )

        self.df_int = self._sequence_to_int()
        self.target_dimension = self.df.select(
            max=pl.col("Sequences").list.len().max()
        ).item()

        self.padded = self._padder()
        self.padded_label = self._labler()

        self.train_dataset, self.val_dataset, self.test_dataset = self.splitter2()

        self.train_df_onehot = self._one_hot(self.train_dataset)
        self.val_df_onehot = self._one_hot(self.val_dataset)
        self.test_df_onehot = self._one_hot(self.test_dataset)

        # self.train_df_ready = self._creater(
        #     self.train_dataset, self.train_df_onehot, "trainset"
        # )

        # self.val_df_ready = self._creater(
        #     self.val_dataset, self.val_df_onehot, "valset"
        # )

        # self.test_df_ready = self._creater(
        #     self.test_dataset, self.test_df_onehot, "testset"
        # )

        self.train_dataset, self.val_dataset, self.test_dataset = self._loader()

        self.weightpath = weightpath
        self.json_path = modelpath

        self.model_values = self.fromjson(self.json_path)

        self.model = self.buildfromjson()

    def fromjson(self, json_path):
        """
        Retrives the model structure from the trial.json file
        Returns the infos in json_data
        """
        with open(json_path, "r") as f:
            json_data = json.load(f)
        # print(json_data)
        json_data = json_data["hyperparameters"]["values"]
        # print(json_data)

        return json_data

    def buildfromjson(self):
        """
        The model is rebuild via the data from the trial.json
        Additionally the already learned weights are loaded and applied to the model
        Returned is the ready to be trained model
        """

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

        if self.dimension==1:
            model.add(Dense(1, activation="sigmoid"))
        else:
            model.add(Dense(3, activation="softmax"))



        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy", "precision", "recall", "AUC"],
        )

        model.load_weights(self.weightpath)

        model.summary()
        return model

    def _sequence_to_int(self):
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
            "X": 21,  # Unknown or special character                (21 for all other AA)
            "Z": 21,  # Glutamine (Q) or Glutamic acid (E)
            "B": 21,  # Asparagine (N) or Aspartic acid (D)
            "U": 21,  # Selenocysteine
            "O": 21,  # Pyrrolysin
        }

        self.df = self.df.drop_nulls(subset=["Sequences"])

        def encode_sequence(seq):
            return [amino_acid_to_int[amino_acid] for amino_acid in seq]

        print(type(self.df))

        self.df = self.df.with_columns(
            pl.col("Sequences")
            .map_elements(encode_sequence, return_dtype=pl.List(pl.Int16))
            .alias("Sequences")
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done encoding\nElapsed Time: {elapsed_time:.4f} seconds")
        return self.df

    def _padder(self):
        """
        Pads the sequences to the target dimension with value 21 = unidentified aa
        Returns the padded df
        """
        start_time = time.time()
        sequences = self.df_int["Sequences"].to_list()

        padded = pad_sequences(
            sequences,
            maxlen=self.target_dimension,
            padding="post",
            truncating="post",
            value=21,
        )
        # print(padded)
        self.df_int = self.df_int.with_columns(pl.lit(padded).alias("Sequences"))
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Done padding\nElapsed Time: {elapsed_time:.4f} seconds")
        print(self.df_int)

        return self.df_int

    def _labler(self):
        """
        Creates a new column 'Labels' that translates the categories column to 1 = target domain, 0 = all other
        Returns the df with added 'Labels' column
        """
        start_time = time.time()
        self.padded = self.padded.with_columns(
            pl.when(pl.col("categories") == 0).then(1).otherwise(0).alias("Labels")
        )
        self.padded_label = self.padded
        self.padded_label = self.padded_label.drop("categories")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done labeling\nElapsed Time: {elapsed_time:.4f} seconds")
        return self.padded_label

    def splitter2(self):
        """
        Splits the whole df into three sets: train, val and test set.
        Returns these three df
        """
        start_time = time.time()

        train_df, temp_df = train_test_split(
            self.padded_label,
            test_size=0.4,
            stratify=self.padded_label["Labels"],
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

    def _one_hot(self, _df):
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

    def _creater(self, df, df_onehot, name):
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

    def _loader(self):
        """
        Loads the data from the directory to use as the mdoel input, to cut time
        Used for all further hp seaches when the sets are created
        """
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

    def trainer(self, epochnumber, modelname):
        """
        Function to start the training of the previously created model.
        Tensorboard, early stopping on val-loss and save checkpoints are used
        Finally the model is saved with the name inputted in modelname
        """
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

        print("Starting training")

        self.history = self.model.fit(
            self.train_dataset,
            epochs=epochnumber,
            validation_data=self.val_dataset,
            callbacks=[tensorboard_cb, early_stopping_cb, checkpoint_cb],
            # class_weight=self.class_weight_dict,
        )
        self.model.save(f"models/{modelname}.keras")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done training\nElapsed Time: {elapsed_time:.4f} seconds")


##################################################################################################################################################################
if __name__ == "__main__":
    Testrun = Testrunning(
        "/global/research/students/sapelt/Masters/MasterThesis/logshp/test1/trial_00/trial.json",
        "/global/research/students/sapelt/Masters/MasterThesis/logshp/test1/trial_00/checkpoint.weights.h5",
        dimension=2
    )

    Testrun.trainer(500, "model1605")
