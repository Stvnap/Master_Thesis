###################################################################################################################################
"""
File after Dataset_preprocessing.py
This file is used to create a DNN model using the preprocessed dataset

INFOS:
HARD CODED CPU:1 USE FOR TESTING PURPOSES
Switch to STRATEGY = tf.distribute.MirroredStrategy() and change the with self.strategy: to with self.strategy.scope(): for GPU usage

"""
###################################################################################################################################

import datetime
import os
import time

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

###################################################################################################################################


STRATEGY = tf.distribute.MirroredStrategy()
print(f"Number of devices: {STRATEGY.num_replicas_in_sync}")


BATCH_SIZE = 128 * STRATEGY.num_replicas_in_sync
print(BATCH_SIZE)


print(tf.keras.__version__)
print(tf.__version__)


class MyHyperModel(kt.HyperModel):
    """
    Hypermodel for model structure and HP dimension.
    """

    def __init__(self, target_dimension):
        self.target_dimension = target_dimension

    def build(self, hp):
        """
        actual build of the model with all HP variables
        """

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        n_neurons = hp.Int("n_neurons", min_value=3100, max_value=3400)
        n_hidden = hp.Int("n_hidden", min_value=4, max_value=24, default=8)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        optimizer = hp.Choice("optimizer", values=["adam", "sgd"])
        activation = hp.Choice("activation", values=["leaky_relu", "sigmoid", "elu"])
        dropout_rate = hp.Float("drop_rate", min_value=0.1, max_value=0.5, step=0.1)

        gammaloss = hp.Float("gamma_loss", min_value=0.5, max_value=5.0, step=0.5)
        alphaloss = hp.Choice("alpha_loss", values=[0.1, 0.25, 0.5, 0.75, 0.9])

        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07
            )
        else:
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                nesterov=True,  # , decay=1e-4 for power scheduling
            )

        if activation == "leaky_relu":
            activation = tf.keras.layers.LeakyReLU(negative_slope=0.2)
            kernel_init = "he_normal"
        elif activation == "sigmoid":
            activation = "sigmoid"
            kernel_init = "glorot_normal"
        elif activation == "elu":
            activation = "elu"
            kernel_init = "he_normal"
        else:
            pass

        model = Sequential()
        model.add(Input(shape=(self.target_dimension, 21)))
        model.add(Flatten())

        for current_layer in range(n_hidden):
            if current_layer <= 5:
                model.add(tf.keras.layers.Dropout(rate=dropout_rate))

            model.add(
                Dense(
                    n_neurons,
                    activation=activation,
                    kernel_initializer=kernel_init,
                    kernel_constraint=tf.keras.constraints.max_norm(1.0),
                    kernel_regularizer=tf.keras.regularizers.l2(
                        hp.Float("l2_reg", min_value=1e-6, max_value=1e-2, step=1e-6)
                    ),
                )
            )

        model.add(Dense(1, activation="sigmoid"))

        loss_name = hp.Choice(
            "loss", values=["binary_crossentropy", "binary_focal_crossentropy"]
        )

        if loss_name == "binary_crossentropy":
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        elif loss_name == "binary_focal_crossentropy":
            loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                gamma=gammaloss, alpha=alphaloss, apply_class_balancing=False
            )

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy", "precision", "recall", "AUC"],
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        """
        Fits the created model
        """
        model = model.fit(*args, **kwargs)
        return model


class Starter:
    """
    Class for preparing starting the HP search
    """

    def __init__(self, df_path, strategy=STRATEGY, batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        self.strategy = strategy
        self.df = pd.read_csv(df_path, index_col=False)  # Open CSV file
        self.df_int = self._sequence_to_int()
        self.target_dimension = len(self.df_int["Sequences"].max())
        self.padded = self._padder()
        self.labeled_df = self._labler()
        self.train_df, self.val_df, self.test_df = self.splitter2()
        self.train_df_onehot = self._one_hot(self.train_df)
        self.val_df_onehot = self._one_hot(self.val_df)
        self.test_df_onehot = self._one_hot(self.test_df)

        self.train_df_ready = self._creater(
            self.train_df, self.train_df_onehot, "trainset"
        )

        self.val_df_ready = self._creater(self.val_df, self.val_df_onehot, "valset")

        self.test_df_ready = self._creater(self.test_df, self.test_df_onehot, "testset")

        self.train_dataset, self.val_dataset, self.test_dataset = self._loader()

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
        self.df = self.df.dropna(subset=["Sequences"])  # drop empty entries

        def encode_sequence(seq):
            return [amino_acid_to_int[amino_acid] for amino_acid in seq]

        self.df["Sequences"] = self.df["Sequences"].apply(encode_sequence)

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
        sequences = self.df_int["Sequences"].tolist()

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
        print(self.df_int)

        return self.df_int

    def _labler(self):
        """
        Creates a new column 'Labels' that translates the categories column to 1 = target domain, 0 = all other
        Returns the df with added 'Labels' column
        """
        start_time = time.time()
        self.padded["Labels"] = self.padded["categories"].apply(
            lambda x: 1 if x == 0 else 0
        )
        self.labeled_df = self.padded

        self.class_weights = compute_class_weight(
            "balanced",
            classes=np.unique(self.labeled_df["Labels"]),
            y=self.labeled_df["Labels"],
        )
        self.class_weight_dict = dict(enumerate(self.class_weights))

        print("Class weights", self.class_weight_dict)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done labeling\nElapsed Time: {elapsed_time:.4f} seconds")
        return self.labeled_df

    def splitter2(self):
        """
        splits the whole df into three sets: train, val and test set.
        Returns these three df
        """
        start_time = time.time()

        train_df, temp_df = train_test_split(
            self.labeled_df,
            test_size=0.4,
            stratify=self.labeled_df["Labels"],
            random_state=42,
        )

        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["Labels"], random_state=42
        )

        print(f"Train set shape: {train_df.shape}")
        print(f"Validation set shape: {val_df.shape}")
        print(f"Test set shape: {test_df.shape}")

        train_df.to_csv("trainset.csv", index=False)
        val_df.to_csv("valset.csv", index=False)
        test_df.to_csv("testset.csv", index=False)

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
        start_time = time.time()
        train_dataset = tf.data.Dataset.load("trainset")
        val_dataset = tf.data.Dataset.load("valset")
        test_dataset = tf.data.Dataset.load("testset")

        train_dataset = train_dataset.batch(self.batch_size)
        val_dataset = val_dataset.batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)

        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        print(f"Loading completed in {time.time() - start_time:.2f} seconds")
        return train_dataset, val_dataset, test_dataset

    def tuner(self):
        """
        Iniziation function for the HP search, BayesianOptimization search is used with tensorbaord callbacks,
        as well as model saves for the best model so far. Early stopping is used with patience on 5 and val_loss
        as monitor.
        Finally the best 3 models are saved after one HP search run
        Can be used for multiple HP searches, overwrite=False
        """
        start_time = time.time()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directories outside strategy scope
        log_dir = "./logshp/test1"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs("logshp/weights", exist_ok=True)

        # Setup callbacks
        tensorboard = TensorBoard(log_dir=log_dir)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            f"logshp/weights/run_{timestamp}.weights.h5",
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        # Initialize tuner inside strategy scope
        # with self.strategy.scope():
        self.tuner = kt.BayesianOptimization(
            hypermodel=MyHyperModel(target_dimension=self.target_dimension),
            objective="val_loss",
            max_trials=1,
            overwrite=False,
            directory="./logshp",
            # distribution_strategy=tf.distribute.MirroredStrategy(),
            project_name="test2",
        )

        self.tuner.search_space_summary()

        # Print batch shape for debugging
        for batch in self.train_dataset.take(1):
            print(f"Dataset batch shape: {batch[0].shape}")

        # Search outside strategy scope
        self.tuner.search(
            self.train_dataset,
            epochs=1,
            validation_data=self.val_dataset,
            callbacks=[tensorboard, checkpoint_cb, early_stopping],
            class_weight=self.class_weight_dict,  # removed when using focal loss
        )

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
        os.makedirs(f"./bestmodels/{timestamp}", exist_ok=True)
        models = run.tuner.get_best_models(num_models=3)
        best_model = models[0]
        best_model.summary()
        best_model.save(f"./bestmodels/{timestamp}/best_model.keras")

        second_model = models[1]
        second_model.save(f"./bestmodels/{timestamp}/second_best_model.keras")

        third_model = models[2]
        third_model.save(f"./bestmodels/{timestamp}/third_best_model.keras")

        elapsed_time = time.time() - start_time
        print(f"Done tuning\nElapsed Time: {elapsed_time:.4f} seconds")


###################################################################################################################################

if __name__ == "__main__":
    print(tf.config.list_physical_devices("GPU"), "\n", "\n", "\n", "\n")

    run = Starter("./DataTrainALL.csv")

    run.tuner()
