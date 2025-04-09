###################################################################################################################################
"""'
File after Dataset_preprocessing.pya
This file is used to create a DNN model using the preprocessed dataset

INFOS:
HARDt CODED CPU:1 USE FOR TESTING PURPOSES
Switch to STRATEGY = tf.distribute.MirroredStrategy() and change the with self.strategy: to with self.strategy.scope(): for GPU usage

QUESTIONS:
Still uses double batching (batch two times during data creation and during tuning)
due to the mismatch inc input dimension (dimension expected: [none,target_dimension,21]; without the batching during data creation its [21,target_dimension])
ish there a fix for this?

"""
###################################################################################################################################
# imports

import datetime
import gc
import os
import time
import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Flatten, Input
from keras.losses import BinaryCrossentropy
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
    def __init__(self, target_dimension):
        self.target_dimension = target_dimension

    def build(self, hp):
        # with self.strategy.scope():

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        n_neurons = hp.Int("n_neurons", min_value=3100, max_value=3400)
        n_hidden = hp.Int("n_hidden", min_value=4, max_value=24, default=8)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        optimizer = hp.Choice("optimizer", values=["adam", "sgd"])
        activation = hp.Choice("activation", values=["leaky_relu", "sigmoid", "elu"])
        dropout_rate = hp.Float("drop_rate", min_value=0.1, max_value=0.5, step=0.1)

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

        def loss2(y_true, y_pred):
            bce_loss = BinaryCrossentropy()
            err = bce_loss(y_true, y_pred)
            loss = tf.cond(
                tf.size(y_pred) == 0, lambda: 0.0, lambda: tf.math.reduce_mean(err)
            )
            return loss

        model.compile(
            optimizer=optimizer,
            loss=loss2,
            metrics=["accuracy", "precision", "recall", "AUC"],
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        model = model.fit(*args, **kwargs)
        return model


class Starter:
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

        # self.train_df_ready = self._creater(
        #     self.train_df, self.train_df_onehot,'trainset')

        # self.val_df_ready = self._creater(
        #     self.val_df, self.val_df_onehot,'valset')

        # self.test_df_ready = self._creater(
        #     self.test_df, self.test_df_onehot,'testset')


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
        print(self.df_int)

        return self.df_int


    ############### needs to split dataset here to gather the IDs used in the test set #################
    #### then go on to one hot encoding

    def _labler(self):
        start_time = time.time()
        self.padded["Labels"] = self.padded["categories"].apply(
            lambda x: 0 if x == 0 else 1
        )
        self.labeled_df = self.padded


        self.class_weights = compute_class_weight(
            "balanced",
            classes=np.unique(self.labeled_df["Labels"]),
            y=self.labeled_df["Labels"],
        )
        self.class_weight_dict = dict(enumerate(self.class_weights))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done labeling\nElapsed Time: {elapsed_time:.4f} seconds")
        return self.labeled_df

    def splitter2(self):
        start_time = time.time()

        train_df, temp_df = train_test_split(self.labeled_df, test_size=0.4, stratify=self.labeled_df['Labels'], random_state=42)

        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Labels'], random_state=42)

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


    def _one_hot(self,_df):
        start_time = time.time()
        with tf.device("/CPU:0"):
            df_one_hot = [
                tf.one_hot(int_sequence, 21)
                for int_sequence in _df["Sequences"]
            ]
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Done one hot\nElapsed Time: {elapsed_time:.4f} seconds")
            # print(len(df_one_hot))
            # print(len(self.padded))
            return df_one_hot





    def _creater(self, df,df_onehot,name):
        start_time = time.time()

        with tf.device("/CPU:0"):


            tensor_df = tf.data.Dataset.from_tensor_slices(
                (df_onehot, df["Labels"])
            )


            tensor_df.shuffle(buffer_size=1000, seed=4213122)


            tensor_df.save(name)


            print(f"dataset size: {len(tensor_df)}")

            print(
                f"Creation completed in {time.time() - start_time:.2f} seconds"
            )

            return tensor_df

    def _loader(self):
        start_time = time.time()
        train_dataset = tf.data.Dataset.load("trainset")
        val_dataset = tf.data.Dataset.load("valset")
        test_dataset = tf.data.Dataset.load("testset")

        # print(f"Train dataset size: {len(train_dataset)}")

        # train_dataset = train_dataset.take(len(train_dataset) - 152)

        # if len(train_dataset) % 4 % 128 == 0:
        #     print("true now")
        #     print("division check:", len(train_dataset) % 4 % 128)

        # print(f"Train dataset size now: {len(train_dataset)}")

        train_dataset = train_dataset.batch(self.batch_size)
        val_dataset = val_dataset.batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)

        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        print(
            f"Loading completed in {time.time() - start_time:.2f} seconds"
        )
        return train_dataset, val_dataset, test_dataset

    def tuner(self):
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
            hypermodel=MyHyperModel(
                target_dimension=self.target_dimension
            ),
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
            class_weight=self.class_weight_dict,
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

    run = Starter("./datatestSwissProt.csv")

    # run = Starter("/global/research/students/sapelt/Masters/MasterThesis/datatest1.csv")

    run.tuner()
