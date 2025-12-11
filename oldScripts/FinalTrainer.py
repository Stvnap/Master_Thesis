"""
FinalTrainer.py

File after DNN_pipeline.py
This file is used to train the final model found via the HP search
INFOS:
the trial.json and checkpoint.weights.h5 are both needed to initialize the model sturcture correctly.

Table of Contents:
=========================
Testrunning class
    - __init__
    - fromjson
    - buildfromjson
    - _loader
    - trainer
"""

# -------------------------
# Imports & Globals
# -------------------------

import datetime
import json
import os
import time

import keras
import polars as pl
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential
from tensorflow.python import keras

# from DataAllCreater_part2 import class_weights

STRATEGY = tf.distribute.MirroredStrategy()
BATCH_SIZE = 128 * STRATEGY.num_replicas_in_sync
CLASS_WEIGHTS = {0: 0.35972796, 1: 8.42605461, 2: 9.85784824}
EPOCHS = 500
MODEL_SAVE_PATH = "models/FinalModel.keras"
print(tf.keras.__version__)
print(tf.__version__)
print(pl.__version__)

# -------------------------
# Testrunning Class
# -------------------------


class Testrunning:
    """
    Class to prepare the final model and start the models final training.
    Sets up the model based on the trial.json and checkpoint.weights.h5
    Finally starts the training with the trainer() function
    Returns the trained model and saves it in the models/ directory
    """

    def __init__(
        self,
        modelpath,
        weightpath,
        dimension,
        target_dimension,
        load_weight,
        batch_size=BATCH_SIZE,
        strategy=STRATEGY,
    ):
        # Initialize the Testrunning class with model and training parameters
        self.strategy = strategy
        self.batch_size = batch_size
        self.dimension = dimension
        self.load_weight = load_weight
        self.target_dimension = target_dimension
        self.class_weights = CLASS_WEIGHTS
        self.weightpath = weightpath
        self.json_path = modelpath

        # Load datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self._loader()

        # gather model values and build the model
        self.model_values = self.fromjson(self.json_path)

        # Build the model
        self.model = self.buildfromjson()

    def fromjson(self, json_path):
        """
        Retrives the model structure from the trial.json file
        Returns the infos in json_data
        """
        # Load the JSON file and extract hyperparameters
        with open(json_path, "r") as f:
            json_data = json.load(f)
        # search for hyperparameters key and return its values
        json_data = json_data["hyperparameters"]["values"]

        return json_data

    def buildfromjson(self):
        """
        The model is rebuild via the data from the trial.json
        Additionally the already learned weights are loaded and applied to the model
        Returned is the ready to be trained model
        """

        # Set activation function and optimizer based on model values
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
                learning_rate=self.model_values["lr"] / 2,
                nesterov=True,  # , decay=1e-4 for power scheduling
            )

        # init model and the input layer
        model = Sequential()
        model.add(Input(shape=(self.target_dimension, 21)))
        model.add(Flatten())

        # add hidden layers based on the hyperparameters
        for i in range(self.model_values["n_hidden"]):
            if i <= 5:
                model.add(tf.keras.layers.Dropout(rate=self.model_values["drop_rate"]))

            (
                model.add(
                    Dense(
                        self.model_values["n_neurons"],
                        activation=activation,
                        kernel_initializer=kernel_init,
                        kernel_constraint=tf.keras.constraints.max_norm(1.0),
                        # kernel_regularizer=tf.keras.regularizers.l2(
                        # self.model_values["l2_reg"]
                    ),
                ),
            )
            # )

        # add output layer based on domain dimension (1d or 2d classifier)
        if self.dimension == 1:
            model.add(Dense(1, activation="sigmoid"))
        else:
            model.add(Dense(3, activation="softmax"))

        # compile the model with appropriate loss function and metrics for 2d classification
        if self.dimension != 1:
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[
                    "sparse_categorical_accuracy",
                    tf.keras.metrics.Precision(name="prec_1", class_id=1),
                    tf.keras.metrics.Recall(name="rec_1", class_id=1),
                    tf.keras.metrics.Precision(name="prec_2", class_id=2),
                    tf.keras.metrics.Recall(name="rec_2", class_id=2),
                    # tf.keras.metrics.AUC(name="auc_overall"),
                ],
            )

        # 1D binary classification
        else:
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    "accuracy",
                    tf.keras.metrics.Precision(name="prec"),
                    tf.keras.metrics.Recall(name="rec"),
                ],
            )

        # Load weights if specified
        if self.load_weight is True:
            model.load_weights(self.weightpath)

        # Print model summary and details
        model.summary()
        print("Optimizer:", model.optimizer)
        print("  Learning rate:", model.optimizer.learning_rate.numpy())
        print("Loss:", model.loss)
        print("Metrics:", model.metrics_names)
        for layer in model.layers:
            if hasattr(layer, "activation"):
                act_fn = layer.activation
                # print("Using activation:", act_fn.__name__)
                break
        print("\nDropout layers and rates:")
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                print(f"  {layer.name}: rate = {layer.rate}")
                break

        return model

    def _loader(self):
        """
        Loads the data from the directory to use as the mdoel input, to cut time
        Used for all further hp seaches when the sets are created
        """
        # start and load the datasets
        start_time = time.time()
        train_dataset = tf.data.Dataset.load("./Datasets/PF00210/trainset")
        val_dataset = tf.data.Dataset.load("./Datasets/PF00210/valset")
        test_dataset = tf.data.Dataset.load("./Datasets/PF00210/testset")

        # Shuffle the datasets with buffer size based on dimension
        if self.dimension != 1:
            train_dataset = train_dataset.shuffle(
                buffer_size=train_dataset.cardinality(), seed=42
            )
            val_dataset = val_dataset.shuffle(
                buffer_size=val_dataset.cardinality(), seed=42
            )
            test_dataset = test_dataset.shuffle(
                buffer_size=test_dataset.cardinality(), seed=42
            )

        else:
            train_dataset = train_dataset.shuffle(buffer_size=10000, seed=42)
            val_dataset = val_dataset.shuffle(buffer_size=10000, seed=42)
            test_dataset = test_dataset.shuffle(buffer_size=10000, seed=42)

        # Batch and prefetch the datasets for performance
        train_dataset = train_dataset.batch(self.batch_size)
        val_dataset = val_dataset.batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # prints
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in {elapsed_time:.4f} seconds")

        return train_dataset, val_dataset, test_dataset

    def trainer(self, epochnumber, modelname):
        """
        Function to start the training of the previously created model.
        Tensorboard, early stopping on val-loss and save checkpoints are used
        Finally the model is saved with the name inputted in modelname
        """
        # start time
        start_time = time.time()

        # init log dir for tensorboard
        log_dir = os.path.join("logs", time.strftime("run_%Y_%m_%d-%H_%M"))
        tensorboard_cb = TensorBoard(log_dir=log_dir)

        # create timestep for checkpoint naming
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            f"models/run_{timestamp}.keras", save_best_only=True
        )

        # early stopping callback
        early_stopping_cb = keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_loss"
        )

        # start training with callbacks
        print("Starting training")
        self.history = self.model.fit(
            self.train_dataset,
            epochs=epochnumber,
            validation_data=self.val_dataset,
            callbacks=[tensorboard_cb, early_stopping_cb, checkpoint_cb],
            class_weight=self.class_weights,
        )

        # save the final model
        self.model.save(f"models/{modelname}.keras")

        # end time and print elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done training\nElapsed Time: {elapsed_time:.4f} seconds")


##################################################################################################################################################################
if __name__ == "__main__":
    # init Testrunning class to create model
    Testrun = Testrunning(
        "./logshp/test1_1d/trial_00/trial.json",
        "./logshp/test1_1d/trial_00/checkpoint.weights.h5",
        dimension=1,
        target_dimension=148,
        load_weight=False,
    )

    # Train the model
    Testrun.trainer(EPOCHS, MODEL_SAVE_PATH)
