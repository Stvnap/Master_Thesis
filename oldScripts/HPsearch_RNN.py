###################################################################################################################################
"""
HPsearch_RNN.py

File after Dataset_preprocessing.py
This file is used to create a DNN model using the preprocessed dataset

Table of contents:
=========================
1. MyHyperModel
    build
    fit
2. Starter
    __init__
    _loader
    tuner
"""

# -------------------------
# Imports and Globals
# -------------------------

import datetime
import os
import time

import keras_tuner as kt
import tensorflow as tf
from focal_loss import SparseCategoricalFocalLoss
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import LSTM, Dense, Input
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential

# from Dataset_preprocess_TRAIN_2d import dimension_positive
# from DataAllCreater_part2 import class_weights

STRATEGY = tf.distribute.MirroredStrategy()
BATCH_SIZE = 128 * STRATEGY.num_replicas_in_sync

### CHANGE THESE ACCORDINGLY ###
DIMENSION_POSITIVE = 148
CLASS_WEIGHTS = {0: 0.35972796, 1: 8.42605461, 2: 9.85784824}
BASE_WEIGHTS = [0.35972796, 8.42605461, 9.85784824]

print("Batch Size:", BATCH_SIZE)
print(f"Number of devices: {STRATEGY.num_replicas_in_sync}")

# -------------------------
# MyHyperModel
# -------------------------


class MyHyperModel(kt.HyperModel):
    """
    Hypermodel for model structure and HP dimension.
    """

    def __init__(self, target_dimension, dimension):
        self.target_dimension = target_dimension
        self.dimension = int(dimension)

    def build(self, hp):
        """
        actual build of the model with all HP variables
        """
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # Define neuron and hidden layer HPs
        n_neurons = hp.Int("n_neurons", min_value=3100, max_value=3400)
        n_hidden = hp.Int("n_hidden", min_value=16, max_value=32)

        # Define other HPs
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        optimizer = hp.Choice("optimizer", values=["adam", "sgd"])
        activation = hp.Choice("activation", values=["leaky_relu", "sigmoid", "elu"])
        dropout_rate = hp.Float("drop_rate", min_value=0.05, max_value=0.2, step=0.05)

        # Create optimizer
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07
            )
        else:
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                nesterov=True,  # , decay=1e-4 for power scheduling
            )

        # Create activation and kernel initializer
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

        # Build the model
        model = Sequential()
        model.add(Input(shape=(self.target_dimension, 21)))
        # 2 LSTM layers
        model.add(
            LSTM(
                units=hp.Int("lstm_units1", min_value=128, max_value=512, step=128),
                return_sequences=True,
                dropout=hp.Float("lstm_drop1", 0.0, 0.1, step=0.1),
            )
        )
        model.add(
            LSTM(
                units=hp.Int("lstm_units2", min_value=64, max_value=256, step=64),
                return_sequences=False,
                dropout=hp.Float("lstm_drop2", 0.0, 0.1, step=0.1),
            )
        )

        # Fully connected layers with set HPs
        for current_layer in range(n_hidden):
            if current_layer <= 5:
                model.add(tf.keras.layers.Dropout(rate=dropout_rate))
            model.add(
                Dense(
                    n_neurons,  # number of neurons
                    activation=activation,  # activation function
                    kernel_initializer=kernel_init,  # kernel initializer
                    kernel_constraint=tf.keras.constraints.max_norm(
                        1.0
                    ),  # max norm constraint
                    # kernel_regularizer=tf.keras.regularizers.l2(
                    # hp.Float("l2_reg", min_value=1e-6, max_value=1e-2, step=1e-6)
                ),
            )

        # Output layer depending on multi domain classification or binary domain classification
        if self.dimension == 1:
            model.add(
                Dense(1, activation="sigmoid")
            )  # output layer for binary classification
            loss_name = hp.Choice(  # loss function choice
                "loss",
                values=["binary_crossentropy", "binary_focal_crossentropy"],  #
            )
            if loss_name == "binary_crossentropy":
                loss_fn = tf.keras.losses.BinaryCrossentropy()
            elif (
                loss_name == "binary_focal_crossentropy"
            ):  # if focal loss is chosen, set gamma and alpha as HPs
                gammaloss = hp.Float(
                    "gamma_loss", min_value=0.5, max_value=5.0, step=0.5
                )
                alphaloss = hp.Choice("alpha_loss", values=[0.1, 0.25, 0.5, 0.75, 0.9])
                loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                    gamma=gammaloss, alpha=alphaloss, apply_class_balancing=False
                )
        else:
            model.add(
                Dense(3, activation="softmax")
            )  # output layer for multi-class classification
            loss_name = hp.Choice(  # loss function choice
                "loss",
                values=["categorical_crossentropy", "Sigmoid_Focal_Crossentropy"],
            )
            if loss_name == "categorical_crossentropy":
                loss_fn = SparseCategoricalCrossentropy(from_logits=False)
            elif loss_name == "Sigmoid_Focal_Crossentropy":
                gammaloss = hp.Float(
                    "gamma_loss", min_value=1, max_value=3, step=1
                )  # set gamma as HP
                loss_fn = SparseCategoricalFocalLoss(
                    from_logits=False,
                    gamma=gammaloss,
                    class_weight=[0.35972796, 8.42605461, 9.85784824],
                )

        # Compile the model with chosen optimizer loss function and metrics
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="prec_1", class_id=1),
                tf.keras.metrics.Recall(name="rec_1", class_id=1),
                tf.keras.metrics.Precision(name="prec_2", class_id=2),
                tf.keras.metrics.Recall(name="rec_2", class_id=2),
            ],
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        """
        Fits the created model, very basic just the fit call
        """
        model = model.fit(verbose=1, *args, **kwargs)
        return model


# -------------------------
# Starter
# -------------------------


class Starter:
    """
    Class for preparing starting the HP search
    """

    def __init__(
        self,
        # df_path,
        dimension,
        trainpath,
        valpath,
        testpath,
        strategy=STRATEGY,
        batch_size=BATCH_SIZE,
        dimension_positive=DIMENSION_POSITIVE,
        class_weights=CLASS_WEIGHTS,
    ):
        # init vars
        self.batch_size = batch_size
        self.strategy = strategy
        self.target_dimension = dimension_positive
        self.dimension = dimension
        self.class_weights = class_weights
        self.trainpath = trainpath
        self.valpath = valpath
        self.testpath = testpath

        # load datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self._loader()

    def _loader(self):
        """
        Loads the data from the directory to use as the mdoel input, to cut time
        Used for all further hp seaches when the sets are created
        """
        # start time
        start_time = time.time()

        # load datasets
        train_dataset = tf.data.Dataset.load(self.trainpath)
        val_dataset = tf.data.Dataset.load(self.valpath)
        test_dataset = tf.data.Dataset.load(self.testpath)

        # shuffle, batch and prefetch datasets
        train_dataset = train_dataset.shuffle(buffer_size=10000, seed=42)
        val_dataset = val_dataset.shuffle(buffer_size=10000, seed=42)
        test_dataset = test_dataset.shuffle(buffer_size=10000, seed=42)

        train_dataset = train_dataset.batch(self.batch_size)
        val_dataset = val_dataset.batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)

        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # check dataset shapes and final prints
        for x, y in train_dataset.take(1):
            print("Input shape:", x.shape)
            print("Label shape:", y.shape)
        print(f"Loading completed in {time.time() - start_time:.2f} seconds")
        return train_dataset, val_dataset, test_dataset

    def tuner(self, trials, epochs, run):
        """
        Iniziation function for the HP search, BayesianOptimization search is used with tensorbaord callbacks,
        as well as model saves for the best model so far. Early stopping is used with patience on 5 and val_loss
        as monitor.
        Finally the best 3 models are saved after one HP search run
        Can be used for multiple HP searches, overwrite=False
        """
        # start time and timestamp init
        start_time = time.time()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directories for logs
        log_dir = "./logshp/ModelForSP2d_RNN_tb"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs("logshp/weights", exist_ok=True)

        # Setup callbacks with early stopping, checkpointing and tensorboard
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

        # Initialize tuner
        self.tuner = kt.BayesianOptimization(
            hypermodel=MyHyperModel(
                target_dimension=self.target_dimension, dimension=self.dimension
            ),  # hypermodel with set target dimension
            objective="val_loss",  # objective to minimize val_loss
            max_trials=trials,  # number of trials
            overwrite=False,  # do not overwrite previous results
            directory="./logshp",  # directory to save results
            # distribution_strategy=tf.distribute.MirroredStrategy(), # use mirrored strategy for multi gpu
            project_name="ModelForSP2drun2_RNN",  # project name
        )

        # Print search space summary
        self.tuner.search_space_summary()

        # Search start
        self.tuner.search(
            self.train_dataset,  # training dataset
            epochs=epochs,  # number of epochs per trial
            validation_data=self.val_dataset,  # validation dataset
            callbacks=[tensorboard, checkpoint_cb, early_stopping],  # callbacks
            class_weight=self.class_weights,  # removed when using focal loss
        )

        # Print results summary and save best 3 models
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
        os.makedirs(f"./bestmodels/{timestamp}", exist_ok=True)
        models = run.tuner.get_best_models(num_models=3)
        best_model = models[0]
        # print best model summary
        best_model.summary()
        best_model.save(f"./bestmodels/{timestamp}/best_model.keras")

        second_model = models[1]
        second_model.save(f"./bestmodels/{timestamp}/second_best_model.keras")

        third_model = models[2]
        third_model.save(f"./bestmodels/{timestamp}/third_best_model.keras")

        elapsed_time = time.time() - start_time
        print(f"Done tuning\nElapsed Time: {elapsed_time:.4f} seconds")


# -------------------------
# main
# -------------------------


def main():
    """
    start the whole HP search
    """
    print(tf.config.list_physical_devices("GPU"), "\n", "\n", "\n", "\n")
    run = Starter(
        dimension=2,
        trainpath="./trainsetSP2d",
        valpath="./valsetSP2d",
        testpath="./testsetSP2d",
    )
    run.tuner(trials=15, epochs=10, run=run)


##################################################
if __name__ == "__main__":
    main()
