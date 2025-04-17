import time
import numpy as np
import pandas as pd
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(  # Disable GPU, for testing purposes, crashes on GPU
    [], "GPU"
)
from keras.preprocessing.sequence import pad_sequences
from keras.saving import load_model
from sklearn.metrics import classification_report, confusion_matrix

from Testrunner import BATCH_SIZE

##########################################################################################

df_path="./DataEvalSwiss80%.csv"
model_path='./models/my_modelnewlabeling.keras'

def predicting(modelpath, Evalset):
    def sequence_to_int(df_path):
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
        df = pd.read_csv(df_path, index_col=False)  # Open CSV file

        df = df.dropna(subset=["Sequences"])

        def encode_sequence(seq):
            return [amino_acid_to_int[amino_acid] for amino_acid in seq]

        df["Sequences"] = df["Sequences"].apply(encode_sequence)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done encoding\nElapsed Time: {elapsed_time:.4f} seconds")
        return df

    def padder(df_int):
        start_time = time.time()
        sequences = df_int["Sequences"].tolist()
        # print(type(sequences))
        # print(sequences[0:3])
        # print(self.target_dimension)
        padded = pad_sequences(
            sequences,
            maxlen=148,
            padding="post",
            truncating="post",
            value=21,
        )
        # print(padded)
        df_int["Sequences"] = list(padded)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Done padding\nElapsed Time: {elapsed_time:.4f} seconds")

        return df_int

    def labler(padded):
        start_time = time.time()
        padded["Labels"] = padded["overlap"].apply(lambda x: 1 if x == 1 else 0)
        padded_label = padded
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done labeling\nElapsed Time: {elapsed_time:.4f} seconds")
        return padded_label

    def one_hot(padded):
        start_time = time.time()
        with tf.device("/CPU:0"):
            df_one_hot = np.array(
                [
                    tf.one_hot(int_sequence, 21).numpy()
                    for int_sequence in padded["Sequences"]
                ]
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Done one hot\nElapsed Time: {elapsed_time:.4f} seconds")
            return df_one_hot

    df = sequence_to_int(Evalset)
    padder_df = padder(df)
    print(padder_df["Sequences"].apply(len).max())
    labled_df = labler(padder_df)

    # print(labled_df)$
    df_onehot = one_hot(padder_df)
    X_train = df_onehot
    y_train = labled_df["Labels"]
    # print(y_train)

    start_time = time.time()
    print("starting prediction")
    model = load_model(
        modelpath,
        custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU},
    )
    print("model loaded")
    with tf.device("/CPU:0"):
        predictions = model.predict(X_train, batch_size=BATCH_SIZE)
    print("predictions made")
    print(predictions)
    predictions_bool = []
    for value in predictions:
        if value >= 0.5:
            bool = 1
        else:
            bool = 0
        predictions_bool.append(bool)

    print(predictions_bool[0:10])
    true_labels = y_train

    print(len(true_labels))
    print(len(predictions_bool))

    def accuracy_score(y_true, y_pred):
        return (y_true == y_pred).mean()


    accuracy = accuracy_score(true_labels, predictions_bool)

    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Predictions: {predictions}")
    # print(f"True Labels: {true_labels}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Done predicting\nElapsed Time: {elapsed_time:.4f} seconds")
    print(confusion_matrix(true_labels, predictions_bool))
    print(classification_report(true_labels, predictions_bool))

    return predictions, true_labels


#####################################################################################

if __name__ == "__main__":
    predictions,true_labels=predicting(model_path, df_path)
