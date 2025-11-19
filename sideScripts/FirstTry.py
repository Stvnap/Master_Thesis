####
# check dataset for bias for some domains (some domains are propably overrepresented)
####
################################################################################
# First try to create an protein embedding based on a one hot vector representing the amino acids
# Different models are tested (one linear model, one 1d CNN)
# The data is taken from swissprot database 
################################################################################
# creation of the dataset using a pre downloaded uniprot.dat text file
# search for the keyword "DOMAIN" in the file and extract the domain sequences
# the sequences are stored in a list and the corresponding domain descriptions are stored in another list

from Bio import SwissProt
from Bio.SeqFeature import UnknownPosition

def extract_described_domains_from_swissprot(filepath):
    with open(filepath, "r") as handle:
        for record in SwissProt.parse(handle):
            full_seq = record.sequence
            accession = record.accessions[0] if record.accessions else record.entry_name

            for feature in record.features:
                if feature.type == "DOMAIN":                # search for the keyword "DOMAIN"
                    start_pos = feature.location.start
                    end_pos   = feature.location.end

                    # Skip if positions are unknown
                    if isinstance(start_pos, UnknownPosition) or isinstance(end_pos, UnknownPosition):
                        continue

                    # Try multiple qualifier keys
                    desc = (
                        feature.qualifiers.get("description")
                        or feature.qualifiers.get("note")
                        or feature.qualifiers.get("label")
                        or feature.qualifiers.get("region_name")
                        or ""
                    )
                    desc = desc.strip()  # Remove extra whitespace

                    # Only proceed if we have a non-empty descriptor
                    if desc:
                        start = int(start_pos)
                        end   = int(end_pos)
                        
                        # Slice the domain sequence
                        domain_seq = full_seq[start:end]

                        yield (
                            accession,
                            desc,
                            start + 1,  # converting to 1-based index for output
                            end,
                            domain_seq
                        )


dom_seq_all=[]
dom_desc_all=[]

# Usage
if __name__ == "__main__":
    swissprot_file = "/home/steven/Downloads/uniprot_sprot.dat"
    
    for dom_info in extract_described_domains_from_swissprot(swissprot_file):
        acc, dom_desc, dom_start, dom_end, dom_seq = dom_info
        print(f"Accession: {acc}")
        print(f"Domain: {dom_desc}")
        print(f"Location: {dom_start}-{dom_end}")
        print(f"Domain Sequence (first 30 chars): {dom_seq[:30]}...")
        print("-" * 50)

        dom_seq_all.append(dom_seq)
        dom_desc_all.append(dom_desc)



#Creation of dictionary for the amino acids, X Z B U are special characters found in the dataset and are treated as unknown

amino_acid_to_int = {
    'A': 1,  # Alanine
    'C': 2,  # Cysteine
    'D': 3,  # Aspartic Acid
    'E': 4,  # Glutamic Acid
    'F': 5,  # Phenylalanine
    'G': 6,  # Glycine
    'H': 7,  # Histidine
    'I': 8,  # Isoleucine
    'K': 9,  # Lysine
    'L': 10, # Leucine
    'M': 11, # Methionine
    'N': 12, # Asparagine
    'P': 13, # Proline
    'Q': 14, # Glutamine
    'R': 15, # Arginine
    'S': 16, # Serine
    'T': 17, # Threonine
    'V': 18, # Valine
    'W': 19, # Tryptophan
    'Y': 20, # Tyrosine
    'X': 21, # Unknown or special character
    'Z': 22, # Glutamine (Q) or Glutamic acid (E)
    'B': 23, # Asparagine (N) or Aspartic acid (D) 
    'U': 24  # Selenocysteine
}

print(len(dom_seq_all))



# Convert the protein sequence to a list of integers
def sequence_to_int(sequence, mapping):
    return [mapping[amino_acid] for amino_acid in sequence]

sequences_int = [sequence_to_int(sequence, amino_acid_to_int) for sequence in dom_seq_all]

print(len(sequences_int))

# padding the sequences to 330 aa to create a fixed length input for the model, padding is done with zeros at the end of the sequence
# needed for the model to have a fixed input dimension
# could need further investigating into the optimal length of the sequences

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 330
padded_sequences = pad_sequences(sequences_int, maxlen=max_length, padding='post', truncating='post')

print(padded_sequences)


# create the one hot vector for the amino acids
import tensorflow as tf

# Function to convert the list of integers to a one-hot encoded tensor
def one_hot_encode_sequence(int_sequence, num_classes=24):

    return tf.one_hot(int_sequence, num_classes)
sequences_one_hot = [one_hot_encode_sequence(int_sequence) for int_sequence in padded_sequences]


# test prints
print(sequences_one_hot)
print(sequences_one_hot[0],dom_desc_all[0])
print(len(sequences_one_hot),len(dom_desc_all))




# determine the number of unique domains in dom_desc_all
unique_desc = set(dom_desc_all)
# Number of unique strings:
num_unique = len(unique_desc)
print(num_unique)

# create a set for the domain descriptions
unique_domains = sorted(set(dom_desc_all))
# Create a TensorFlow StringLookup Layer
lookup_layer = tf.keras.layers.StringLookup(vocabulary=unique_domains)

# Convert your dataset so each label is transformed from string to integer, needed for the model if we use sparse_categorical_crossentropy
def map_string_to_int(seq, label_str):
    label_id = lookup_layer(label_str)
    return seq, label_id

# convertion step
combined_dataset = tf.data.Dataset.from_tensor_slices((sequences_one_hot, dom_desc_all))



# creation of train, validation and test dataset from the combined dataset
import tensorflow as tf

# Create a combined dataset of (one_hot, label) pairs
combined_dataset = tf.data.Dataset.from_tensor_slices((sequences_one_hot, dom_desc_all))
# shuffle
combined_dataset = combined_dataset.shuffle(buffer_size=218468, reshuffle_each_iteration=False)

# split sizes
dataset_size = len(sequences_one_hot)
train_size = int(0.70 * dataset_size)   # 70%
val_size   = int(0.15 * dataset_size)   # 15%
test_size  = dataset_size - train_size - val_size  # 15%

# split the sets
train_dataset = combined_dataset.take(train_size)
val_dataset   = combined_dataset.skip(train_size).take(val_size)
test_dataset  = combined_dataset.skip(train_size + val_size)

# Convert label strings to integer IDs (see above)
train_dataset = train_dataset.map(map_string_to_int)
val_dataset   = val_dataset.map(map_string_to_int)
test_dataset  = test_dataset.map(map_string_to_int)

# Batch & prefetch
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset   = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset  = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)



#First linear model with some tweaks to improve the performance (dropout, L2 regularization)
#input shape is the 330 aa one hot vector (padded)
#output shape is the number of unique domains in the whole dataset

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers


model = Sequential([
    Flatten(input_shape=(330, 24)),  
    Dense(300, 
          activation='relu', 
          kernel_regularizer=regularizers.l2(0.001)),  # L2 on this Dense layer
    Dropout(0.3),
    Dropout(0.3),
    Dense(300, 
          activation='relu', 
          kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(3675, 
          activation='softmax', 
          kernel_regularizer=regularizers.l2(0.001)) 
])
model.summary()

# Compile the model with basic settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Plot the model as a png
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)



# second model is a 1d CNN model with global max pooling
# input shape is the 330 aa one hot vector (padded)
# output shape is the number of unique domains in the whole dataset


import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers


model1dCNN = Sequential([
    # For input shape (sequence_length=330, embedding_dim=24)
    layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                  input_shape=(330, 24)),
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.GlobalMaxPooling1D(),  # pool over the sequence axis
    layers.Dense(256, activation='relu'),
    layers.Dense(3675, activation='softmax')  # final output for your classes
])
model1dCNN.summary()

# basic settings for compiling
model1dCNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Plot the model as png
# plot_model(model1dCNN, to_file='model.png', show_shapes=True, show_layer_names=True)



# Train the linear model with the train dataset
# with reduced learning rate on plateau, early stopping and checkpointing and Tensorboard visualization
# validation set is used to monitor the performance of the model

import os
import time
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-6, verbose=1)

# Set up TensorBoard callback
log_dir = os.path.join("logs", time.strftime("run_%Y_%m_%d-%H_%M_%S"))
tensorboard_cb = TensorBoard(log_dir=log_dir)

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model_third.h5",
save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
restore_best_weights=True,monitor='val_loss')

# Train the model with TensorBoard callback
history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[tensorboard_cb,early_stopping_cb,checkpoint_cb,reduce_lr])


# load in model previously created and evaluate it on the test dataset

model = keras.models.load_model("my_keras_model_second.h5")
model.evaluate(test_dataset)



# Train the 1dCNN model with the train dataset
# with reduced learning rate on plateau, early stopping and checkpointing and Tensorboard visualization
# validation set is used to monitor the performance of the model
import os
import time
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-6, verbose=1)

# Set up TensorBoard callback
log_dir = os.path.join("logs", time.strftime("run_%Y_%m_%d-%H_%M_%S"))
tensorboard_cb = TensorBoard(log_dir=log_dir)

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model_1dCNN_2.h5",
save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
restore_best_weights=True,monitor='val_loss')

# Train the model with TensorBoard callback
history = model1dCNN.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[tensorboard_cb,early_stopping_cb,checkpoint_cb,reduce_lr])



# load in model previously created and evaluate it on the test dataset

model = keras.models.load_model("my_keras_model_1dCNN.h5")
model.evaluate(test_dataset)



# second 1d CNN model with some tweaks to improve the performance (dropout, L2 regularization, batch normalization, max & global pooling) based on chatGPTs suggestions
# still the same input and output shape as the previous model
# still needs to be compiled with the same settings

import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers

# Final 1D CNN model
model_1dCNN = Sequential([
    # 1) First Conv Block
    layers.Conv1D(
        filters=128,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-3),
        input_shape=(330, 24)  # (sequence_length=330, embedding_dim=24)
    ),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    # 2) Second Conv Block
    layers.Conv1D(
        filters=128,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-3)
    ),
    layers.BatchNormalization(),

    # 3) Global Pooling & Dropout
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.3),

    # 4) Dense Head
    layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-3)
    ),
    layers.Dropout(0.3),

    # 5) Final Output Layer
    layers.Dense(3675, activation='softmax')  # Adjust if you have a different # of classes
])

# Compile with an optimizer, loss, and evaluation metric
model_1dCNN.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model_1dCNN.summary()
# plot_model(model_1dCNN, to_file='model.png', show_shapes=True, show_layer_names=True)



# Train the 1dCNN model with the train dataset
# with reduced learning rate on plateau, early stopping and checkpointing and Tensorboard visualization
# validation set is used to monitor the performance of the model
# still needs to be actually trained, not done yet

import os
import time
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-6, verbose=1)

# Set up TensorBoard callback
log_dir = os.path.join("logs", time.strftime("run_%Y_%m_%d-%H_%M_%S"))
tensorboard_cb = TensorBoard(log_dir=log_dir)

checkpoint_cb = keras.callbacks.ModelCheckpoint("model_1dCNN_2.h5",
save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
restore_best_weights=True,monitor='val_loss')

# Train the model with TensorBoard callback
history = model_1dCNN.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[tensorboard_cb,early_stopping_cb,checkpoint_cb,reduce_lr])