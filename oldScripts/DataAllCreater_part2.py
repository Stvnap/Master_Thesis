"""
DataAllCreater_part2.py

File for creating the whole Training Dataset in tf.Dataset format to be used for training for 1D and 2D models.
INFOS:
Use enough RAM for this script

Table of Contents:
=========================
1. testsplit1d
2. testsplit2d
3. compute_classweights
"""

# -------------------------
# Imports & Globals
# -------------------------

import time

import numpy as np
import polars as pl
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

DF_PATH = "./saved_dataset2dNew"


# -------------------------
# Functions for 1d splitting
# -------------------------


def splitcalculation(ds):
    """
    Calculates the ratio of positives (1) / all (0+1) in the whole dataset
    """
    # start time
    start_time = time.time()

    # init counters
    total = len(ds)
    positives = 0

    # count positives, two methods to ensure compatibility
    try:
        for _, label, _ in ds.as_numpy_iterator():
            positives += int(label)
    except:
        for _, label, _ in ds:
            positives += int(label)

    # final print
    print("Positives:", positives, "Total:", total)
    print(f"Splitcalculation done in: {time.time() - start_time:.2f} seconds")

    return positives / total


def splitter(dataset, frac, size):
    """
    main split function to split the dataset,
    takes the label,id, sample(seq) out of the ds
    fills up the testset with positive hits up to the previously
    calculated fraction
    elif fills up the testset to the desired soze
    fills the trainset with the rest
    returns trainset, testset
    """
    # start print and inits
    start_time = time.time()
    testset = []
    trainset = []

    # calculate max sizes to split to
    max_testset_size = int(size * len(dataset))
    max_test_positives = int(frac * max_testset_size)

    # counters
    current_test_positives = 0
    current_test_total = 0

    # main loop to split with different methods to ensure compatibility
    for seq, label, id_ in dataset:
        try:
            label_val = int(label.numpy())
        except:
            label_val = int(label)
        try:
            id_val = id_.numpy().decode("utf-8")
        except:
            id_val = id_
        try:
            sample = (seq.numpy(), label_val, id_val)
        except:
            sample = (seq, label_val, id_val)

        # filling testset with positives first
        if label_val == 1 and current_test_positives < max_test_positives:
            testset.append(sample)
            current_test_positives += 1
            current_test_total += 1
        # filling testset to max size
        elif current_test_total < max_testset_size:
            testset.append(sample)
            current_test_total += 1
        # filling trainset with the rest
        else:
            trainset.append(sample)

    # final print
    print(
        f"Split done in: {time.time() - start_time:.2f} seconds Train={len(trainset)}, Test={len(testset)}, Positives in test={current_test_positives}"
    )

    return trainset, testset


def save_ids_as_parquet(dataset_list, out_path):
    """
    Save function to save the IDs found in the given set to parquet files
    """
    # start time
    start_time = time.time()

    # check for empty list
    if not dataset_list:
        print(f"  ID list for {out_path} is empty.")
        df = pl.DataFrame({"ID": []})
    # make df with IDs
    else:
        ids = [id_ for _, _, id_ in dataset_list]
        df = pl.DataFrame({"ID": ids})

    # save parquet and print info
    df.write_parquet(out_path)
    print(
        f"Saved Parquet in: {time.time() - start_time:.2f} seconds | {len(df)} IDs → {out_path}"
    )


def convert_and_save(dataset_list, save_path):
    """
    Save function for creating the final tf.dataset from the list
    """
    # start time
    start_time = time.time()

    # check for empty list
    if not dataset_list:
        print(f"Cannot save '{save_path}' — dataset is empty.")
        raise KeyError("dataset empty")

    # unzip list into seqs, labels, ids
    seqs, labels, ids = zip(*dataset_list)

    # create tf.dataset and save
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            (
                list(seqs),
                list(labels),  # id is dropped, no need for training
            )
        )
    ds.save(save_path)

    # final print
    print(f"Saved df in {save_path} in: {time.time() - start_time:.2f} seconds")

    return ds


def positive_label_percentage(ds):
    """
    Calculates the percentage of samples with label == 1 in a tf.data.Dataset.
    Assumes dataset yields (features, label) or (seq, label, id).
    """

    # init counters
    total = 0
    positives = 0

    # loop trough each sample
    for sample in ds:
        # Unpack label
        if len(sample) == 2:
            _, label = sample
        else:
            _, label, _ = sample

        # Convert to int and count positives
        label_val = label
        positives += label_val
        total += 1

    # error check for empty dataset
    if total == 0:
        raise ValueError("Dataset is empty.")

    # calculate and print percentage
    percentage = (positives / total) * 100
    print(f"Positive label percentage: {percentage:.4f}% ({positives}/{total})")
    return percentage


# -------------------------
# Testsplit function for 1d
# -------------------------


def testsplit1d(df_path, doublecheck=False):
    """
    Final preprocessing function to split the tf.dataset into train,val,testset
    Spliting is done via a manual stratified split by label to ensure equal
    distribution of target domains in all sets
    For each set a parquet file is saved with the IDs included in the sets
    Finally each set is saved in a tf.datasets for the training
    """
    # start time
    start_time = time.time()

    # load in
    df = tf.data.Dataset.load(df_path)

    # Step 1: Full train/test split
    frac_full = splitcalculation(df)
    trainset, testset = splitter(df, frac_full, size=0.4)
    print("Train/Test split done")

    # Step 2: Further split test into val/test
    frac_test = splitcalculation(testset)  # positive-rate within your testset
    valset, testset = splitter(testset, frac_test, size=0.5)
    print("Test/Val split done")

    # Save IDs as Parquet
    save_ids_as_parquet(trainset, "train_IDsSP2d.parquet")
    save_ids_as_parquet(valset, "val_IDsSP2d.parquet")
    save_ids_as_parquet(testset, "test_IDsSP2d.parquet")

    # Save tf.datasets
    convert_and_save(trainset, "./trainsetSP2d")
    convert_and_save(valset, "./valsetSP2d")
    convert_and_save(testset, "./testsetSP2d")

    if doublecheck is True:
        positive_label_percentage(trainset)
        positive_label_percentage(valset)
        positive_label_percentage(testset)

    print(f"Complete split done in: {time.time() - start_time:.2f} seconds")

    return trainset, valset, testset, class_weights


# -------------------------
# Functions for 2d splitting
# -------------------------


def splitcalculation2d(ds):
    """
    Returns the ratio of class 1 and class 2 relative to the dataset. For 2d datasets.
    """
    # counters and total
    total = len(ds)
    count_1, count_2 = 0, 0

    # count positives, two methods to ensure compatibility
    try:
        for _, label, _ in ds.as_numpy_iterator():
            label = int(label)
            if label == 1:
                count_1 += 1
            elif label == 2:
                count_2 += 1
    except:
        for _, label, _ in ds:
            label = int(label)
            if label == 1:
                count_1 += 1
            elif label == 2:
                count_2 += 1

    # print info
    print(f"Total: {total}, Class 1: {count_1}, Class 2: {count_2}")

    return total, count_1, count_2


def splitter2d(dataset, total, count_1, count_2, size):
    """
    main split function to split the dataset,
    takes the label,id, sample(seq) out of the ds
    fills up the testset with positive hits up to the previously
    calculated fraction
    elif fills up the testset to the desired soze
    fills the trainset with the rest
    returns trainset, testset
    """
    # start time and init
    start_time = time.time()
    testset = []
    trainset = []

    # calculate fractions and get max sizes
    frac_1 = count_1 / total
    frac_2 = count_2 / total
    max_test_size = int(size * total)
    max_1 = int(max_test_size * frac_1)
    max_2 = int(max_test_size * frac_2)

    # counters
    curr_test_1, curr_test_2, curr_test_total = 0, 0, 0

    # main loop to split with different methods to ensure compatibility
    for seq, label, id_ in dataset:
        # try to convert values
        try:
            label_val = int(label.numpy())
            id_val = id_.numpy().decode("utf-8")
            seq_val = seq.numpy()
        # or take as is
        except:
            label_val = int(label)
            id_val = id_
            seq_val = seq

        # make sample tuple
        sample = (seq_val, label_val, id_val)

        # filling testset with positives first
        if label_val == 1 and curr_test_1 < max_1:
            testset.append(sample)
            curr_test_1 += 1
            curr_test_total += 1
        # filling testset with class 2 next
        elif label_val == 2 and curr_test_2 < max_2:
            testset.append(sample)
            curr_test_2 += 1
            curr_test_total += 1
        # filling testset to max size
        elif curr_test_total < max_test_size:
            testset.append(sample)
            curr_test_total += 1
        # filling trainset with the rest
        else:
            trainset.append(sample)

    # final print
    print(
        f"Split done in: {time.time() - start_time:.2f} seconds | Train={len(trainset)}, Test={len(testset)} | Class 1: {curr_test_1}, Class 2: {curr_test_2}"
    )
    return trainset, testset


def save_ids_as_parquet2d(dataset_list, out_path):
    """
    Save function to save the IDs found in the given set to parquet files
    """
    # start time
    start_time = time.time()

    # check for empty list
    if not dataset_list:
        print(f"  ID list for {out_path} is empty.")
        df = pl.DataFrame({"ID": [], "Label": []})

    # make df with IDs and Labels
    else:
        ids = [id_ for _, _, id_ in dataset_list]
        labels = [label for _, label, _ in dataset_list]
        df = pl.DataFrame(
            {
                "ID": ids,
                "Label": labels,
            }
        )

    # save parquet and print info
    df.write_parquet(out_path)
    print(
        f"Saved Parquet in: {time.time() - start_time:.2f} seconds | "
        f"{len(df)} rows → {out_path}"
    )


def convert_and_save2d(dataset_list, save_path):
    """
    Save function for creating the final tf.dataset
    """
    # start time
    start_time = time.time()
    # check for empty list
    if not dataset_list:
        print(f"  Cannot save '{save_path}' — dataset is empty.")
        raise KeyError("dataset empty")

    # unzip list into
    seqs, labels, ids = zip(*dataset_list)
    # create tf.dataset and save
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            (
                list(seqs),
                list(labels),  # id is dropped, no need for training
            )
        )
    # save ds and final print
    ds.save(save_path)
    print(f"Saved df in {save_path} in: {time.time() - start_time:.2f} seconds")
    return ds


def positive_label_percentage2d(ds):
    """
    Calculates the percentage of samples with label == 1 or 2 in a tf.data.Dataset.
    Assumes dataset yields (seq, label) or (seq, label, id) and uses integer labels.
    """
    # init counters
    total = 0
    positives1 = 0
    positives2 = 0
    # go through each sample
    for sample in ds:
        # Unpack label two methods
        if len(sample) == 2:
            _, label = sample
        else:
            _, label, _ = sample

        # Convert to int
        label_val = int(label.numpy()) if hasattr(label, "numpy") else int(label)

        # Count positives
        if label_val == 1:
            positives1 += 1
        elif label_val == 2:
            positives2 += 1
        total += 1

    # error check for empty dataset
    if total == 0:
        print("  Warning: Dataset is empty.")
        return 0.0

    # calculate and print percentages
    percentage1 = (positives1 / total) * 100
    percentage2 = (positives2 / total) * 100
    print(f"Positive class 1 percentage: {percentage1:.4f}% ({positives1}/{total})")
    print(f"Positive class 2 percentage: {percentage2:.4f}% ({positives2}/{total})")

    return percentage1, percentage2


# -------------------------
# Testsplit function for 2d
# -------------------------


def testsplit2d(df_path, doublecheck=False):
    """
    Final preprocessing function to split the tf.dataset into train,val,testset
    Spliting is done via a manual stratified split by label to ensure equal
    distribution of target domains in all sets
    For each set a parquet file is saved with the IDs included in the sets
    Finally each set is saved in a tf.datasets for the training
    """
    # start time
    start_time = time.time()

    # load in
    df = tf.data.Dataset.load(df_path)

    # Full train/test split
    total, count_1, count_2 = splitcalculation2d(df)
    trainset, testset = splitter2d(df, total, count_1, count_2, size=0.4)
    print("Train/Test split done")

    # Further split test into val/test
    total_test, count_1_test, count_2_test = splitcalculation2d(testset)
    valset, testset = splitter2d(
        testset, total_test, count_1_test, count_2_test, size=0.5
    )
    print("Test/Val split done")

    # Save IDs as Parquet
    save_ids_as_parquet2d(trainset, "train_IDsSP2d.parquet")
    save_ids_as_parquet2d(valset, "val_IDsSP2d.parquet")
    save_ids_as_parquet2d(testset, "test_IDsSP2d.parquet")

    # Save tf.datasets
    convert_and_save2d(trainset, "./trainsetSP2d")
    convert_and_save2d(valset, "./valsetSP2d")
    convert_and_save2d(testset, "./testsetSP2d")

    # doublecheck percentages of splits
    if doublecheck is True:
        positive_label_percentage2d(trainset)
        positive_label_percentage2d(valset)
        positive_label_percentage2d(testset)

    print(f"Complete split done in: {time.time() - start_time:.2f} seconds")

    return trainset, valset, testset


# -------------------------
# Class weight computation Function
# -------------------------


def compute_classweights(df_path):
    """
    Computes class weights for imbalanced datasets to be used as training weights.
    """
    # load in
    df = tf.data.Dataset.load(df_path)
    print("loaded")

    # extract labels and get unique classes
    label_ds = df.map(lambda x, y, z: y)
    labels = list(label_ds.as_numpy_iterator())
    classes = np.unique(labels)

    # compute class weights package from sklearn
    class_weights = compute_class_weight(
        "balanced",
        classes=classes,
        y=labels,
    )

    return class_weights


##################################################################################

# weight computation can be called outside this script
class_weights = compute_classweights(DF_PATH)
print("Class weights:", class_weights)
if __name__ == "__main__":
    print("Starting data splitting...")
    train_out, val_out, test_out = testsplit2d(DF_PATH, doublecheck=True)
    print("all done")
