###################################################################################################################################
"""
File for creating the whole Training Dataset in tf.Dataset format to be used for training

INFOS:
Use enough RAM for this script

"""
###################################################################################################################################

import random
import time
from collections import Counter, defaultdict
import numpy as np
import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

###################################################################################################################################


def testsplit1d(df_path,doublecheck=False):
    """
    Final preprocessing function to split the tf.dataset into train,val,testset
    Spliting is done via a manual stratified split by label to ensure equal
    distribution of target domains in all sets
    For each set a parquet file is saved with the IDs included in the sets
    Finally each set is saved in a tf.datasets for the training
    """
    start_time = time.time()
    # load in
    df = tf.data.Dataset.load(df_path)

    def splitcalculation(ds):
        """
        Calculates the ratio of positives (1) / all (0+1) in the whole dataset
        """
        start_time = time.time()

        total = len(ds)
        positives = 0
        try:
            for _, label, _ in ds.as_numpy_iterator():
                positives += int(label)
        except:
            for _, label, _ in ds:
                positives += int(label)
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
        start_time = time.time()
        testset = []
        trainset = []

        # total_positives = sum(1 for _, label, _ in dataset if int(label) == 1)

        max_testset_size = int(size * len(dataset))
        # print(max_testset_size)
        max_test_positives = int(frac * max_testset_size)
        # print(max_test_positives)

        current_test_positives = 0
        current_test_total = 0

        for seq, label, id_ in dataset:
            try:
                label_val = int(label.numpy())
                # print(label_val)

            except:
                label_val = int(label)
                # print(label_val)
            try:
                id_val = id_.numpy().decode("utf-8")
            except:
                id_val = id_
            try:
                sample = (seq.numpy(), label_val, id_val)
            except:
                sample = (seq, label_val, id_val)

            if label_val == 1 and current_test_positives < max_test_positives:
                testset.append(sample)
                current_test_positives += 1
                current_test_total += 1
            elif current_test_total < max_testset_size:
                testset.append(sample)
                current_test_total += 1
            else:
                trainset.append(sample)

        print(
            f"✅ Split done in: {time.time() - start_time:.2f} seconds Train={len(trainset)}, Test={len(testset)}, Positives in test={current_test_positives}"
        )
        return trainset, testset

        


    def save_ids_as_parquet(dataset_list, out_path):
        """
        Save function to save the IDs found in the given set to parquet files
        """
        start_time = time.time()
        if not dataset_list:
            print(f"⚠️ ID list for {out_path} is empty.")
            df = pl.DataFrame({"ID": []})
        else:
            ids = [id_ for _, _, id_ in dataset_list]
            df = pl.DataFrame({"ID": ids})

        df.write_parquet(out_path)
        print(
            f"✅ Saved Parquet in: {time.time() - start_time:.2f} seconds | {len(df)} IDs → {out_path}"
        )

    def convert_and_save(dataset_list, save_path):
        """
        Save function for creating the final tf.dataset
        """
        start_time = time.time()
        if not dataset_list:
            print(f"⚠️ Cannot save '{save_path}' — dataset is empty.")
            raise KeyError("dataset empty")

        seqs, labels, ids = zip(*dataset_list)
        with tf.device("/CPU:0"):
            ds = tf.data.Dataset.from_tensor_slices(
                (
                    list(seqs),
                    list(labels),  # id is dropped, no need for training
                )
            )
        ds.save(save_path)
        print(f"Saved df in {save_path} in: {time.time() - start_time:.2f} seconds")
        return ds

    def positive_label_percentage(ds):
        """
        Calculates the percentage of samples with label == 1 in a tf.data.Dataset.
        Assumes dataset yields (features, label) or (seq, label, id).
        """
        total = 0
        positives = 0

        for sample in ds:
            if len(sample) == 2:
                _, label = sample
            else:
                _, label, _ = sample

            label_val = label
            positives += label_val
            total += 1

        if total == 0:
            print("⚠️ Warning: Dataset is empty.")
            return 0.0

        percentage = (positives / total) * 100
        print(f"✅ Positive label percentage: {percentage:.4f}% ({positives}/{total})")
        return percentage

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



def testsplit2d(df_path,doublecheck=False):
    """
    Final preprocessing function to split the tf.dataset into train,val,testset
    Spliting is done via a manual stratified split by label to ensure equal
    distribution of target domains in all sets
    For each set a parquet file is saved with the IDs included in the sets
    Finally each set is saved in a tf.datasets for the training
    """
    start_time = time.time()
    # load in
    df = tf.data.Dataset.load(df_path)

    def splitcalculation(ds):
        """
        Returns the ratio of class 1 and class 2 relative to the dataset.
        """
        total = len(ds)
        count_1, count_2 = 0, 0
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

        print(f"Total: {total}, Class 1: {count_1}, Class 2: {count_2}")
        return total, count_1, count_2


    def splitter(dataset, total, count_1, count_2, size):
        """
        main split function to split the dataset,
        takes the label,id, sample(seq) out of the ds
        fills up the testset with positive hits up to the previously
        calculated fraction
        elif fills up the testset to the desired soze
        fills the trainset with the rest
        returns trainset, testset
        """
        start_time = time.time()
        testset, trainset = [], []

        frac_1 = count_1 / total
        frac_2 = count_2 / total

        max_test_size = int(size * total)
        max_1         = int(max_test_size * frac_1)
        max_2         = int(max_test_size * frac_2)

        curr_test_1, curr_test_2, curr_test_total = 0, 0, 0

        for seq, label, id_ in dataset:
            try:
                label_val = int(label.numpy())
                id_val = id_.numpy().decode("utf-8")
                seq_val = seq.numpy()
            except:
                label_val = int(label)
                id_val = id_
                seq_val = seq

            sample = (seq_val, label_val, id_val)

            if label_val == 1 and curr_test_1 < max_1:
                testset.append(sample)
                curr_test_1 += 1
                curr_test_total += 1
            elif label_val == 2 and curr_test_2 < max_2:
                testset.append(sample)
                curr_test_2 += 1
                curr_test_total += 1
            elif curr_test_total < max_test_size:
                testset.append(sample)
                curr_test_total += 1
            else:
                trainset.append(sample)

        print(
            f"✅ Split done in: {time.time() - start_time:.2f} seconds | Train={len(trainset)}, Test={len(testset)} | Class 1: {curr_test_1}, Class 2: {curr_test_2}"
        )
        return trainset, testset
    

    def save_ids_as_parquet(dataset_list, out_path):
        """
        Save function to save the IDs found in the given set to parquet files
        """
        start_time = time.time()
        if not dataset_list:
            print(f"⚠️ ID list for {out_path} is empty.")
            df = pl.DataFrame({"ID": [], "Label": []})
        else:
            ids    = [id_    for _, _, id_    in dataset_list]
            labels = [label for _, label, _   in dataset_list]
            df = pl.DataFrame({
                "ID":    ids,
                "Label": labels,
            })

        df.write_parquet(out_path)
        print(
            f"✅ Saved Parquet in: {time.time() - start_time:.2f} seconds | "
            f"{len(df)} rows → {out_path}"
        )

    def convert_and_save(dataset_list, save_path):
        """
        Save function for creating the final tf.dataset
        """
        start_time = time.time()
        if not dataset_list:
            print(f"⚠️ Cannot save '{save_path}' — dataset is empty.")
            raise KeyError("dataset empty")

        seqs, labels, ids = zip(*dataset_list)
        with tf.device("/CPU:0"):
            ds = tf.data.Dataset.from_tensor_slices(
                (
                    list(seqs),
                    list(labels),  # id is dropped, no need for training
                )
            )
            # ds = ds.map(lambda seq, label: (seq, tf.one_hot(label, depth=3)),
            # num_parallel_calls=tf.data.AUTOTUNE)
        ds.save(save_path)
        print(f"Saved df in {save_path} in: {time.time() - start_time:.2f} seconds")
        return ds

    def positive_label_percentage(ds):
        """
        Calculates the percentage of samples with label == 1 or 2 in a tf.data.Dataset.
        Assumes dataset yields (seq, label) or (seq, label, id) and uses integer labels.
        """
        total = 0
        positives1 = 0
        positives2= 0
        for sample in ds:
            # Unpack label
            if len(sample) == 2:
                _, label = sample
            else:
                _, label, _ = sample

            # Convert to int
            label_val = int(label.numpy()) if hasattr(label, "numpy") else int(label)

            # Count positives
            if label_val ==1:
                positives1 += 1
            elif label_val==2:
                positives2 +=1
            total += 1

        if total == 0:
            print("⚠️ Warning: Dataset is empty.")
            return 0.0

        percentage1 = (positives1 / total) * 100
        percentage2 = (positives2 / total) * 100

        print(f"✅ Positive class 1 percentage: {percentage1:.4f}% ({positives1}/{total})")
        print(f"✅ Positive class 2 percentage: {percentage2:.4f}% ({positives2}/{total})")

        return percentage1,percentage2


    # Step 1: Full train/test split
    total, count_1, count_2 = splitcalculation(df)
    trainset, testset = splitter(df, total, count_1, count_2, size=0.4)
    print("Train/Test split done")

    # Step 2: Further split test into val/test
    total_test, count_1_test, count_2_test = splitcalculation(testset)
    valset, testset = splitter(testset, total_test, count_1_test, count_2_test, size=0.5)
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

    return trainset, valset, testset



def compute_classweights(df_path):
    df = tf.data.Dataset.load(df_path)
    print("loaded")
    label_ds = df.map(lambda x, y, z: y)
    labels = list(label_ds.as_numpy_iterator())
    classes = np.unique(labels)

    class_weights = compute_class_weight(
        "balanced",
        classes=classes,
        y=labels,
    )
    return class_weights

##################################################################################

df_path = "./saved_dataset2dNew"
class_weights=compute_classweights(df_path)
print("Class weights:",class_weights)

if __name__ == "__main__":
    print("Starting data splitting...")

    train_out, val_out, test_out = testsplit2d(df_path,doublecheck=True)

    print("all done")
