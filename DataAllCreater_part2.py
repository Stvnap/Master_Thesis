###################################################################################################################################
"""
File for creating the whole Training Dataset in tf.Dataset format to be used for training

INFOS:
Use enough RAM for this script

"""
###################################################################################################################################

import random
from collections import Counter, defaultdict

import numpy as np
import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split
###################################################################################################################################


def splitter3_stratified_grouped(
    ds_path, train_frac=0.6, val_frac=0.2, test_frac=0.2, shuffle_buffer=10000, seed=42
):
    """
    Loads a saved tf.data.Dataset of (seq, label, id), performs a grouped,
    approximately stratified 60/20/20 split (IDs kept within the same split).
    Writes train/val/test ID lists to Parquet, drops 'id' from each split,
    and saves the TF datasets.
    """

    # Step 1: Load dataset once to collect (id, label) pairs
    ds = tf.data.Dataset.load(ds_path)

    label_by_id = {}
    id_counts = Counter()

    for _, label, id_ in ds:
        id_val = (
            id_.numpy().decode("utf-8")
            if isinstance(id_.numpy(), bytes)
            else str(id_.numpy())
        )
        label_val = int(label.numpy())
        if id_val not in label_by_id:
            label_by_id[id_val] = label_val
        id_counts[id_val] += 1

    # Step 2: Group IDs by label
    label_to_ids = defaultdict(list)
    for id_val, label in label_by_id.items():
        label_to_ids[label].append(id_val)

    # Step 3: Stratified split of IDs per label
    random.seed(seed)
    train_ids, val_ids, test_ids = set(), set(), set()

    for label, id_list in label_to_ids.items():
        random.shuffle(id_list)
        n = len(id_list)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        train_ids.update(id_list[:n_train])
        val_ids.update(id_list[n_train : n_train + n_val])
        test_ids.update(id_list[n_train + n_val :])

    # Step 4: Map IDs to split
    id_to_split = {}
    for id_ in train_ids:
        id_to_split[id_] = "train"
    for id_ in val_ids:
        id_to_split[id_] = "val"
    for id_ in test_ids:
        id_to_split[id_] = "test"

    # Step 5: Tag each sample with its split
    def assign_split(seq, label, id_):
        id_str = tf.numpy_function(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x),
            [id_],
            tf.string,
        )
        split = tf.case(
            [
                (
                    tf.sets.set_intersection([id_str], [tf.constant(list(train_ids))])[
                        0
                    ]
                    != "",
                    lambda: "train",
                ),
                (
                    tf.sets.set_intersection([id_str], [tf.constant(list(val_ids))])[0]
                    != "",
                    lambda: "val",
                ),
            ],
            default=lambda: "test",
        )
        return split, seq, label, id_

    # Instead of using `tf.sets`, we can pass the mapping as a hash table in Python:
    def assign_split_py(seq, label, id_):
        id_val = (
            id_.numpy().decode("utf-8")
            if isinstance(id_.numpy(), bytes)
            else str(id_.numpy())
        )
        split = id_to_split.get(id_val, "test")
        return split.encode("utf-8"), seq, label, id_

    tagged = ds.map(
        lambda seq, label, id_: tf.py_function(
            assign_split_py,
            [seq, label, id_],
            [
                tf.string,
                tf.TensorSpec(shape=seq.shape, dtype=seq.dtype),
                tf.int32,
                tf.string,
            ],
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Step 6: Split + save
    def make_split(split_tag):
        return (
            tagged.filter(
                lambda tag, *_: tf.equal(tag, tf.constant(split_tag.encode("utf-8")))
            )
            .map(
                lambda tag, seq, lab, id_: (seq, lab, id_),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .shuffle(shuffle_buffer, seed=seed)
        )

    train_ds = make_split("train")
    val_ds = make_split("val")
    test_ds = make_split("test")

    # Step 7: Write ID lists
    def extract_ids(split_ds, out_path):
        ids = list(
            split_ds.map(lambda seq, lab, id_: id_).unbatch().as_numpy_iterator()
        )
        pl.DataFrame({"ID": ids}).write_parquet(out_path)

    extract_ids(train_ds, "train_IDs.parquet")
    extract_ids(val_ds, "val_IDs.parquet")
    extract_ids(test_ds, "test_IDs.parquet")

    # Step 8: Drop 'id' and save each dataset
    def drop_id_and_save(split_ds, folder_name):
        ds_nokey = split_ds.map(
            lambda seq, lab, id_: (seq, lab), num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_nokey.save(folder_name)
        return ds_nokey

    train_out = drop_id_and_save(train_ds, "trainsetALL")
    val_out = drop_id_and_save(val_ds, "valsetALL")
    test_out = drop_id_and_save(test_ds, "testsetALL")

    return train_out, val_out, test_out



def testsplit(df_path):
    df = tf.data.Dataset.load(df_path)

    print("loaded")

    sequences = []
    labels = []
    ids=[]
    for seq, label,id in df:
        sequences.append(seq.numpy())
        labels.append(label.numpy())
        ids.append(id.numpy())


    print(sequences[1],labels[1],ids[1])



    seq_train, seq_temp, label_train, label_temp, id_train, id_temp = train_test_split(
        sequences, labels, ids,
        stratify=labels,
        test_size=0.2,
        random_state=42
    )

    seq_val, seq_test, label_val, label_test, id_val, id_test = train_test_split(
        seq_temp, label_temp, id_temp,
        stratify=label_temp,
        test_size=0.5,
        random_state=42
    )
    print("splitted")

    train_ds = tf.data.Dataset.from_tensor_slices((seq_train, label_train))
    val_ds = tf.data.Dataset.from_tensor_slices((seq_val, label_val))
    test_ds  = tf.data.Dataset.from_tensor_slices((seq_test,  label_test))


    print(id_train[0:50])

    def extract_ids(split_ds, out_path):
        pl.DataFrame({"ID": split_ds}).write_parquet(out_path)

    id_train=extract_ids(id_train, "train_IDs.parquet")
    extract_ids(id_val, "val_IDs.parquet")
    extract_ids(id_test, "test_IDs.parquet")




    return train_ds,val_ds,test_ds





def testsplit2(df_path):
    """
    Final preprocessing function to split the tf.dataset into train,val,testset
    Spliting is done via a manual stratified split by label to ensure equal
    distribution of target domains in all sets
    For each set a parquet file is saved with the IDs included in the sets
    Finally each set is saved in a tf.datasets for the training
    """
    # load in
    df = tf.data.Dataset.load(df_path)
    print("loaded")

    def splitcalculation(ds):
        """
        Calculates the ratio of positives (1) / all (0+1) in the whole dataset
        """
        total = len(ds)
        positives = 0
        try:
            for _, label, _ in ds.as_numpy_iterator():
                positives += int(label)
        except:
            for _, label, _ in ds:
                positives += int(label)
        print("Positives",positives,"Total",total)
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
        testset = []
        trainset = []

        # total_positives = sum(1 for _, label, _ in dataset if int(label) == 1)

        max_testset_size = int(size * len(dataset))
        print(max_testset_size)
        max_test_positives = int(frac * max_testset_size)
        print(max_test_positives)


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

            if label_val == 1 and current_test_positives < max_test_positives: #and current_test_total < max_testset_size:
                testset.append(sample)
                current_test_positives += 1
                current_test_total += 1
            elif current_test_total < max_testset_size:
                testset.append(sample)
                current_test_total += 1
            else:
                trainset.append(sample)

        print(f"✅ Split done: Train={len(trainset)}, Test={len(testset)}, Positives in test={current_test_positives}")
        return trainset, testset




    def save_ids_as_parquet(dataset_list, out_path):
        """
        Save function to save the IDs found in the given set to parquet files
        """
        if not dataset_list:
            print(f"⚠️ ID list for {out_path} is empty.")
            df = pl.DataFrame({"ID": []})
        else:
            ids = [id_ for _, _, id_ in dataset_list]
            df = pl.DataFrame({"ID": ids})
        
        df.write_parquet(out_path)
        print(f"✅ Saved {len(df)} IDs to {out_path}")


    def convert_and_save(dataset_list, save_path):
        """
        Save function for creating the final tf.dataset
        """
        if not dataset_list:
            print(f"⚠️ Cannot save '{save_path}' — dataset is empty.")
            raise KeyError ("dataset empty")

        seqs, labels, ids = zip(*dataset_list)
        ds = tf.data.Dataset.from_tensor_slices((
            list(seqs),
            list(labels),                   # id is dropped, no need for training
        ))
        ds.save(save_path)
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
    frac_test = splitcalculation(testset)   # positive-rate within your testset
    valset, testset = splitter(testset, frac_test, size=0.5)
    print("Test/Val split done")

    # Save IDs as Parquet
    save_ids_as_parquet(trainset, "train_IDs.parquet")
    save_ids_as_parquet(valset,   "val_IDs.parquet")
    save_ids_as_parquet(testset,  "test_IDs.parquet")

    # Save tf.datasets
    convert_and_save(trainset,'./trainset')
    convert_and_save(valset,'./valset')
    convert_and_save(testset,'./testset')


    positive_label_percentage(trainset)
    positive_label_percentage(valset)
    positive_label_percentage(testset)


    # print(trainset, valset, testset)
    return trainset, valset, testset


##################################################################################

if __name__ == "__main__":
    print("Starting data splitting...")
    df_path = "./saved_dataset"

    train_out, val_out, test_out = testsplit2(df_path)

    print("all done")

