import re
import time

import numpy as np
import pandas as pd
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(  # Disable GPU, for testing purposes, crashes on GPU
    [], "GPU"
)
from keras.preprocessing.sequence import pad_sequences
from keras.saving import load_model
from sklearn.metrics import classification_report, confusion_matrix

from Testrunner import BATCH_SIZE

##########################################################################################

pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_colwidth", 75)  # Show full column content
pd.set_option("display.width", 0)


class Predicter_pipeline:
    def __init__(self, model_path, df_path, batch_size=32):
        self.model_path = model_path
        self.df_path = df_path
        self.batch_size = batch_size
        self.model = None
        self.df = None
        self.dimension_positive = 148
        self.stepsize = 40

        ###########################

        predictions, true_labels, predictions_bool = self._predicting(
            model_path, df_path
        )
        print("FIRST PREDICTION DONE")

        results = self._comparer(df_path, predictions_bool)
        concatenated_results = self._concatenator(results)

        start_time = time.time()
        concatenated_results["ConcatenatedWindow"] = concatenated_results[
            "ConcatenatedWindow"
        ].apply(self._simplify_windows)

        concatenated_results.to_csv(
            "./Evalresults/concatenated_resultsFirst.csv", index=False
        )

        ######################

        df = pd.read_csv(df_path, index_col=False)  # ,nrows=200000)
        self.new_windows = self._newwindower(df, concatenated_results)

        self.slididing_new_windows = self._sliding_window_around_hits(
            pd.DataFrame(self.new_windows),
            window_size=self.dimension_positive,
            resultsfromcomparer=results,
            stepsize=111,
        )

        self.seqarray_multiplied = self._multiplier(
            pd.DataFrame(self.new_windows), self.slididing_new_windows
        )

        self.seqarray_final = self._overlapChecker(self.seqarray_multiplied)

        predictions_new, true_labels_new, predictions_bool_new = self._predicting(
            model_path,
            self.seqarray_final,
        )

        print("SECOND PREDICTION DONE")

        ###########################

        # evaluating the refeed finally

        results = self._comparer(self.seqarray_multiplied, predictions_bool_new)
        concatenated_results = self._concatenator(results)
        start_time = time.time()
        concatenated_results["ConcatenatedWindow"] = concatenated_results[
            "ConcatenatedWindow"
        ].apply(self._simplify_windows)

        concatenated_results.to_csv(
            "./Evalresults/concatenated_resultsSecond.csv", index=False
        )

    def _predicting(self, modelpath, Evalset):
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

            if not isinstance(Evalset, pd.DataFrame):
                df = pd.read_csv(
                    df_path, index_col=False
                )  # ,nrows=200000)  # Open CSV file
            else:
                df = Evalset

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
        # print(padder_df["Sequences"].apply(len).max())
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
        # print(predictions)
        predictions_bool = []
        for value in predictions:
            if value >= 0.5:
                bool = 1
            else:
                bool = 0
            predictions_bool.append(bool)

        # print(predictions_bool[0:10])
        true_labels = y_train

        # print(len(true_labels))
        # print(len(predictions_bool))

        def accuracy_score(y_true, y_pred):
            return (y_true == y_pred).mean()

        accuracy = accuracy_score(true_labels, predictions_bool)

        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Predictions: {predictions}")
        # print(f"True Labels: {true_labels}")
        np.set_printoptions(suppress=True)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Done predicting\nElapsed Time: {elapsed_time:.4f} seconds")
        print(confusion_matrix(true_labels, predictions_bool))
        print(
            np.round(
                confusion_matrix(true_labels, predictions_bool, normalize="true"), 5
            )
        )
        print(classification_report(true_labels, predictions_bool))

        return predictions, true_labels, predictions_bool

    def _comparer(self, df_path, predictions_bool):
        start_time = time.time()
        if not isinstance(df_path, pd.DataFrame):
            df = pd.read_csv(df_path, index_col=False)  # ,nrows=200000)
        else:
            df = df_path
        df = df.dropna(subset=["Sequences"])
        try:
            true_labels = df["overlap"].tolist()
        except:
            true_labels = None

        master_labels = []
        master_IDs = []
        masterWindowPos = []

        current_labels = []
        current_IDs = []
        currentWindowPos = []

        print("Building groups...")
        for idx, row in df.iterrows():
            if row["WindowPos"] == "0-148":
                if current_labels and current_IDs and currentWindowPos:
                    master_labels.append(current_labels)
                    master_IDs.append(current_IDs)
                    masterWindowPos.append(currentWindowPos)
                current_labels = []
                current_IDs = []
                currentWindowPos = []

            current_IDs.append(row["ID"])
            current_labels.append(predictions_bool[idx])
            currentWindowPos.append(row["WindowPos"])

        # Append last group
        if current_labels and current_IDs and currentWindowPos:
            master_labels.append(current_labels)
            master_IDs.append(current_IDs)
            masterWindowPos.append(currentWindowPos)

        print("✅ Grouping complete.")
        # print("Sample group:\n")
        # print("Labels:", master_labels[0])
        # print("IDs:", master_IDs[0])
        # print("Window Positions:", masterWindowPos[0])

        # Debug step: Flatten everything for optional DataFrame comparison
        flat_IDs = []
        flat_predictions = []
        flat_windows = []
        for group_labels, group_ids, group_windows in zip(
            master_labels, master_IDs, masterWindowPos
        ):
            flat_IDs.extend(group_ids)
            flat_predictions.extend(group_labels)
            flat_windows.extend(group_windows)

        # Optional: Compare to true labels (if provided)
        if true_labels is not None:
            if len(true_labels) != len(flat_predictions):
                print(
                    f"❌ Mismatch: predictions ({len(flat_predictions)}) vs true labels ({len(true_labels)})"
                )
            else:
                # print("✅ Length check passed: predictions match true labels.")
                combined_df = pd.DataFrame(
                    {
                        "ID": flat_IDs,
                        "Prediction": flat_predictions,
                        "TrueLabel": true_labels,
                        "WindowPos": flat_windows,
                    }
                )
                # print("\n🔍 Sample comparison:")
                # print(combined_df.head(100))
        else:
            combined_df = None

        # Step: Find groups where prediction == 1
        positive_entries = []
        for i, labels in enumerate(master_labels):
            if 1 in labels:
                positions = [j for j, val in enumerate(labels) if val == 1]

                # Debug: Print out prediction locations
                # print(f"\nGroup {i} contains 1 at positions {positions}")
                # print("Predictions:", labels)

                ids_with_1 = [master_IDs[i][j] for j in positions]
                windows_with_1 = [masterWindowPos[i][j] for j in positions]

                # print("Matched IDs:", ids_with_1)
                # print("Matched Windows:", windows_with_1)

                positive_entries.append(
                    {
                        "GroupIndex": i,
                        "PositionsInGroup": positions,
                        "IDs": ids_with_1,
                        "WindowPos": windows_with_1,
                    }
                )

        # print("\n✅ Completed filtering. Total positive groups:", len(positive_entries))

        positive_rows = []

        for i, labels in enumerate(master_labels):
            for j, label in enumerate(labels):
                if label == 1:
                    row_data = {
                        "GroupIndex": i,
                        "PositionInGroup": j,
                        "ID": master_IDs[i][j],
                        "WindowPos": masterWindowPos[i],
                        "Prediction": 1,
                        "TrueLabel": true_labels[i],
                    }
                    positive_rows.append(row_data)

        positive_df = pd.DataFrame(positive_rows)
        # print(f"\n✅ Found {len(positive_df)} positive predictions (label == 1):")
        # print(positive_df.head())

        # Save the DataFrame to a CSV file
        positive_df.to_csv("./Evalresults/positive_predictions.csv", index=False)
        duration = time.time() - start_time
        # print(f"\n✅ Finished comparing windows in \n{duration:.2f} seconds.")
        # print('AFTER COMPARER',positive_df)
        return positive_df

    def _concatenator(self, results):
        start_time = time.time()

        ids = []
        targetwindows = []

        tempid = None
        temptargetwindow = ""

        for idx, row in results.iterrows():
            current_id = row["ID"]
            current_piece = row["WindowPos"][row["PositionInGroup"]]

            if tempid is None or current_id != tempid:
                if tempid is not None:
                    ids.append(tempid)
                    targetwindows.append(temptargetwindow)
                temptargetwindow = current_piece
            else:
                temptargetwindow += ","
                temptargetwindow += current_piece

            tempid = current_id

        # add last result
        if tempid is not None:
            ids.append(tempid)
            targetwindows.append(temptargetwindow)

        duration = time.time() - start_time
        # print(f"\n✅ Finished concatinating windows in \n{duration:.2f} seconds.")
        return pd.DataFrame({"ID": ids, "ConcatenatedWindow": targetwindows})

    def _simplify_windows(self, window_str):
        ranges = window_str.split(",")

        window_ranges = []
        for r in ranges:
            start, end = map(int, r.split("-"))
            window_ranges.append((start, end))

        window_ranges.sort()

        merged_ranges = []
        current_start, current_end = window_ranges[0]

        for start, end in window_ranges[1:]:
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged_ranges.append((current_start, current_end))
                current_start, current_end = start, end

        merged_ranges.append((current_start, current_end))

        merged_str = ",".join([f"{start}-{end}" for start, end in merged_ranges])
        # print(f"merged_str: {merged_str}")
        return merged_str

    ######## CREATION OF ALGORITH TO FEED BACK WIDNOWS AROUND A POSITVE HIT WITH SMALLER STEPSIZE #######
    def _newwindower(self, df, concatenated_results):
        import time

        start_time = time.time()
        new_windows = []

        # Preprocess to speed up lookup
        df_grouped_seq = df.groupby("ID")["Sequences"].apply("".join).to_dict()
        df_grouped_overlap = df.groupby("ID")["overlap"].first().to_dict()
        df_grouped_boundaries = df.groupby("ID")["Boundaries"].first().to_dict()
        window_dict = concatenated_results.set_index("ID")[
            "ConcatenatedWindow"
        ].to_dict()

        total_ids = len(window_dict)

        for idx, (entry_id, window_str) in enumerate(window_dict.items(), start=1):
            if idx % 10000 == 0:
                print(f"Processing {idx}/{total_ids}")

            if entry_id not in df_grouped_seq:
                continue

            windows = window_str.split(",")

            for window in windows:
                try:
                    start, end = map(int, window.split("-"))
                except ValueError:
                    print(f"Invalid window format for ID {entry_id}: {window}")
                    continue

                if start == 0:
                    new_windows.append(
                        {
                            "ID": entry_id,
                            "Start": start,
                            "End": end,
                            "Sequences": df_grouped_seq[entry_id],
                            "overlap": df_grouped_overlap.get(entry_id, None),
                            "Boundaries": df_grouped_boundaries.get(entry_id, None),
                        }
                    )

        end_time = time.time()
        ("NEWWINDOWS", new_windows[0:10])
        return new_windows

    ############## CREATION OF NEW WINDOWS AROUND POSITIVE HITS ######################
    def _sliding_window_around_hits(
        self, seqarray, resultsfromcomparer, window_size=148, stepsize=111
    ):
        start_time = time.time()
        step = window_size - stepsize
        context_windows = []
        self.end_window = []

        # Group hits by ID for fast lookup
        hits_by_id = resultsfromcomparer.groupby("ID")

        for idx, row in seqarray.iterrows():
            seq_id = row["ID"]
            sequence = row["Sequences"]
            seqlen = len(sequence)
            self.end_window.append(seqlen)

            region_windows_all = []

            # Get all hits for this sequence
            if seq_id in hits_by_id.groups:
                hits = hits_by_id.get_group(seq_id)

                for _, hit_row in hits.iterrows():
                    try:
                        window_str = hit_row["WindowPos"][hit_row["PositionInGroup"]]
                        match = re.match(r"\(?(\d+)-(\d+)\)?", window_str)
                        if not match:
                            continue
                        hit_start, hit_end = map(int, match.groups())
                    except (IndexError, KeyError, TypeError):
                        continue

                    context_start = max(0, hit_start - 74)
                    context_end = min(seqlen, hit_end + 74)
                    region = sequence[context_start:context_end]
                    region_len = len(region)

                    # Generate overlapping windows
                    for i in range(0, region_len - window_size + 1, step):
                        window_seq = region[i : i + window_size]
                        region_windows_all.append(window_seq)

                    last_window = region[-window_size:]
                    if not region_windows_all or region_windows_all[-1] != last_window:
                        region_windows_all.append(last_window)

            context_windows.append(region_windows_all)

        elapsed_time = time.time() - start_time
        print(
            f"\tDone sliding windows around hits\n\tElapsed Time: {elapsed_time:.4f} seconds"
        )
        print(
            f"len(seqarray): {len(seqarray)}, len(context_windows): {len(context_windows)}"
        )

        return pd.Series(context_windows)

    def _multiplier(self, seqarray_full, sliding):
        start_time = time.time()

        # Predefine lists for better performance
        sequences = []
        categories = []
        ids = []
        boundaries_all = []
        window_positions = []

        category_index = 0

        for nested_list in sliding:
            current_row = seqarray_full.iloc[category_index]
            current_category = current_row["overlap"]
            current_id = current_row["ID"]
            current_boundary = current_row["Boundaries"]

            len_nested = len(nested_list)

            for i in range(len_nested):
                seq = nested_list[i]
                sequences.append(seq)
                categories.append(current_category)
                ids.append(current_id)
                boundaries_all.append(current_boundary)

                # Calculate WindowPos as string
                if i == len_nested - 1 and len_nested > 1:
                    # last window gets special end_window value
                    last_window_start = (
                        self.end_window[category_index] - self.dimension_positive
                    )
                    last_window_end = self.end_window[category_index]
                    window_pos = f"{last_window_start}-{last_window_end}"
                else:
                    start = i * self.dimension_positive - (
                        self.stepsize * i if i > 0 else 0
                    )
                    end = (i + 1) * self.dimension_positive - (
                        self.stepsize * i if i > 0 else 0
                    )
                    window_pos = f"{start}-{end}"

                window_positions.append(window_pos)

            category_index += 1

            if category_index % 10000 == 0:
                print(
                    "Multiplication iteration:", category_index, "/", len(seqarray_full)
                )

        # Convert once to DataFrame at the end
        sliding_df = pd.DataFrame(
            {
                "Sequences": sequences,
                "overlap": categories,
                "ID": ids,
                "Boundaries": boundaries_all,
                "WindowPos": window_positions,
            }
        )

        elapsed_time = time.time() - start_time
        # print(sliding_df.head(100))
        print(f"\t Done multiplying\n\t Elapsed Time: {elapsed_time:.4f} seconds")
        return sliding_df

    def _overlapChecker(self, seqarray_multiplied):
        """

        Searching for windows that overlap >= 50% with the boundaries of the domain to anotated them as positive domains (1)


        """
        start_time = time.time()

        overlaps = []

        for idx in range(len(seqarray_multiplied)):
            try:
                row = seqarray_multiplied.iloc[idx]

                # Parse window range
                window_start, window_end = map(int, row["WindowPos"].split("-"))
                window_length = window_end - window_start

                # Skip if category != 0 (we only assign overlap if original category was 0)
                # if row["categories"] != 0:
                #     overlaps.append(0)
                #     continue

                # Parse possibly multiple boundaries
                boundary_ranges = row["Boundaries"].split(",")
                overlap_found = False

                for br in boundary_ranges:
                    boundary_start, boundary_end = map(int, br.split("-"))
                    boundary_length = boundary_end - boundary_start

                    # Compute overlap
                    overlap = min(window_end, boundary_end) - max(
                        window_start, boundary_start
                    )
                    overlap = max(overlap, 0)

                    reference_length = min(window_length, boundary_length)

                    if overlap >= 0.7 * reference_length:
                        overlap_found = True
                        break

                overlaps.append(1 if overlap_found else 0)

            except Exception:
                overlaps.append(0)

            if idx % 100000 == 0:
                print(f"Overlap check iteration: {idx}/{len(seqarray_multiplied)}")

        # Assign final overlap column
        seqarray_multiplied["overlap"] = overlaps
        elapsed_time = time.time() - start_time
        # print(seqarray_multiplied.head(100))
        print(f"\t Done checking overlap\n\t Elapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_multiplied


#####################################################################################

if __name__ == "__main__":
    df_path = "./DataEvalSwiss70%.csv"
    model_path = "./models/my_modelnewlabeling.keras"
    Predicter = Predicter_pipeline(model_path, df_path, batch_size=BATCH_SIZE)
