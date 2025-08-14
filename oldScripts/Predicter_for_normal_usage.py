###################################################################################################################################
"""
File for normal usage

INFOS:
"""
###################################################################################################################################

import re
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.saving import load_model
from sklearn.metrics import classification_report, confusion_matrix

from FinalTrainer import BATCH_SIZE

##########################################################################################

pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_colwidth", 75)  # Show full column content


class Predicter_pipeline:
    """
    Class for the complete evaluation process and final prediction file.
    Executes the complete program in __init__()
    """

    def __init__(self, model_path, df_path,outfilepath, flank_size, step, batch_size=32):
        self.flank_size = flank_size
        self.step = step
        self.model_path = model_path
        self.df_path = df_path
        self.outfilepath = outfilepath
        self.batch_size = batch_size
        self.model = None
        self.df = None
        self.dimension_positive = 148

        ########################### FIRST PREDICTION & POSTPROCESSING ############################

        predictions, predictions_bool = self._predicting(
            model_path, df_path
        )
        print("FIRST PREDICTION DONE")

        results = self._comparer(df_path, predictions_bool)

        # print(results.head(50))

        concatenated_results = self._concatenator(results)

        concatenated_results["ConcatenatedWindow"] = concatenated_results[
            "ConcatenatedWindow"
        ].apply(self._simplify_windows)


        # print(concatenated_results.head(50))

        ###################### NEW WINDOWS AROUND POSITIVE HITS & SECOND PREDICTION ######################

        try:
            df = pd.read_csv(df_path, index_col=False)  # ,nrows=100000)
        except:
            df = self.df_path
        self.new_windows = self._newwindower(df, concatenated_results)

        self.slididing_new_windows = self._sliding_window_around_hits(
            pd.DataFrame(self.new_windows),
            window_size=self.dimension_positive,
            resultsfromcomparer=results,
            step=self.step,
            flank_size=self.flank_size,
        )

        self.seqarray_multiplied = self._multiplier(
            pd.DataFrame(self.new_windows),
            self.slididing_new_windows,
            window_size=self.dimension_positive,
            flank_size=self.flank_size,
            step=self.step,
        )

        self.seqarray_final = self._overlapChecker(self.seqarray_multiplied, 0.7)

        # print("FINAL ARRAY:", self.seqarray_final[0:30])
        predictions_new, self.predictions_bool_new = self._predicting(
            model_path,
            self.seqarray_final,
        )

        print("SECOND PREDICTION DONE")

        ########################### POSTPROCESSING OF SECOND PREDICTION ############################

        self.results = self._comparer(self.seqarray_final, self.predictions_bool_new)
        self.concatenated_results = self._concatenator(self.results)
        self.concatenated_results["ConcatenatedWindow"] = self.concatenated_results[
            "ConcatenatedWindow"
        ].apply(self._simplify_windows)

        # self.concatenated_results.to_csv(
        #     "./Evalresults/concatenated_resultsSecond.csv", index=False
        # )

        ###################### EVALUATION, HIT REFINEMENT & FINAL FILE CREATION ####################

        self.seqarray_final["Prediction"] = self.predictions_bool_new
        overlaps, ids, predictions, window_positions = self._binary_finder(
            self.seqarray_final
        )
        baseline_flags, thr_flags, true_flags, pos_IDs,pos_preds = self._hitcomparer(
            ids, predictions, overlaps, threshold=0.15
        )

        self.df2 = pd.DataFrame(
            {
                "IDS": ids,
                "WINDOW_POSITIONS": window_positions,
                "OVERLAPS": overlaps,
                "PREDICTIONS": predictions,
            }
        )

        df_filtered = self.df2[thr_flags].reset_index(drop=True)

        df_final = self._lastlistcreater(df_filtered,outfilepath)

    def _predicting(self, modelpath, Evalset):
        """
        Prediction function using the model trained beforehand.
        Based on similar functions used during training.

        Returns predictions (floats), true_labels (from the eval file), predictions_bool (int: 0 or 1)
        """

        def sequence_to_int(df_path):
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

            if not isinstance(Evalset, pd.DataFrame):
                df = pd.read_csv(df_path, index_col=False)  # ,nrows=100000)
            else:
                df = Evalset

            df = df.dropna(subset=["Sequences"])  # drop any NAs

            def encode_sequence(seq):
                return [amino_acid_to_int[amino_acid] for amino_acid in seq]

            df["Sequences"] = df["Sequences"].apply(encode_sequence)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Done encoding\nElapsed Time: {elapsed_time:.4f} seconds")
            return df

        def padder(df_int):
            """
            Pads the sequences to the target dimension with value 21 = unidentified aa
            Returns the padded df
            """
            start_time = time.time()
            sequences = df_int["Sequences"].tolist()

            padded = pad_sequences(
                sequences,
                maxlen=148,
                padding="post",
                truncating="post",
                value=21,
            )
            df_int["Sequences"] = list(padded)
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Done padding\nElapsed Time: {elapsed_time:.4f} seconds")

            return df_int

        def one_hot(padded):
            """
            Creates one hot tensors for further pipelining it into the model
            Returns a new tensor with only the one hot encoded sequences
            """
            start_time = time.time()

            # Force proper 2D NumPy array
            sequences_np = np.stack(
                padded["Sequences"].values
            )  # works if it's a Pandas DataFrame
            sequences = tf.convert_to_tensor(sequences_np, dtype=tf.int32)

            with tf.device("/CPU:0"):
                one_hot_tensor = tf.one_hot(sequences, depth=21)

            elapsed_time = time.time() - start_time
            print(f"Done one hot\nElapsed Time: {elapsed_time:.4f} seconds")
            return one_hot_tensor

        df = sequence_to_int(Evalset)
        padder_df = padder(df)
        # labled_df = labler(padder_df)

        df_onehot = one_hot(padder_df)
        X_train = df_onehot

        ########################### START OF PREDICTION ##############################

        start_time = time.time()

        print("starting prediction")
        model = load_model(
            modelpath,
            custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU},
        )
        print("model loaded")

        # prediction
        # with tf.device("/CPU:0"):
        predictions = model.predict(X_train, batch_size=BATCH_SIZE)
        print("predictions made")

        # bool predictions
        predictions_bool = []
        for value in predictions:
            if value >= 0.5:
                bool = 1
            else:
                bool = 0
            predictions_bool.append(bool)


        print("Hits:",sum(predictions_bool))


        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Done predicting\nElapsed Time: {elapsed_time:.4f} seconds")


        return predictions, predictions_bool

    def _comparer(self, df_path, predictions_bool):
        """
        Function to create a df with the predictions corresponding IDs & Window Positions
        Used for the additional info (IDs, WP) is the df for Evaluation (df_path)
        Returned is the dataframe containing all predictions with IDs & WP
        """
        start_time = time.time()
        if not isinstance(df_path, pd.DataFrame):
            df = pd.read_csv(df_path, index_col=False)  # ,nrows=100000)
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
        # if true_labels is not None:
        #     if len(true_labels) != len(flat_predictions):
        #         print(
        #             f"❌ Mismatch: predictions ({len(flat_predictions)}) vs true labels ({len(true_labels)})"
        #         )
        #     else:
        #         # print("✅ Length check passed: predictions match true labels.")
        #         combined_df = pd.DataFrame(
        #             {
        #                 "ID": flat_IDs,
        #                 "Prediction": flat_predictions,
        #                 "TrueLabel": true_labels,
        #                 "WindowPos": flat_windows,
        #             }
        #         )

        # else:
        #     combined_df = None

        # # Step: Find groups where prediction == 1
        # positive_entries = []
        # for i, labels in enumerate(master_labels):
        #     if 1 in labels:
        #         positions = [j for j, val in enumerate(labels) if val == 1]

        #         ids_with_1 = [master_IDs[i][j] for j in positions]
        #         windows_with_1 = [masterWindowPos[i][j] for j in positions]

        #         positive_entries.append(
        #             {
        #                 "GroupIndex": i,
        #                 "PositionsInGroup": positions,
        #                 "IDs": ids_with_1,
        #                 "WindowPos": windows_with_1,
        #             }
        #         )

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
                    }
                    positive_rows.append(row_data)

        positive_df = pd.DataFrame(positive_rows)
        # print(f"\n✅ Found {len(positive_df)} positive predictions (label == 1):")
        # print(positive_df.head())

        # Save the DataFrame to a CSV file
        # positive_df.to_csv("./Evalresults/positive_predictions.csv", index=False)

        duration = time.time() - start_time
        print(f"\n✅ Finished comparing windows in \n{duration:.2f} seconds.")
        # print('AFTER COMPARER',positive_df)
        return positive_df

    def _concatenator(self, results):
        """
        Creates a df with each ID having one entry (row)
        with all windows where a positive domain was predicted.
        Returns a df with IDs, concatenated Windows, Predictions,
        and the true labels found by InterproScan.
        """
        start_time = time.time()

        ids = []
        targetwindows = []
        predictions = []
        truelabels = []

        tempid = None
        temptargetwindow = ""
        tempprediction = None
        temptruelabel = None

        for idx, row in results.iterrows():
            current_id = row["ID"]
            current_piece = row["WindowPos"][row["PositionInGroup"]]

            if tempid is None or current_id != tempid:
                if tempid is not None:
                    ids.append(tempid)
                    targetwindows.append(temptargetwindow)
                    predictions.append(tempprediction)
                    truelabels.append(temptruelabel)
                temptargetwindow = current_piece
                tempprediction = row["Prediction"]
            else:
                temptargetwindow += "," + current_piece

            tempid = current_id

        # Add last group
        if tempid is not None:
            ids.append(tempid)
            targetwindows.append(temptargetwindow)
            predictions.append(tempprediction)
            truelabels.append(temptruelabel)

        duration = time.time() - start_time
        print(f"\n✅ Finished concatenating windows in \n{duration:.2f} seconds.")

        return pd.DataFrame(
            {
                "ID": ids,
                "ConcatenatedWindow": targetwindows,
                "Prediction": predictions,
                "TrueLabel": truelabels,
            }
        )

    def _simplify_windows(self, window_str):
        """
        Apply function to simplify the concatenated Windows columns
        if two windows overlap/ are next to each other
        """
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
        return merged_str

    ######## CREATION OF ALGORITH TO FEED BACK WIDNOWS AROUND A POSITVE HIT WITH SMALLER STEPSIZE #######

    def _newwindower(self, df, concatenated_results):
        """
        Inputs: df used for the first prediction & results from after simplify_windows
        Creates a df with IDs and the full length sequences for each entry
        """

        start_time = time.time()
        new_windows = []

        # Preprocess to speed up lookup
        df_grouped_seq = df.groupby("ID")["Sequences"].apply("".join).to_dict()
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

                new_windows.append(
                    {
                        "ID": entry_id,
                        "Start": start,
                        "End": end,
                        "Sequences": df_grouped_seq[entry_id],
                    }
                )

        duration = time.time() - start_time
        print(f"\n✅ Finished loading in new Sequences in \n{duration:.2f} seconds.")
        return new_windows

    ############## CREATION OF NEW WINDOWS AROUND POSITIVE HITS ######################

    def _sliding_window_around_hits(
        self,
        seqarray: pd.DataFrame,
        resultsfromcomparer: pd.DataFrame,
        window_size,
        step,
        flank_size,
    ):
        """
        Input: df from _newwindower(), results from comparer, window_size, step size, flank size
        For each sequence:
        1) Extract each hit interval from resultsfromcomparer
        2) Expand that interval by ±flank_size
        3) Slide a window of length `window_size` in increments of `step`
            across that flanked region, keeping all windows
        4) Finally, append the last window at the very end of the full sequence
        Returns a series with all new created sliding windows around a hit per hit
        """
        start_time = time.time()
        hits_by_id = resultsfromcomparer.groupby("ID")
        all_contexts = []
        self.end_window = []

        for _, row in seqarray.iterrows():
            seq_id, sequence = row["ID"], row["Sequences"]
            seqlen = len(sequence)
            self.end_window.append(seqlen)

            # 1) Collect and flank each hit
            flanked_regions = []
            if seq_id in hits_by_id.groups:
                for _, hit in hits_by_id.get_group(seq_id).iterrows():
                    raw = hit["WindowPos"][hit["PositionInGroup"]]
                    m = re.match(r"\(?(\d+)-(\d+)\)?", raw)
                    if not m:
                        continue
                    hstart, hend = map(int, m.groups())
                    # 2) apply flank
                    start = max(0, hstart - flank_size)
                    end = min(seqlen, hend + flank_size)
                    flanked_regions.append((start, end))

            # merge overlapping flanked regions
            flanked_regions.sort()
            merged = []
            for region in flanked_regions:
                if not merged or region[0] > merged[-1][1]:
                    merged.append(list(region))
                else:
                    # extend end if overlapping
                    merged[-1][1] = max(merged[-1][1], region[1])

            # 3) slide within each merged region
            windows = []
            for start, end in merged:
                region = sequence[start:end]
                for i in range(0, len(region) - window_size + 1, step):
                    windows.append(region[i : i + window_size])

            if merged:
                last_end = merged[-1][1]  # the ‘end’ coordinate of the last region
                final_start = max(last_end - window_size, 0)  # back up by window_size
                final = sequence[final_start : final_start + window_size]
                # only append if it’s not a duplicate of the last
                if not windows or windows[-1] != final:
                    windows.append(final)
            else:
                # (optional) if there were no hits at all, you could still
                # fall back to the very end of the full sequence:
                final = sequence[-window_size:]
                if not windows or windows[-1] != final:
                    windows.append(final)

            all_contexts.append(windows)

        elapsed = time.time() - start_time
        print(f"Done sliding windows around hits — {elapsed:.3f}s")
        return pd.Series(all_contexts)

    def _multiplier(self, seqarray_full, sliding, window_size, flank_size, step):
        """
        Multiplies the IDs, boundaries, categories
        corresponding to the number of additionally created windows,
        to have one sliding widnow sequence, with the corresponding IDS, boudnaries and Category.
        Additionally the window position of the windows are given in a new column.
        Returned is a final df with: Sequences, Categories, IDs, Boundaries, WindowPos.
        """
        start_time = time.time()
        sequences = []
        categories = []
        ids = []
        boundaries_all = []
        window_positions = []

        if len(sliding) != len(seqarray_full):
            raise ValueError(
                f"Mismatch: sliding has {len(sliding)} elements, "
                f"but seqarray_full has {len(seqarray_full)} rows."
            )

        for idx, nested_list in enumerate(sliding):
            row = seqarray_full.iloc[idx]
            # current_category = row["overlap"]
            current_id = row["ID"]
            start_pos = row["Start"]
            end_pos = row["End"]

            n = len(nested_list)
            for i, seq in enumerate(nested_list):
                sequences.append(seq)
                # categories.append(current_category)
                ids.append(current_id)

                # sliding‐window start
                window_start = max(start_pos - flank_size + step * i, 0)

                # if this is the last window, anchor it at end_pos + flank_size
                if i == n - 1 and not i == 0:
                    window_end = end_pos + flank_size
                    window_start = max(window_end - window_size, 0)
                else:
                    window_end = window_start + window_size

                # build and append the label every time
                label = f"{window_start}-{window_end}"
                window_positions.append(label)

            if (idx + 1) % 10000 == 0:
                print(f"Multiplication iteration: {idx + 1} / {len(seqarray_full)}")

        # now all lists are the same length, so this will succeed:
        sliding_df = pd.DataFrame(
            {
                "Sequences": sequences,
                "ID": ids,
                "WindowPos": window_positions,
            }
        )
        

        elapsed_time = time.time() - start_time
        print(f"\tDone multiplying — {elapsed_time:.4f}s")
        return sliding_df

    def _overlapChecker(self, seqarray_multiplied, threshold):
        """
        Searching for windows that overlap >= {threshold%}
        with the boundaries of the domain to anotated them as positive domains (1)
        Returns the inputted df with a extra column 'overlaps',
        which is used as the true labels for the accuracy caluclation in predicting()
        """
        start_time = time.time()

        overlaps = []

        for idx in range(len(seqarray_multiplied)):
            try:
                row = seqarray_multiplied.iloc[idx]

                # Parse window range
                window_start, window_end = map(int, row["WindowPos"].split("-"))
                window_length = window_end - window_start

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

                    if overlap >= threshold * reference_length:
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
        print(f"\t Done checking overlap\n\t Elapsed Time: {elapsed_time:.4f} seconds")
        return seqarray_multiplied

    def _binary_finder(self, seqarray_final):
        """
        This function takes the final seqarray and returns lists
        containing the overlaps, IDs, predictions, window_positions and boundaries.
        These nested lists are binary, with 1 indicating a positive prediction/true overlap and 0 indicating a negative.
        Additionally, window_positions stores the full boundary (start of first row, end of last row) for each ID group.
        """
        previous_id = None
        overlaps = []
        predictions = []
        ids = []
        window_positions = []
        boundaries = []

        temp_overlap = []
        temp_prediction = []
        temp_boundaries = []
        group_window_start = None  # stores first start in the group
        group_window_end = None  # stores latest end in the group

        for _, row in seqarray_final.iterrows():
            current_id = row["ID"]
            current_overlap = row["overlap"]
            current_prediction = row["Prediction"]
            # current_boundaries = row["Boundaries"]
            current_window_start = int(row["WindowPos"].split("-")[0])
            current_window_end = int(row["WindowPos"].split("-")[1])

            if current_id == previous_id:
                temp_overlap.append(current_overlap)
                temp_prediction.append(current_prediction)
                group_window_end = current_window_end
            else:
                if previous_id is not None:
                    overlaps.append(temp_overlap)
                    predictions.append(temp_prediction)
                    ids.append(previous_id)
                    window_positions.append(f"{group_window_start}-{group_window_end}")
                    boundaries.append(temp_boundaries)
                # start new group
                temp_overlap = [current_overlap]
                temp_prediction = [current_prediction]
                group_window_start = current_window_start
                group_window_end = current_window_end
                # temp_boundaries = [current_boundaries]

            previous_id = current_id

        # flush the last group
        if temp_overlap:
            overlaps.append(temp_overlap)
            predictions.append(temp_prediction)
            ids.append(previous_id)
            window_positions.append(f"{group_window_start}-{group_window_end}")
            # boundaries.append(temp_boundaries)

        return overlaps, ids, predictions, window_positions

    def _hitcomparer(self, ids, predictions, overlaps, threshold=0.5):
        """
        For each ID:
        - baseline_flag  = (sum(preds) >= 1)
        - thr_flag       = (sum(preds) >= threshold*len(preds)
                            AND max_consec_ones >= threshold*len(preds))
        - true_flag      = (sum(overlaps) >= 1)
        Prints overall accuracy & precision for both baseline and thr_flag models,
        and returns:
        baseline_flags, thr_flags, true_flags,
        pos_IDs, pos_preds   # for the thr_flag model
        """

        def max_consecutive_ones(seq):
            max_run = curr = 0
            for v in seq:
                # print(v)
                if v:
                    # print('here')
                    curr += 1
                    if curr > max_run:
                        max_run = curr
                else:
                    curr = 0
            return max_run

        baseline_flags = []
        thr_flags = []
        true_flags = []

        for preds, offs in zip(predictions, overlaps):
            # 1) baseline = any 1's?
            base = sum(preds) >= 1
            baseline_flags.append(base)

            # 2) threshold+consecutive
            if len(preds) == 0:
                thr = False
            else:
                needed = threshold * len(preds)
                thr = sum(preds) >= needed and max_consecutive_ones(preds) >= needed
            thr_flags.append(thr)

            # 3) ground truth
            true = sum(offs) >= 1
            true_flags.append(true)

        pos_IDs = [i for i, f in zip(ids, thr_flags) if f]
        pos_preds = [p for p, f in zip(predictions, thr_flags) if f]

        def accuracy(flags):
            return sum(f == t for f, t in zip(flags, true_flags)) / len(ids)

        def precision(flags):
            tp = sum(f and t for f, t in zip(flags, true_flags))
            fp = sum(f and not t for f, t in zip(flags, true_flags))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0


        return baseline_flags, thr_flags, true_flags, pos_IDs, pos_preds

    def _lastlistcreater(self, df_filtered,outfilepath):
        """
        Creates the last df that is represented as the final resuls,
        containign the IDs and their corresponding predicted domain boundaries.
        The resulting df is saved as a .csv
        """
        temp_start = []
        window_predicted_all = []
        for _, row in df_filtered.iterrows():
            temp_start = []
            for i in range(len(row["PREDICTIONS"])):
                if row["PREDICTIONS"][i] == 1:
                    window_start_predicted = int(
                        row["WINDOW_POSITIONS"].split("-")[0]
                    ) + (self.step * i)
                    temp_start.append(window_start_predicted)
            window_start = min(temp_start)
            window_end_predicted = window_start + self.dimension_positive
            window_predicted = f"{window_start}-{window_end_predicted}"
            window_predicted_all.append(window_predicted)

        df_final = pd.DataFrame(
            {"ID": df_filtered["IDS"], "Domain Boundaries": window_predicted_all}
        )

        print(df_final.head(50))
        print("Final count of hits:",len(df_final))
        df_final.to_csv(outfilepath, index=False)

        return df_final


#####################################################################################

if __name__ == "__main__":
    df_path = "./TESTESTESTSS.csv"
    model_path = "./models/my_modelnewlabeling.keras"
    Predicter = Predicter_pipeline(
        model_path, df_path,outfilepath="./Outtest.csv", flank_size=30, step=10, batch_size=BATCH_SIZE
    )
