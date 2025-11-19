#########################################


import tensorflow as tf
import polars as pl
import time

def converter(ds_path, out_path):
    """
    Converts a saved tf.data.Dataset to CSV with columns: ID, Sequences, Labels
    """
    start_time = time.time()

    # Load dataset (must be a TF-saved dataset directory, not .csv)
    ds = tf.data.Dataset.load(ds_path)

    ids = []
    seqs = []
    labels = []

    for seq, label, id_ in ds:
        try:
            seqs.append(seq.numpy().tolist())  # convert to list of ints
            labels.append(int(label.numpy()))
            ids.append(id_.numpy().decode("utf-8"))  # decode bytes to string
        except:
            # If already decoded
            seqs.append(seq)
            labels.append(label)
            ids.append(id_)

    df = pl.DataFrame({
        "ID": ids,
        "Sequences": seqs,
        "Labels": labels,
    })

    df.write_csv(out_path)
    print(f"✅ Saved CSV in {time.time() - start_time:.2f} seconds | {len(df)} entries → {out_path}")

# Example usage
converter("./trainsetSP2d", "./DataTrainALLOHenc.csv")
