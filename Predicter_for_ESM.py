import os
import pickle

import esm
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics import Precision, Recall
from ESM_Embeddings_HP_search import LitClassifier, FFNClassifier


# -------------------------
# 1. GLobal settings
# -------------------------

CSV_PATH = "./Dataframes/Evalsets/DataTrainSwissPro_esm_shuffled.csv"
CATEGORY_COL = "categories"
SEQUENCE_COL = "Sequences"
MODEL_PATH = "./models/optuna_bestmodel.pt"
CACHE_PATH = "pickle/Predicer_embeddings_test_10000_domains.pkl"



NUM_CLASSES = 3
BATCH_SIZE = 16
EMB_BATCH = 4
LR = 1e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 500

DEVICE = "cpu"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 2. Dataset & embedding
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return f"seq{idx}", self.seqs[idx]


class ESMDataset:
    def __init__(self,skip_df=None):
        def map_label(cat):
            if cat == 0:
                return 1
            elif cat == 1:
                return 2
            else:
                return 0

        if skip_df == None:
            df = pd.read_csv(CSV_PATH,nrows=10000)
            df["label"] = df[CATEGORY_COL].apply(map_label)
            # print(df["label"][0:10])

            test_ids="./Dataframes/test_IDsSP2d.parquet"



            df.drop(columns=[CATEGORY_COL, "ID"], inplace=True)

            self.df = df

            print("Data loaded")
        else:
            self.skip_df = skip_df

        self.model, self.alphabet, self.batch_converter = self.esm_loader()



        self.embeddings = self._embed(self.df[SEQUENCE_COL].tolist()).cpu()

        self.labels = torch.tensor(
            self.df["label"].values, dtype=torch.long
        )

        print("Embeddings computed")

    def esm_loader(self):
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.to(DEVICE).eval()
        return model, alphabet, alphabet.get_batch_converter()

    def _embed(self, seqs):
        """
        Tokenize & run ESM in minibatches.
        The pooling below is done in one vectorized step per batch,
        """
        loader = DataLoader(
            SeqDataset(seqs),
            batch_size=EMB_BATCH,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.batch_converter,
            drop_last=False,
        )
        all_vecs = []
        total_batches = len(loader)
        for idx, (_, _, batch_tokens) in enumerate(
            loader
        ):  # loop based on chatGPT code
            if idx % 100 == 0 or idx == total_batches - 1:
                print(f"Embedding batch {idx + 1}/{total_batches}")

            # print(batch_tokens[0])

            batch_tokens = batch_tokens.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                results = self.model(
                    batch_tokens, repr_layers=[33], return_contacts=False
                )
            token_repr = results["representations"][33]

            mask = batch_tokens != self.alphabet.padding_idx

            lengths = mask.sum(dim=1).unsqueeze(-1)

            masked_repr = token_repr * mask.unsqueeze(-1)

            pooled = masked_repr.sum(dim=1) / lengths

            all_vecs.append(pooled)

        return torch.cat(all_vecs, dim=0)


# -------------------------
# 3. Predicter
# -------------------------

def predict(modelpath, loader, df):
    model = torch.load(modelpath, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE).eval()
    print("Model loaded successfully.\n")

    predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs, _ = batch
            inputs = inputs.to(DEVICE)
            output = model(inputs)
            _, preds = torch.max(output, dim=1)
            predictions.extend(preds.cpu().numpy())

    true_labels = df["label"].to_numpy()

    # Overall metrics
    accuracy = accuracy_score(true_labels, predictions)
    weighted_precision = precision_score(true_labels, predictions, average="weighted")
    confusion = confusion_matrix(true_labels, predictions, normalize="true")

    # Class‐specific precision & recall for classes 1 and 2
    prec_per_class = precision_score(
        true_labels, predictions, labels=[1, 2], average=None
    )
    rec_per_class = recall_score(
        true_labels, predictions, labels=[1, 2], average=None
    )

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted precision (all classes): {weighted_precision:.4f}")
    print(f"Precision for class 1: {prec_per_class[0]:.4f}")
    print(f"Recall    for class 1: {rec_per_class[0]:.4f}")
    print(f"Precision for class 2: {prec_per_class[1]:.4f}")
    print(f"Recall    for class 2: {rec_per_class[1]:.4f}\n")
    print(f"Confusion Matrix (rows=true, cols=pred):\n{confusion}\n")

    return predictions, true_labels



# -------------------------
# 5. Main
# -------------------------

def main():
    os.makedirs("pickle", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            df_embeddings, df = pickle.load(f)
        print("Loaded cached embeddings & labels from disk.")


    else:

        esm_data = ESMDataset()
        df_embeddings = esm_data.embeddings
        df_labels = esm_data.labels
        df= esm_data.df



        df_embeddings = TensorDataset(df_embeddings, df_labels)


        print("Dataset building complete")

        df_embeddings = DataLoader(
            df_embeddings, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
        )
        print("Data loader created")

        with open(CACHE_PATH, "wb") as f:
            pickle.dump(
                (df_embeddings, df),
                f,
            )
        print("Computed embeddings & labels, then wrote them to cache.")


    # print(df["label"][0:10])

    predictions,true_labels = predict(MODEL_PATH, df_embeddings,df)
    print("Predictions made")


    #############################
    # SECOND ROUND: collect all positive predictions
    #############################

    # positive_embeddings = []
    # positive_labels = []

    # for i in range(len(predictions)):
    #     if predictions[i] != 0:
    #         # print(df.iloc[i][SEQUENCE_COL])
    #         positive_embeddings.append(df_embeddings.dataset[i][0])
    #         positive_labels.append(true_labels[i])


    # print(f"Positive predictions: {len(positive_embeddings)}")


    # print("Positive Data loader created")
    # positive_predictions,positive_true_labels = predict(MODEL_PATH, positive_embeddings,positive_labels)

    # print("Second predictions made")

    


###############################################################################################################
if __name__ == "__main__":
    main()

