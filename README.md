# DOPAMINE: DOmain Prediction and Annotation using Machine learning Inference with Neural network-based Evaluation (v.1.1)

DOPAMINE uses state-of-the-art AI and DL methods to predict Pfam domain IDs and their locations within protein sequences.

---

## Table of Contents
- [General](#general)
- [Usage](#usage)
  - [Installation](#installation)
  - [Command-Line Flags](#command-line-flags)
  - [Example Commands](#example-commands)
  - [System Requirements](#system-requirements)
  - [Checkpoints & Resume](#checkpoints--resume)
  - [Output Format](#output-format)
- [Pipeline Overview](#pipeline-overview)
- [Developer Documentation](#developer-documentation)
  - [Running Scripts with Multi-GPU Support](#running-scripts-with-multi-gpu-support)
  - [Script Descriptions](#script-descriptions)
  - [Hyperparameter Search Spaces](#hyperparameter-search-spaces)
- [Project Structure](#project-structure)

---

## General

| | |
|---|---|
| **Author** | Steven Apelt |
| **Department** | Molecular Evolution and Bioinformatics |
| **University** | University of Münster |
| **Supervisors** | Erich Bornberg-Bauer, Elias Dohmen, Carsten Kemena-Rinke |
| **Project Timeline** | March 2025 - January 2026 |
| **GitHub** | [Link](https://github.com/Stvnap/DOPAMINE) |

---

## Usage

### Installation

This package uses **UV** for dependency management. Install UV following the [official guide](https://docs.astral.sh/uv/getting-started/installation/).

Start by cloning this Repo:
```bash
git clone https://github.com/Stvnap/DOPAMINE.git
cd DOPAMINE
```
Then install dependencies:
```bash
uv sync
```
Download the models:
```bash
pip install -U huggingface_hub
huggingface-cli download Stvnap/DOPAMINE --local-dir ./models
```

To verify installation run:
```bash
uv run DOPAMINE.py --input=./Testfile.fasta --output=./Testoutput.csv
```


### Command-Line Flags

| Flag | Description | Default | Required |
|------|-------------|---------|----------|
| `--input` | Path to input FASTA file containing protein sequences | — | ✅ |
| `--output` | Path for output CSV file | — | ✅ |
| `--model` | ESM model for embedding generation | `esm2_t33_650M_UR50D` | ❌ |
| `--gpus` | Number of GPUs to allocate | `1` | ❌ |

> **⚠️ Note:** The `--model` flag is currently work in progress. Not working

### Example Commands

**Basic usage:**
```bash
uv run DOPAMINE.py --input=./Users.fasta --output=./Output.csv
```

**Multi-GPU usage:**
```bash
uv run DOPAMINE.py --input=./Users.fasta --output=./Output.csv --gpus=2
```
> **⚠️ Note:** multi GPU usage is still experimentally and under development

### System Requirements

#### Memory & Storage
- **RAM**: Allocate **≥50GB per GPU** for optimal performance. Even though processing occurs on GPU, embeddings are offloaded to CPU RAM.
- **Disk Space**: Reserve sufficient space based on input size (one proteom with ~40 000 seqs requires at least **50GB**)

#### Checkpoint System

The program implements automatic checkpointing after each major step:

✅ **Implemented:**
- Resume from available embedding files
- Checkpoint after each embedding step
- Checkpoint after classification step
- Automatic recovery from last embedded chunk during classification

⚠️ **Not Yet Implemented:**
- Domain boundary embedding recovery

**Checkpoint Storage:**
- Temporary files stored in `tempUsage/` directory
- Automatically cleared after completion

### Output Format

The program generates a comprehensive CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `Sequence_ID` | Unique identifier for each protein sequence |
| `Sequence_Length` | Total length of the protein sequence (residues) |
| `Window_Start_Pos` | Start position of the processed window |
| `Window_End_Pos` | End position of the processed window |
| `Prediction` | Predicted Pfam domain ID |
| `Domain_Start` | Predicted start position of the domain |
| `Domain_End` | Predicted end position of the domain |
| `Sequence` | Full windowed sequence (max 1000 residues) |

---

## Pipeline Overview

The pipeline operates in two distinct modes:

- **Usage Mode**: For end users performing predictions
- **Training Mode**: For developers training and evaluating models

### Current Architecture of DOPAMINE

![Current Architecture of DOPAMINE](https://i.imgur.com/AJuqoNC.png)

---

## Developer Documentation

### Running Scripts with Multi-GPU Support

All scripts except [`DOPAMINE.py`](DOPAMINE.py) require `torchrun` for multi-GPU execution. Therefore each script must be ran with its needed flags at the end accordingly:

```bash
torchrun --nproc-per-node=<NUM_GPUS> \
         --rdzv-backend=c10d \
         --rdzv-endpoint=localhost:0 \
         <SCRIPT_NAME>.py \
         [SCRIPT_FLAGS]
```

**Example:**
```bash
torchrun --nproc-per-node=4 \
         --rdzv-backend=c10d \
         --rdzv-endpoint=localhost:0 \
         ./ESM_Embeddings_HP_search.py \
         --csv_path ./input.csv \
         --HP_mode
```

> **💡 Tips:**
> - `--rdzv-backend=c10d` and `--rdzv-endpoint=localhost:0` enable distributed training on a single node
> - Each script has configurable globals at the top of the file. Change them if needed!

---

### Script Descriptions

#### [`DOPAMINE.py`](DOPAMINE.py)

**Purpose:** Main entry point for the prediction pipeline (usage mode)

**What it does:**
- Orchestrates all pipeline scripts in correct order
- Generates final output CSV with predictions

**Usage:**
```bash
uv run DOPAMINE.py --input=<FASTA> --output=<CSV> [--gpus=<N>]
```

---

#### [`Dataset_preprocess_v3.py`](Dataset_preprocess_v3.py)

**Purpose:** Preprocesses Pfam protein match data

**Requirements:**
- `Protein_match_complete.xml` from Pfam database
- CSV file with protein IDs and sequences (2 columns)
- Column names specified in script globals
- Runs normal with **UV** not `torchrun` as no GPU is used

**Output:**
- CSV with columns: `domain_start`, `domain_end`, `protein_id`, `Pfam_id`, `taxid`, `Sequence`

**Data Source:**
- UniProt database: SwissProt + TrEMBL (as of July 7, 2025)

---

#### [`TestsetCreater.py`](TestsetCreater.py)

**Purpose:** Generates stratified train/test splits for model evaluation

**Requirements:**
- Preprocessed CSV from [`Dataset_preprocess_v3.py`](Dataset_preprocess_v3.py)
- Runs normal with **UV** not `torchrun` as no GPU is used

**Output:**
- Main training set CSV
- Test evaluation set CSV

**Features:**
- Stratified split based on `split_percentage`
- Domain of Life distribution analysis
- Length distribution analysis
- Ensures all Pfam IDs present in training set

---

#### [`ESM_Embedder.py`](ESM_Embedder.py)

**Purpose:** Generates ESM-based protein sequence embeddings

**⚠️ Note:** Cannot be run directly; called by other scripts

**Embedding Types:**

The `ESMDataset` class generates different embeddings based on flag combinations:

| Flag | `True` | `False` | Required |
|------|--------|---------|----------|
| `domain_boundary_detection` | Transformer embedding | FFN classifier embedding | ✅ |
| `training` | Training embeddings | Evaluation embeddings | ✅ |
| `usage_mode` | Usage embeddings | Train/eval embeddings | Auto-assigned |

**Sequence Processing:**
- Sequences >1000 residues are windowed
- If `len(seq) % 1000 != 0`, an additional 1000-residue window from the end is added
- Ensures no sequence information is lost

**Output Format (HDF5):**

Each `.h5` file contains the following datasets:

| Dataset | Description |
|---------|-------------|
| `embedding` | ESM transformer protein embeddings |
| `label` | Target labels for training/evaluation |
| `start` | Domain start positions (evaluation only) |
| `end` | Domain end positions (evaluation only) |

**Structure:** Each dataset is indexed by `chunk_num` and process `Rank`. Custom dataloaders are in place to handle those `h5` files automatically.

---

#### [`ESM_Embeddings_HP_search.py`](ESM_Embeddings_HP_search.py)

**Purpose:** Hyperparameter tuning and training for the FFN classifier

**Requirements:**
- Main set CSV from [`TestsetCreater.py`](TestsetCreater.py)
- Embeddings from [`ESM_Embedder.py`](ESM_Embedder.py) (auto-generated if missing)

**Entry Points:**
- `main_usage()` - Usage mode
- `main()` - Training/HP optimization (requires `--HP_mode` flag)

**Configuration:**
- Global settings at top of script
- TensorBoard logs: `./logs/FINAL/{PROJECT_NAME}/`

**Usage:**
```bash
# Hyperparameter search
torchrun --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  ./ESM_Embeddings_HP_search.py --csv_path ./input.csv --HP_mode
```

---

#### [`DomainFinder.py`](DomainFinder.py)

**Purpose:** Domain boundary detection using Transformer architecture

**Requirements:**
- Preprocessed CSV from [`Dataset_preprocess_v3.py`](Dataset_preprocess_v3.py) (SwissProt only due to computational constraints)
- Embeddings from [`ESM_Embedder.py`](ESM_Embedder.py) (auto-generated if missing)

**Entry Points:**
- `main()` - Usage mode
- `main_trainer()` - Training/HP optimization (requires `--TrainerMode` flag)

**Configuration:**
- Global settings at top of script
- TensorBoard logs: `./logs/{PROJECT_NAME}/`

**Usage:**
```bash
# Hyperparameter mode
torchrun --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  ./DomainFinder.py --TrainerMode
```

---

#### [`Predicter_for_ESM.py`](Predicter_for_ESM.py)

**Purpose:** Evaluates model performance on test sets with noisy sequences

**Requirements:**
- Test set CSV from [`TestsetCreater.py`](TestsetCreater.py)
- Evaluation embeddings from [`ESM_Embedder.py`](ESM_Embedder.py) (auto-generated if missing)

**Configuration:**
- Global settings at top of script
- TensorBoard logs: `./models/FINAL/{NUM_CLASSES - 1}d_uncut_ALL/`

---

#### [`sideScripts/`](./sideScripts/)

**Purpose:** Utility scripts for development and data conversion. All are started via **UV**.

**Contains:**
- FASTA to CSV converter (used in [`DOPAMINE.py`](DOPAMINE.py))
- Development and helper scripts
- Other format converters

---

### Hyperparameter Search Spaces

The Optuna package was used for HP Search.

#### Transformer (Domain Boundary Detection)


| Hyperparameter | Type | Range/Options | Description |
|----------------|------|---------------|-------------|
| `d_model` | Categorical | 256, 512, 768, 1024 | Model embedding dimension |
| `n_heads` | Categorical | 4, 8, 16 | Number of attention heads* |
| `n_layers` | Integer | 2–4 (step: 2) | Number of transformer layers |
| `d_ff` | Calculated | 4 × `d_model` | Feed-forward network dimension |
| `max_seq_len` | Integer | 100–1000 (step: 100) | Maximum sequence length for positional encoding |
| `dropout` | Float | 0.1–0.5 | Dropout rate for regularization |
| `dropout_attn` | Float | 0.1–0.5 | Attention layer dropout rate |
| `lr` | Float | 1e-5–1e-4 (log) | Learning rate |
| `weight_decay` | Float | 1e-5–1e-2 (log) | L2 regularization weight |
| `optimizer` | Categorical | `adam`, `adamw`, `sgd` | Optimizer type |
| `activation` | Categorical | `relu`, `gelu`, `leaky_relu` | Activation function |

**\*Constraint:** `d_model` must be divisible by `n_heads` (enforced automatically)

**Weight Initialization:**
- **ReLU / LeakyReLU:** Kaiming (He) initialization
- **GELU:** Xavier (Glorot) initialization

**Optimizer Details:**
- **Adam/AdamW:** Adaptive learning rate with momentum and bias correction
- **SGD:** Stochastic Gradient Descent

---

#### Feed-Forward Network (Domain Classification)


| Hyperparameter | Type | Range/Options | Description |
|----------------|------|---------------|-------------|
| `num_neurons` | Integer | 640–1440 (step: 200) | Neurons per hidden layer |
| `num_hidden_layers` | Integer | 1–3 (step: 1) | Number of hidden layers |
| `dropout` | Float | 0.05–0.5 | Dropout rate for regularization |
| `lr` | Float | 1e-4–1e-1 (log) | Learning rate |
| `weight_decay` | Float | 1e-5–1e-1 (log) | L2 regularization weight |
| `optimizer` | Categorical | `adam`, `adamw`, `sgd`, `rmsprop`, `nadam` | Optimizer type |
| `activation` | Categorical | `relu`, `gelu`, `leaky_relu` | Activation function |

**Weight Initialization:**
- **ReLU / LeakyReLU:** Kaiming (He) initialization
- **GELU / SiLU:** Xavier (Glorot) initialization

**Optimizer Details:**
- **Adam/AdamW/NAdam:** Adaptive learning rate optimizers with momentum
- **SGD:** Stochastic Gradient Descent
- **RMSprop:** Root Mean Square Propagation

---

## Project Structure

```
.
├── DOPAMINE.py                      # Main entry point (usage mode)
├── Dataset_preprocess_v3.py         # Pfam data preprocessing
├── TestsetCreater.py                # Train/test split generation
├── ESM_Embedder.py                  # Embedding generation engine
├── DomainFinder.py                  # Transformer for boundary detection
├── ESM_Embeddings_HP_search.py      # FFN classifier training & HP tuning
├── Predicter_for_ESM.py             # Model evaluation on test sets
├── pyproject.toml                   # Project dependencies
├── uv.lock                          # Locked dependencies
├── README.md                        # Documentation (this file)
│
├── sideScripts/                     # Utility scripts & converters
├── oldScripts/                      # Archived development versions
├── backup/                          # Backup files
│
├── models/                          # Trained model checkpoints
├── logs/                            # TensorBoard training logs
├── logshp/                          # Hyperparameter search logs
├── temp/                            # Temporary processing files
│
├── Dataframes/                      # Processed data files
├── Results/                         # Evaluation results
├── Evalresults/                     # Model evaluation outputs
├── ExpFiles/                        # Experiment configurations
├── pickle/                          # Serialized Python objects
└── shells/                          # Shell scripts for batch jobs
```

---

## Citation

ESM citations:
```
@article{rives2019biological,
  author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob},
  title={Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences},
  year={2019},
  doi={10.1101/622803},
  url={https://www.biorxiv.org/content/10.1101/622803v4},
  journal={PNAS}
}
```
```
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contact

For questions or issues, please contact:
- **Author:** Steven Apelt
- **E-Mail** sapelt@uni-muenster.de
