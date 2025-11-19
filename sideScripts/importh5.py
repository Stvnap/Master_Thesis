import h5py
import numpy as np
import torch

def read_embeddings_file(h5_file_path):
    """Read embeddings and labels from your H5 file format"""
    
    with h5py.File(h5_file_path, "r") as f:
        print("Available keys:", list(f.keys()))
        
        # Get embedding keys (based on your naming pattern)
        embedding_keys = sorted([
            k for k in f.keys() 
            if k.startswith("embeddings_") or 
               k.startswith("train_embeddings_") or 
               k.startswith("val_embeddings_")
        ])
        
        # Get corresponding label keys
        label_keys = sorted([
            k for k in f.keys() 
            if k.startswith("labels_") or 
               k.startswith("train_labels_") or 
               k.startswith("val_labels_")
        ])
        
        print(f"Found {len(embedding_keys)} embedding chunks")
        print(f"Found {len(label_keys)} label chunks")
        
        # Read all embeddings and labels
        all_embeddings = []
        all_labels = []
        
        for emb_key in embedding_keys:
            embeddings = f[emb_key][:]  # Read all data
            all_embeddings.append(embeddings)
            print(f"{emb_key}: shape {embeddings.shape}")
        
        for label_key in label_keys:
            labels = f[label_key][:]
            all_labels.append(labels)
            print(f"{label_key}: shape {labels.shape}")
        
        # Concatenate all chunks
        if all_embeddings:
            combined_embeddings = np.concatenate(all_embeddings, axis=0)
        if all_labels:
            combined_labels = np.concatenate(all_labels, axis=0)
    
    return combined_embeddings, combined_labels

def inspect_first_example(embeddings, labels):
    """Print detailed information about the first example"""
    
    print("\n" + "="*80)
    print("FIRST EXAMPLE INSPECTION")
    print("="*80)
    
    if len(embeddings) > 0 and len(labels) > 0:
        first_embedding = embeddings[0]
        first_label = labels[0]
        
        print(f"First embedding shape: {first_embedding.shape}")
        print(f"First embedding dtype: {first_embedding.dtype}")
        print(f"First label: {first_label}")
        print(f"First label dtype: {first_label.dtype if hasattr(first_label, 'dtype') else type(first_label)}")
        
        print(f"\nFirst embedding statistics:")
        print(f"  Min value: {np.min(first_embedding):.6f}")
        print(f"  Max value: {np.max(first_embedding):.6f}")
        print(f"  Mean value: {np.mean(first_embedding):.6f}")
        print(f"  Std value: {np.std(first_embedding):.6f}")
        
        print(f"\nFirst 10 values of first embedding:")
        print(first_embedding[:10])
        
        if len(first_embedding) > 10:
            print(f"\nLast 10 values of first embedding:")
            print(first_embedding[-10:])
        
        print(f"\nFirst embedding (full values):")
        print(first_embedding)
        
        # Convert to tensor and show tensor info
        embedding_tensor = torch.tensor(first_embedding, dtype=torch.float32)
        label_tensor = torch.tensor(first_label)
        
        print(f"\nAs PyTorch tensors:")
        print(f"Embedding tensor shape: {embedding_tensor.shape}")
        print(f"Embedding tensor dtype: {embedding_tensor.dtype}")
        print(f"Label tensor: {label_tensor}")
        print(f"Label tensor dtype: {label_tensor.dtype}")
        print(f"All labels: {labels}")
    else:
        print("No data found to inspect!")
    
    print("="*80)

# Usage
try:
    embeddings, labels = read_embeddings_file("/scratch/tmp/sapelt/Master_Thesis/temp/embeddings_classification_1000d.h5")
    print(f"\nTotal combined embeddings shape: {embeddings.shape}")
    print(f"Total combined labels shape: {labels.shape}")
    
    # Inspect the first example in detail
    inspect_first_example(embeddings, labels)
    
except FileNotFoundError:
    print("H5 file not found. Make sure the path is correct.")
except Exception as e:
    print(f"Error reading H5 file: {e}")
