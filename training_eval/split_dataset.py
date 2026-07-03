import os
import json
import random
from collections import defaultdict

def split_dataset(manifest_path, output_dir="./data", test_ratio=0.1, seed=42):
    """
    Split dataset into train+val (90%) and test (10%) sets.
    Creates mapping files to track which chunks are in each split.
    
    Args:
        manifest_path: Path to preprocessed_manifest.jsonl
        output_dir: Where to save split mapping files
        test_ratio: Fraction for test set (default 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all chunk entries from manifest
    chunks = []
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    print(f"📊 Total chunks in manifest: {len(chunks)}")
    
    # Extract chunk filenames
    chunk_files = [os.path.basename(chunk['audio_chunk_path']) for chunk in chunks]
    
    # Shuffle and split
    indices = list(range(len(chunk_files)))
    random.shuffle(indices)
    
    test_count = max(1, int(len(indices) * test_ratio))
    train_val_count = len(indices) - test_count
    
    test_indices = set(indices[:test_count])
    train_val_indices = set(indices[test_count:])
    
    # Separate chunk names
    test_chunks = [chunk_files[i] for i in sorted(test_indices)]
    train_val_chunks = [chunk_files[i] for i in sorted(train_val_indices)]
    
    # Save mapping files
    train_val_file = os.path.join(output_dir, "train_val_chunks.txt")
    test_file = os.path.join(output_dir, "test_chunks.txt")
    
    with open(train_val_file, 'w') as f:
        for chunk in train_val_chunks:
            f.write(chunk + '\n')
    
    with open(test_file, 'w') as f:
        for chunk in test_chunks:
            f.write(chunk + '\n')
    
    print(f"✅ Train+Val set: {len(train_val_chunks)} chunks ({100*(1-test_ratio):.1f}%)")
    print(f"✅ Test set: {len(test_chunks)} chunks ({100*test_ratio:.1f}%)")
    print(f"💾 Saved: {train_val_file}")
    print(f"💾 Saved: {test_file}")
    
    return train_val_chunks, test_chunks

if __name__ == "__main__":
    manifest_path = "./data/english_mini_batch/preprocessed_manifest.jsonl"
    split_dataset(manifest_path)
