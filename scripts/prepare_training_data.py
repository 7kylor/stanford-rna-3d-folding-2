#!/usr/bin/env python3
"""
Phase 2: Prepare Training Data for ML Distance Prediction

=============================================================================
WHY THIS SCRIPT EXISTS
=============================================================================
Template-based modeling achieved only TM=0.036 (target was 0.30) because:
1. 27/28 test sequences are NOVEL - no close templates exist
2. Template search can only use existing structures, not predict new ones

ML distance prediction learns the RELATIONSHIP between sequence and 3D distances,
allowing it to predict structures for novel sequences that have no templates.

=============================================================================
WHAT THIS SCRIPT DOES
=============================================================================
1. Loads 7.8M coordinate rows from train_labels.csv (332 MB)
2. Groups coordinates by target_id (5,716 unique structures)
3. Computes pairwise C1'-C1' distance matrices for each structure
4. Generates sequence features (one-hot encoding + positional)
5. Creates train/validation split (80/20 = 4,573/1,143 structures)
6. Saves in efficient format for PyTorch DataLoader

=============================================================================
HOW IT WORKS
=============================================================================
Step 1: Load coordinates
    - Parse train_labels.csv row by row
    - Group by target_id (e.g., "157D_1" -> target "157D", residue 1)
    - Extract (x, y, z) coordinates for each residue

Step 2: Compute distance matrices
    - For each structure with L residues, compute L×L distance matrix
    - D[i,j] = ||coord[i] - coord[j]||_2 (Euclidean distance)
    - These are the GROUND TRUTH targets for ML training

Step 3: Generate features
    - One-hot encode sequence (4 channels: A, C, G, U)
    - Add positional encoding (sinusoidal or learned)
    - Optionally include MSA covariation features

Step 4: Create splits
    - 80% train (4,573 structures for gradient updates)
    - 20% validation (1,143 structures for hyperparameter tuning)
    - Stratify by sequence length to ensure balanced representation

=============================================================================
OUTPUT FORMAT
=============================================================================
Saves to output/training_data/:
    - train_features.pkl: List of (seq_onehot, length) tuples
    - train_distances.pkl: List of (L, L) distance matrices
    - train_targets.pkl: List of target_ids (for debugging)
    - val_features.pkl, val_distances.pkl, val_targets.pkl
    - metadata.json: Statistics and split information
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def log_change(what: str, why: str, how: str) -> None:
    """Log a change in detailed format for documentation."""
    logger.info("=" * 70)
    logger.info(f"WHAT: {what}")
    logger.info(f"WHY:  {why}")
    logger.info(f"HOW:  {how}")
    logger.info("=" * 70)


def load_train_labels(csv_path: str) -> pd.DataFrame:
    """
    Load training labels from CSV.
    
    WHAT: Load 7.8M coordinate rows from train_labels.csv
    WHY:  This is our ground truth - actual 3D coordinates of RNA structures
    HOW:  Chunked reading to handle 332 MB file without memory issues
    """
    log_change(
        what="Loading training labels CSV",
        why="Source of 7.8M ground truth coordinates for 5,716 RNA structures",
        how="Chunked pandas read_csv with 500K rows per chunk"
    )
    
    chunks = []
    total_rows = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=500000):
        chunks.append(chunk)
        total_rows += len(chunk)
        logger.info(f"  Loaded {total_rows:,} rows...")
    
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Total: {len(df):,} rows loaded")
    
    return df


def parse_target_and_resid(id_str: str) -> Tuple[str, int]:
    """
    Parse ID like '157D_1' into (target_id='157D', resid=1).
    
    WHAT: Extract target and residue number from compound ID
    WHY:  Need to group coordinates by structure and order by residue
    HOW:  Split on last underscore, convert residue part to integer
    """
    parts = id_str.rsplit('_', 1)
    return parts[0], int(parts[1])


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix.
    
    WHAT: Convert 3D coordinates to L×L distance matrix
    WHY:  Distances are more stable prediction targets than raw coordinates
          (invariant to rotation/translation)
    HOW:  D[i,j] = sqrt(sum((coords[i] - coords[j])^2))
    
    Args:
        coords: (L, 3) array of 3D coordinates
        
    Returns:
        (L, L) symmetric distance matrix
    """
    # Using broadcasting: (L, 1, 3) - (1, L, 3) = (L, L, 3)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    return distances.astype(np.float32)


def sequence_to_onehot(sequence: str) -> np.ndarray:
    """
    Convert RNA sequence to one-hot encoding.
    
    WHAT: Encode sequence as (L, 4) binary matrix
    WHY:  Neural networks need numeric input, one-hot preserves base identity
    HOW:  Map A=0, C=1, G=2, U=3 (N=average of all)
    
    Args:
        sequence: RNA sequence string (e.g., "ACGU")
        
    Returns:
        (L, 4) one-hot encoded array
    """
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    L = len(sequence)
    onehot = np.zeros((L, 4), dtype=np.float32)
    
    for i, base in enumerate(sequence):
        if base.upper() in base_to_idx:
            onehot[i, base_to_idx[base.upper()]] = 1.0
        else:
            # Unknown base (N): uniform distribution
            onehot[i] = 0.25
    
    return onehot


def add_positional_encoding(onehot: np.ndarray, max_len: int = 2000) -> np.ndarray:
    """
    Add sinusoidal positional encoding.
    
    WHAT: Append positional information to sequence features
    WHY:  Transformers/attention need position info; distances depend on 
          sequence separation (nearby residues more likely to be close in 3D)
    HOW:  sin/cos at different frequencies as in "Attention is All You Need"
    
    Args:
        onehot: (L, 4) one-hot sequence
        max_len: Maximum sequence length supported
        
    Returns:
        (L, 4 + d_pos) features with positional encoding appended
    """
    L = onehot.shape[0]
    d_pos = 16  # Dimensions for positional encoding
    
    positions = np.arange(L)[:, np.newaxis]  # (L, 1)
    dims = np.arange(d_pos)[np.newaxis, :]    # (1, d_pos)
    
    # Different frequency for each dimension
    freqs = 1.0 / (10000 ** (dims / d_pos))
    
    # Interleave sin and cos
    pos_enc = np.zeros((L, d_pos), dtype=np.float32)
    pos_enc[:, 0::2] = np.sin(positions * freqs[:, 0::2])
    pos_enc[:, 1::2] = np.cos(positions * freqs[:, 1::2])
    
    return np.concatenate([onehot, pos_enc], axis=1)


def extract_sequence_from_coords(group: pd.DataFrame) -> str:
    """
    Extract sequence from coordinate DataFrame.
    
    WHAT: Get RNA sequence from residue names column
    WHY:  Need sequence for one-hot encoding (input to ML model)
    HOW:  Map each resname to single-letter code (A, C, G, U)
    """
    base_map = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U', 'T': 'U'}
    sequence = []
    for resname in group['resname']:
        base = base_map.get(str(resname).upper(), 'N')
        sequence.append(base)
    return ''.join(sequence)


def build_structures(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Group coordinates by target and build structure dictionary.
    
    WHAT: Transform raw rows into per-structure coordinate arrays
    WHY:  ML training needs complete structures, not individual atoms
    HOW:  Group by target_id, sort by residue number, extract coords
    
    Returns:
        Dict[target_id, {
            'sequence': str,
            'coords': np.ndarray (L, 3),
            'length': int
        }]
    """
    log_change(
        what="Building structure dictionary from raw coordinates",
        why="Group 7.8M rows into 5,716 complete structures",
        how="GroupBy target_id, sort by residue, handle multiple copies"
    )
    
    structures = {}
    
    # Add target_id column
    df['target_id'] = df['ID'].apply(lambda x: x.rsplit('_', 1)[0])
    
    target_groups = df.groupby('target_id')
    logger.info(f"  Processing {len(target_groups)} unique targets...")
    
    for target_id, group in tqdm(target_groups, desc="Building structures"):
        # Sort by residue number
        group = group.copy()
        group['resid_num'] = group['ID'].apply(lambda x: int(x.rsplit('_', 1)[1]))
        group = group.sort_values('resid_num')
        
        # Handle multiple copies - take first
        if 'copy' in group.columns:
            first_copy = group['copy'].min()
            group = group[group['copy'] == first_copy]
        
        # Extract sequence and coordinates
        sequence = extract_sequence_from_coords(group)
        coords = group[['x_1', 'y_1', 'z_1']].values.astype(np.float32)
        
        # Center coordinates (important for ML training stability)
        coords = coords - coords.mean(axis=0)
        
        structures[target_id] = {
            'sequence': sequence,
            'coords': coords,
            'length': len(sequence),
        }
    
    return structures


def prepare_ml_dataset(
    structures: Dict[str, Dict],
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    """
    Prepare ML training and validation datasets.
    
    WHAT: Convert structures to feature/target pairs for training
    WHY:  ML models need (input, target) pairs in consistent format
    HOW:  One-hot + positional encoding for input, distance matrix for target
    
    Args:
        structures: Dict from build_structures()
        val_ratio: Fraction for validation (0.2 = 20%)
        random_seed: For reproducible splits
        
    Returns:
        (train_data, val_data, metadata)
    """
    log_change(
        what="Preparing ML dataset with train/val split",
        why=f"Need {int((1-val_ratio)*100)}% for training, {int(val_ratio*100)}% for validation",
        how="Stratified by length, generate features and distance matrices"
    )
    
    np.random.seed(random_seed)
    
    # Sort targets by length for stratified split
    sorted_targets = sorted(structures.items(), key=lambda x: x[1]['length'])
    
    # Stratified split: take every 5th structure for validation
    n_val = int(len(sorted_targets) * val_ratio)
    val_indices = set(range(0, len(sorted_targets), int(1/val_ratio)))
    
    train_data = {'features': [], 'distances': [], 'targets': [], 'lengths': []}
    val_data = {'features': [], 'distances': [], 'targets': [], 'lengths': []}
    
    logger.info("  Generating features and distance matrices...")
    
    for idx, (target_id, struct) in enumerate(tqdm(sorted_targets, desc="Preparing data")):
        # Generate features
        onehot = sequence_to_onehot(struct['sequence'])
        features = add_positional_encoding(onehot)
        
        # Generate distance matrix
        distances = compute_distance_matrix(struct['coords'])
        
        # Add to appropriate split
        data = val_data if idx in val_indices else train_data
        data['features'].append(features)
        data['distances'].append(distances)
        data['targets'].append(target_id)
        data['lengths'].append(struct['length'])
    
    # Compute statistics
    metadata = {
        'created': datetime.now().isoformat(),
        'random_seed': random_seed,
        'train_count': len(train_data['targets']),
        'val_count': len(val_data['targets']),
        'train_lengths': {
            'min': min(train_data['lengths']),
            'max': max(train_data['lengths']),
            'mean': np.mean(train_data['lengths']),
            'median': np.median(train_data['lengths']),
        },
        'val_lengths': {
            'min': min(val_data['lengths']),
            'max': max(val_data['lengths']),
            'mean': np.mean(val_data['lengths']),
            'median': np.median(val_data['lengths']),
        },
        'feature_dim': train_data['features'][0].shape[1],  # 4 + 16 = 20
    }
    
    return train_data, val_data, metadata


def save_dataset(
    train_data: Dict,
    val_data: Dict,
    metadata: Dict,
    output_dir: Path,
) -> None:
    """
    Save prepared dataset to disk.
    
    WHAT: Persist training data for future use
    WHY:  Expensive to recompute; enables resumable training
    HOW:  Pickle for arrays, JSON for metadata (human readable)
    """
    log_change(
        what=f"Saving dataset to {output_dir}",
        why="Enable fast loading for training without recomputation",
        how="Pickle for numpy arrays, JSON for metadata"
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    with open(output_dir / 'train_features.pkl', 'wb') as f:
        pickle.dump(train_data['features'], f)
    with open(output_dir / 'train_distances.pkl', 'wb') as f:
        pickle.dump(train_data['distances'], f)
    with open(output_dir / 'train_targets.pkl', 'wb') as f:
        pickle.dump(train_data['targets'], f)
    
    # Save validation data
    with open(output_dir / 'val_features.pkl', 'wb') as f:
        pickle.dump(val_data['features'], f)
    with open(output_dir / 'val_distances.pkl', 'wb') as f:
        pickle.dump(val_data['distances'], f)
    with open(output_dir / 'val_targets.pkl', 'wb') as f:
        pickle.dump(val_data['targets'], f)
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"  Saved to {output_dir}/")


def main():
    """Main entry point for training data preparation."""
    
    print("=" * 70)
    print("PHASE 2: PREPARE TRAINING DATA FOR ML DISTANCE PREDICTION")
    print("=" * 70)
    print()
    
    # Paths
    csv_path = PROJECT_ROOT / "data" / "train_labels.csv"
    output_dir = PROJECT_ROOT / "output" / "training_data"
    
    logger.info(f"Input:  {csv_path}")
    logger.info(f"Output: {output_dir}")
    print()
    
    # Step 1: Load raw data
    print("-" * 70)
    print("STEP 1: LOAD TRAINING LABELS")
    print("-" * 70)
    df = load_train_labels(str(csv_path))
    print()
    
    # Step 2: Build structures
    print("-" * 70)
    print("STEP 2: BUILD STRUCTURES")
    print("-" * 70)
    structures = build_structures(df)
    print()
    
    # Step 3: Prepare ML dataset
    print("-" * 70)
    print("STEP 3: PREPARE ML DATASET")
    print("-" * 70)
    train_data, val_data, metadata = prepare_ml_dataset(structures)
    print()
    
    # Step 4: Save
    print("-" * 70)
    print("STEP 4: SAVE DATASET")
    print("-" * 70)
    save_dataset(train_data, val_data, metadata, output_dir)
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Training structures: {metadata['train_count']:,}")
    print(f"Validation structures: {metadata['val_count']:,}")
    print(f"Training length range: {metadata['train_lengths']['min']:.0f} - {metadata['train_lengths']['max']:.0f}")
    print(f"Feature dimensions: {metadata['feature_dim']}")
    print(f"Output directory: {output_dir}")
    print()
    print("Ready for Phase 2 ML training!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
