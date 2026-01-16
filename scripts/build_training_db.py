#!/usr/bin/env python3
"""
Build comprehensive template database from training data.

This script converts train_labels.csv into a searchable template database,
expanding our templates from 242 to 5,717 structures.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_sequence_from_residues(residues: List[str]) -> str:
    """Convert list of residue names to sequence string."""
    base_map = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U', 'T': 'U'}
    sequence = []
    for res in residues:
        base = base_map.get(res.upper(), 'N')
        sequence.append(base)
    return ''.join(sequence)


def load_train_labels(csv_path: str) -> pd.DataFrame:
    """Load training labels with progress."""
    logger.info(f"Loading {csv_path}...")
    
    # Use chunked reading for large files
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=500000):
        chunks.append(chunk)
        logger.info(f"  Loaded {len(chunks) * 500000} rows...")
    
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Total rows: {len(df):,}")
    return df


def parse_target_id(id_str: str) -> Tuple[str, int]:
    """Parse ID like '157D_1' into (target_id='157D', resid=1)."""
    parts = id_str.rsplit('_', 1)
    return parts[0], int(parts[1])


def build_structures(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Group coordinates by target and build structure dictionary.
    
    Returns:
        Dict[target_id, {
            'sequence': str,
            'coords': np.ndarray (L, 3),
            'length': int,
            'chains': list,
        }]
    """
    structures = {}
    
    # Group by target
    df['target_id'] = df['ID'].apply(lambda x: x.rsplit('_', 1)[0])
    
    target_groups = df.groupby('target_id')
    logger.info(f"Processing {len(target_groups)} unique targets...")
    
    for target_id, group in tqdm(target_groups, desc="Building structures"):
        # Sort by residue number
        group = group.copy()
        group['resid_num'] = group['ID'].apply(lambda x: int(x.rsplit('_', 1)[1]))
        group = group.sort_values('resid_num')
        
        # Handle multiple copies - take first copy (copy=1 or lowest)
        if 'copy' in group.columns:
            first_copy = group['copy'].min()
            group = group[group['copy'] == first_copy]
        
        # Extract sequence
        sequence = extract_sequence_from_residues(group['resname'].tolist())
        
        # Extract coordinates
        coords = group[['x_1', 'y_1', 'z_1']].values.astype(np.float32)
        
        # Center coordinates
        coords = coords - coords.mean(axis=0)
        
        structures[target_id] = {
            'sequence': sequence,
            'coords': coords,
            'length': len(sequence),
            'chains': group['chain'].unique().tolist() if 'chain' in group.columns else ['A'],
        }
    
    return structures


def build_kmer_index(structures: Dict[str, Dict], k: int = 6) -> Dict[str, List[str]]:
    """Build k-mer index for fast sequence search."""
    kmer_index = defaultdict(list)
    
    for target_id, struct in structures.items():
        seq = struct['sequence']
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if 'N' not in kmer:  # Skip ambiguous
                kmer_index[kmer].append(target_id)
    
    return dict(kmer_index)


def save_template_db(structures: Dict, kmer_index: Dict, output_path: str):
    """Save as pickled template database."""
    db = {
        'structures': structures,
        'kmer_index': kmer_index,
        'version': '2.0',
        'source': 'train_labels.csv',
        'num_templates': len(structures),
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(db, f)
    
    logger.info(f"Saved template database to {output_path}")
    logger.info(f"  Templates: {len(structures)}")
    logger.info(f"  K-mers indexed: {len(kmer_index)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build template DB from training data")
    parser.add_argument('--input', '-i', default='data/train_labels.csv',
                        help='Path to train_labels.csv')
    parser.add_argument('--output', '-o', default='output/training_template_db.pkl',
                        help='Output pickle file')
    parser.add_argument('--kmer-size', '-k', type=int, default=6,
                        help='K-mer size for indexing')
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_train_labels(args.input)
    
    # Build structures
    structures = build_structures(df)
    
    # Build k-mer index
    logger.info("Building k-mer index...")
    kmer_index = build_kmer_index(structures, k=args.kmer_size)
    
    # Save
    save_template_db(structures, kmer_index, str(output_path))
    
    # Print statistics
    lengths = [s['length'] for s in structures.values()]
    logger.info(f"\nStatistics:")
    logger.info(f"  Total structures: {len(structures)}")
    logger.info(f"  Length range: {min(lengths)} - {max(lengths)}")
    logger.info(f"  Mean length: {np.mean(lengths):.1f}")
    logger.info(f"  Median length: {np.median(lengths):.1f}")


if __name__ == '__main__':
    main()
