#!/usr/bin/env python3
"""
Debug script to investigate why 9G4J (88.6% identity with 9G4I) gets TM=0.000
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pickle

def main():
    print("=== Debugging 9G4J Template Matching ===\n")
    
    # Load training template DB
    db_path = PROJECT_ROOT / "output" / "training_template_db.pkl"
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    
    structures = db['structures']
    
    # Load test sequences
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "sequences" / "test_sequences.csv")
    
    # Get 9G4J sequence
    target_row = test_df[test_df['target_id'] == '9G4J'].iloc[0]
    test_seq = target_row['sequence']
    print(f"9G4J test sequence: {len(test_seq)}nt")
    print(f"  {test_seq[:50]}...")
    
    # Check if 9G4I exists in training
    if '9G4I' in structures:
        train_struct = structures['9G4I']
        print(f"\n9G4I training structure found:")
        print(f"  Sequence length: {train_struct['length']}")
        print(f"  Sequence: {train_struct['sequence'][:50]}...")
        print(f"  Coords shape: {train_struct['coords'].shape}")
        
        # Check sequence identity
        train_seq = train_struct['sequence']
        if len(test_seq) == len(train_seq):
            matches = sum(a == b for a, b in zip(test_seq, train_seq))
            identity = matches / len(test_seq)
            print(f"\n  Sequence identity: {identity*100:.1f}%")
        else:
            print(f"\n  Length mismatch: test={len(test_seq)}, train={len(train_seq)}")
            
            # Find overlapping k-mers
            k = 6
            test_kmers = set(test_seq[i:i+k] for i in range(len(test_seq)-k+1))
            train_kmers = set(train_seq[i:i+k] for i in range(len(train_seq)-k+1))
            overlap = test_kmers & train_kmers
            print(f"  Shared {k}-mers: {len(overlap)}/{len(test_kmers)} ({100*len(overlap)/len(test_kmers):.1f}%)")
    else:
        print("\n9G4I NOT found in training!")
        
        # Search for similar IDs
        similar = [k for k in structures.keys() if k.startswith('9G4')]
        print(f"  IDs starting with '9G4': {similar}")
    
    # Load ground truth for 9G4J
    gt_df = pd.read_csv(PROJECT_ROOT / "data" / "validation_labels.csv")
    gt_9g4j = gt_df[gt_df['ID'].str.startswith('9G4J_')]
    print(f"\n9G4J ground truth: {len(gt_9g4j)} residues")
    
    # Check if coordinates look reasonable
    if len(gt_9g4j) > 0:
        coords = gt_9g4j[['x_1', 'y_1', 'z_1']].values
        print(f"  Coord range X: {coords[:,0].min():.1f} to {coords[:,0].max():.1f}")
        print(f"  Coord range Y: {coords[:,1].min():.1f} to {coords[:,1].max():.1f}")
        print(f"  Coord range Z: {coords[:,2].min():.1f} to {coords[:,2].max():.1f}")
    
    # Debug template search
    print("\n=== Template Search Debug ===")
    from rna_tbm.template_db import TemplateDB
    
    template_db = TemplateDB.load_from_training(str(db_path))
    print(f"Loaded template DB: {len(template_db)} templates")
    
    # Search for templates
    hits = template_db.search(
        test_seq,
        max_hits=20,
        min_identity=0.2,
        min_coverage=0.2
    )
    
    print(f"\nTemplate hits for 9G4J ({len(test_seq)}nt):")
    for i, hit in enumerate(hits[:10]):
        print(f"  {i+1}. {hit.pdb_id} chain {hit.chain_id}: {hit.identity*100:.1f}% id, {hit.coverage*100:.1f}% cov")
    
    # Check if 9G4I appears
    hit_ids = [h.pdb_id for h in hits]
    if '9G4I' in hit_ids:
        print("\n✓ 9G4I found as template hit!")
    else:
        print("\n✗ 9G4I NOT in template hits")
        print("  This explains why 9G4J got TM=0.000")

if __name__ == "__main__":
    main()
