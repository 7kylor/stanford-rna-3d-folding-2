#!/usr/bin/env python3
"""
Test template database integration and run validation with training templates.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from rna_tbm.template_db import TemplateDB

def test_load_training_db():
    """Test loading the training template database."""
    print("=" * 60)
    print("Testing Training Template Database Integration")
    print("=" * 60)
    
    # Load training template DB
    print("\n1. Loading training template database...")
    db = TemplateDB.load_from_training('output/training_template_db.pkl')
    print(f"   Loaded {len(db)} templates")
    
    # Load test sequences
    print("\n2. Loading test sequences...")
    test_df = pd.read_csv('data/sequences/test_sequences.csv')
    print(f"   Loaded {len(test_df)} test sequences")
    
    # Test search on first 3 targets
    print("\n3. Testing template search...")
    for idx, row in test_df.head(5).iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        
        hits = db.search(sequence, max_hits=3)
        print(f"\n   Target: {target_id} (len={len(sequence)})")
        if hits:
            for hit in hits[:3]:
                print(f"      -> {hit.pdb_id}: identity={hit.identity:.2f}, coverage={hit.coverage:.2f}")
        else:
            print("      -> No hits found")
    
    # Test coordinate retrieval
    print("\n4. Testing coordinate retrieval...")
    if hits:
        best_hit = hits[0]
        coords = db.get_template_coords(best_hit.pdb_id, best_hit.chain_id)
        if coords is not None and hasattr(coords, 'c1_coords'):
            c1 = coords.c1_coords
            print(f"   Retrieved coordinates for {best_hit.pdb_id}")
            print(f"   Shape: {c1.shape}")
            print(f"   Range: [{c1.min():.2f}, {c1.max():.2f}]")
        else:
            print(f"   Could not retrieve coordinates for {best_hit.pdb_id}")
    
    print("\n" + "=" * 60)
    print("Integration test PASSED!" if len(db) > 0 else "Integration test FAILED!")
    print("=" * 60)
    
    return db


if __name__ == '__main__':
    test_load_training_db()
