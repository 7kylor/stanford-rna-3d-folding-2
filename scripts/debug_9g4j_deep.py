#!/usr/bin/env python3
"""
Deep debug: Why isn't 9G4I appearing as a candidate for 9G4J?
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pickle
from collections import defaultdict

def main():
    # Load training template DB
    db_path = PROJECT_ROOT / "output" / "training_template_db.pkl"
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    
    structures = db['structures']
    
    # Get 9G4J sequence (test)
    import pandas as pd
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "sequences" / "test_sequences.csv")
    test_seq = test_df[test_df['target_id'] == '9G4J'].iloc[0]['sequence']
    
    # Get 9G4I sequence (training)
    train_seq = structures['9G4I']['sequence']
    
    print(f"Test (9G4J):  {len(test_seq)}nt")
    print(f"Train (9G4I): {len(train_seq)}nt")
    
    # Build k-mer index for 9G4I
    k = 6
    train_kmers = {}
    for pos in range(len(train_seq) - k + 1):
        kmer = train_seq[pos:pos + k]
        if kmer not in train_kmers:
            train_kmers[kmer] = []
        train_kmers[kmer].append(pos)
    
    # Count k-mer hits from test to train
    hit_positions = []
    for q_pos in range(len(test_seq) - k + 1):
        kmer = test_seq[q_pos:q_pos + k]
        if kmer in train_kmers:
            for t_pos in train_kmers[kmer]:
                hit_positions.append((q_pos, t_pos))
    
    print(f"\nK-mer hits: {len(hit_positions)}")
    
    # Group by diagonal
    diag_counts = defaultdict(int)
    for q_pos, t_pos in hit_positions:
        diag = q_pos - t_pos
        diag_counts[diag] += 1
    
    print(f"\nDiagonals with hits:")
    sorted_diags = sorted(diag_counts.items(), key=lambda x: -x[1])[:10]
    for diag, count in sorted_diags:
        print(f"  diag={diag:4d}: {count} hits")
    
    # Best diagonal analysis
    if sorted_diags:
        best_diag = sorted_diags[0][0]
        offset = -best_diag
        
        print(f"\nBest diagonal: {best_diag}")
        print(f"Offset: {offset}")
        
        # Compute alignment on best diagonal
        aligned = 0
        matches = 0
        aligned_pairs = []
        
        for q_idx in range(len(test_seq)):
            t_idx = q_idx + offset
            if 0 <= t_idx < len(train_seq):
                aligned += 1
                if test_seq[q_idx] == train_seq[t_idx]:
                    matches += 1
                    aligned_pairs.append((q_idx, t_idx))
        
        identity = matches / aligned if aligned > 0 else 0
        coverage = aligned / len(test_seq)
        
        print(f"\nOn best diagonal:")
        print(f"  Aligned positions: {aligned}/{len(test_seq)}")
        print(f"  Matches: {matches}")
        print(f"  Identity: {identity*100:.1f}%")
        print(f"  Coverage: {coverage*100:.1f}%")
        
        # Check what percent of 9G4I is covered
        print(f"  9G4I coverage: {aligned}/{len(train_seq)} = {aligned/len(train_seq)*100:.1f}%")
    
    # Check if 9G4I would pass the k-mer pre-filter
    print("\n=== K-mer Pre-filter Check ===")
    query_kmers = len(test_seq) - k + 1
    hit_count_for_9g4i = len(hit_positions)  # This is total hits, not unique
    
    # Actually count unique (pdb_id, chain_id) hits
    print(f"Query k-mers: {query_kmers}")
    print(f"Hit count for 9G4I: {hit_count_for_9g4i}")
    print(f"Threshold (10%): {query_kmers * 0.1}")
    
    if hit_count_for_9g4i >= query_kmers * 0.1:
        print("✓ 9G4I SHOULD pass k-mer pre-filter")
    else:
        print("✗ 9G4I fails k-mer pre-filter")
    
    # Now simulate the full search to see where 9G4I drops out
    print("\n=== Simulating Full Search ===")
    from rna_tbm.template_db import TemplateDB
    
    # Manually build and check
    template_db = TemplateDB.load_from_training(str(db_path))
    
    # Check if 9G4I is in the kmer_index at all
    in_index = False
    for kmer in test_seq[:10]:  # Just check first few
        for entry in template_db.kmer_index.get(kmer, []):
            if entry[0] == '9G4I':
                in_index = True
                break
        if in_index:
            break
    
    print(f"9G4I in k-mer index: {in_index}")
    
    # Check minimum identity threshold
    print(f"\nMinimum identity threshold in search: 0.20")
    print(f"Our computed identity: {identity*100:.1f}%")
    
    if identity >= 0.20:
        print("✓ Should pass identity threshold")
    else:
        print("✗ Fails identity threshold")

if __name__ == "__main__":
    main()
