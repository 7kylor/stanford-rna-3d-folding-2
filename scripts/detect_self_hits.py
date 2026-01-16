#!/usr/bin/env python3
"""
Self-hit detection: Check if test sequences match training templates.

This is a CRITICAL step for achieving high accuracy - if a test target 
has an exact or near-exact match in training data, we can achieve TM > 0.95.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_template_db(db_path: str) -> Dict:
    """Load training template database."""
    with open(db_path, 'rb') as f:
        return pickle.load(f)


def load_test_sequences(csv_path: str) -> Dict[str, str]:
    """Load test sequences from CSV."""
    df = pd.read_csv(csv_path)
    return dict(zip(df['target_id'], df['sequence']))


def sequence_identity(seq1: str, seq2: str) -> float:
    """Calculate sequence identity between two sequences."""
    if len(seq1) != len(seq2):
        # Use length-normalized identity for different lengths
        min_len = min(len(seq1), len(seq2))
        matches = sum(c1 == c2 for c1, c2 in zip(seq1[:min_len], seq2[:min_len]))
        return matches / max(len(seq1), len(seq2))
    else:
        matches = sum(c1 == c2 for c1, c2 in zip(seq1, seq2))
        return matches / len(seq1)


def find_exact_matches(test_sequences: Dict[str, str], 
                       template_db: Dict) -> Dict[str, List[str]]:
    """Find exact sequence matches between test and training."""
    exact_matches = {}
    
    # Build sequence to template ID mapping
    seq_to_templates = {}
    for template_id, data in template_db['structures'].items():
        seq = data['sequence']
        if seq not in seq_to_templates:
            seq_to_templates[seq] = []
        seq_to_templates[seq].append(template_id)
    
    for test_id, test_seq in test_sequences.items():
        if test_seq in seq_to_templates:
            exact_matches[test_id] = seq_to_templates[test_seq]
    
    return exact_matches


def find_high_identity_matches(test_sequences: Dict[str, str],
                               template_db: Dict,
                               identity_threshold: float = 0.95) -> Dict[str, List[Tuple[str, float]]]:
    """Find high-identity matches (>95% by default)."""
    high_matches = {}
    
    for test_id, test_seq in test_sequences.items():
        matches = []
        test_len = len(test_seq)
        
        for template_id, data in template_db['structures'].items():
            template_seq = data['sequence']
            template_len = len(template_seq)
            
            # Skip if length difference is too large for high identity
            if abs(test_len - template_len) > max(test_len, template_len) * (1 - identity_threshold):
                continue
            
            identity = sequence_identity(test_seq, template_seq)
            if identity >= identity_threshold:
                matches.append((template_id, identity))
        
        if matches:
            # Sort by identity descending
            matches.sort(key=lambda x: -x[1])
            high_matches[test_id] = matches
    
    return high_matches


def analyze_self_hits(test_csv: str, template_db_path: str):
    """Main analysis function."""
    logger.info("Loading template database...")
    db = load_template_db(template_db_path)
    logger.info(f"  Loaded {db['num_templates']} templates")
    
    logger.info("Loading test sequences...")
    test_seqs = load_test_sequences(test_csv)
    logger.info(f"  Loaded {len(test_seqs)} test sequences")
    
    # Find exact matches
    logger.info("\n=== EXACT MATCHES ===")
    exact = find_exact_matches(test_seqs, db)
    if exact:
        for test_id, templates in exact.items():
            logger.info(f"  {test_id}: EXACT MATCH with {templates}")
            logger.info(f"    -> Expected TM-score: ~1.0")
    else:
        logger.info("  No exact matches found")
    
    # Find high identity matches
    logger.info("\n=== HIGH IDENTITY MATCHES (>95%) ===")
    high_id = find_high_identity_matches(test_seqs, db, 0.95)
    if high_id:
        for test_id, matches in high_id.items():
            if test_id not in exact:  # Don't repeat exact matches
                best = matches[0]
                logger.info(f"  {test_id}: {best[1]*100:.1f}% identity with {best[0]}")
                logger.info(f"    -> Expected TM-score: >0.9")
    else:
        logger.info("  No high-identity matches found")
    
    # Find moderate matches (80-95%)
    logger.info("\n=== MODERATE MATCHES (80-95%) ===")
    moderate = find_high_identity_matches(test_seqs, db, 0.80)
    moderate_only = {k: v for k, v in moderate.items() 
                     if k not in exact and k not in high_id}
    if moderate_only:
        for test_id, matches in moderate_only.items():
            best = matches[0]
            if best[1] < 0.95:
                logger.info(f"  {test_id}: {best[1]*100:.1f}% identity with {best[0]}")
                logger.info(f"    -> Expected TM-score: 0.6-0.9")
    else:
        logger.info("  No moderate-identity matches found")
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    n_exact = len(exact)
    n_high = len(high_id) - n_exact
    n_moderate = len(moderate_only)
    n_novel = len(test_seqs) - n_exact - n_high - n_moderate
    
    logger.info(f"  Exact matches (TM ~1.0): {n_exact}/{len(test_seqs)}")
    logger.info(f"  High identity (TM >0.9): {n_high}/{len(test_seqs)}")
    logger.info(f"  Moderate identity (TM 0.6-0.9): {n_moderate}/{len(test_seqs)}")
    logger.info(f"  Novel/Low identity: {n_novel}/{len(test_seqs)}")
    
    expected_avg = (n_exact * 1.0 + n_high * 0.9 + n_moderate * 0.75 + n_novel * 0.3) / len(test_seqs)
    logger.info(f"\n  Expected average TM-score: {expected_avg:.2f}")
    
    return exact, high_id, moderate


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect self-hits between test and training")
    parser.add_argument('--test', '-t', default='data/sequences/test_sequences.csv',
                        help='Test sequences CSV')
    parser.add_argument('--db', '-d', default='output/training_template_db.pkl',
                        help='Template database path')
    args = parser.parse_args()
    
    analyze_self_hits(args.test, args.db)


if __name__ == '__main__':
    main()
