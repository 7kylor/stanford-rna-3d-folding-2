#!/usr/bin/env python3
"""
Comprehensive validation comparing baseline TBM vs enhanced hybrid pipeline.
Measures TM-scores against ground truth labels.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def compute_tm_score(pred_coords: np.ndarray, ref_coords: np.ndarray) -> float:
    """
    Compute TM-score between predicted and reference coordinates.
    
    TM-score is a length-independent metric for structural similarity.
    Range: [0, 1], higher is better.
    """
    L = len(pred_coords)
    
    if L == 0 or len(ref_coords) != L:
        return 0.0
    
    # Handle NaN values
    valid_mask = ~(np.isnan(pred_coords).any(axis=1) | np.isnan(ref_coords).any(axis=1))
    if np.sum(valid_mask) < 3:
        return 0.0
    
    pred = pred_coords[valid_mask]
    ref = ref_coords[valid_mask]
    L_valid = len(pred)
    
    # d0 normalization factor (length-dependent)
    d0 = 1.24 * (L_valid - 15) ** (1/3) - 1.8
    d0 = max(0.5, d0)
    
    # Center both structures
    pred_centered = pred - pred.mean(axis=0)
    ref_centered = ref - ref.mean(axis=0)
    
    # Optimal rotation (Kabsch algorithm)
    H = pred_centered.T @ ref_centered
    try:
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        pred_rotated = pred_centered @ R
    except np.linalg.LinAlgError:
        pred_rotated = pred_centered
    
    # Compute TM-score
    distances = np.sqrt(np.sum((pred_rotated - ref_centered) ** 2, axis=1))
    tm_score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / L_valid
    
    return tm_score


def load_ground_truth(labels_file: str) -> dict:
    """Load ground truth coordinates from labels file."""
    df = pd.read_csv(labels_file)
    
    ground_truth = {}
    
    for _, row in df.iterrows():
        # ID format is like "8ZNQ_1" where 8ZNQ is target_id and 1 is resid
        id_str = row['ID']
        parts = id_str.rsplit('_', 1)
        target_id = parts[0]
        resid = int(parts[1])
        
        if target_id not in ground_truth:
            ground_truth[target_id] = {}
        
        # Use x_1, y_1, z_1 which are the first conformer coordinates
        x = row['x_1']
        y = row['y_1']
        z = row['z_1']
        
        # Skip invalid coordinates (marked as -1e+18)
        if x < -1e10:
            continue
            
        ground_truth[target_id][resid] = np.array([x, y, z])
    
    # Convert to arrays
    result = {}
    for target_id, residues in ground_truth.items():
        if not residues:
            continue
        max_resid = max(residues.keys())
        coords = np.full((max_resid, 3), np.nan)
        for resid, coord in residues.items():
            coords[resid - 1] = coord
        result[target_id] = coords
    
    return result


def run_baseline_pipeline(sequences_df, config):
    """Run baseline TBM pipeline."""
    from rna_tbm import TBMPipeline
    
    pipeline = TBMPipeline(config)
    pipeline.load_or_build_database()
    
    predictions = {}
    for idx, row in sequences_df.iterrows():
        result = pipeline.predict_single(
            target_id=row['target_id'],
            sequence=row['sequence'],
            temporal_cutoff=row.get('temporal_cutoff', '2099-12-31'),
            num_models=5,
        )
        # Use first model
        predictions[row['target_id']] = result.models[0]
    
    return predictions


def run_hybrid_pipeline(sequences_df, config):
    """Run enhanced hybrid pipeline."""
    from rna_tbm.hybrid_pipeline import HybridPipeline
    
    pipeline = HybridPipeline(
        config,
        template_weight=0.5,
        distance_weight=0.5,
        use_covariation=True,
        use_embeddings=True,
    )
    pipeline.load_or_build_database()
    
    predictions = {}
    for idx, row in sequences_df.iterrows():
        result = pipeline.predict_single(
            target_id=row['target_id'],
            sequence=row['sequence'],
            temporal_cutoff=row.get('temporal_cutoff', '2099-12-31'),
            num_models=5,
        )
        # Use first model
        predictions[row['target_id']] = result.models[0]
    
    return predictions


def main():
    """Run validation and compare pipelines."""
    from rna_tbm.config import PipelineConfig, PathConfig
    
    print("=" * 70)
    print("Stanford RNA 3D Folding - Pipeline Performance Comparison")
    print("=" * 70)
    
    # Set up paths
    paths = PathConfig()
    
    # Find validation files
    sequences_file = PROJECT_ROOT / "data" / "sequences" / "validation_sequences.csv"
    labels_file = PROJECT_ROOT / "data" / "validation_labels.csv"
    
    if not sequences_file.exists():
        print(f"Error: Validation sequences not found at {sequences_file}")
        return 1
    
    if not labels_file.exists():
        print(f"Error: Validation labels not found at {labels_file}")
        return 1
    
    # Load data
    print(f"\n1. Loading validation data...")
    sequences_df = pd.read_csv(sequences_file)
    ground_truth = load_ground_truth(str(labels_file))
    
    # Limit for speed
    max_targets = 10
    if len(sequences_df) > max_targets:
        sequences_df = sequences_df.head(max_targets)
    
    print(f"   Sequences: {len(sequences_df)}")
    print(f"   Ground truth targets: {len(ground_truth)}")
    
    # Create config
    config = PipelineConfig(paths=paths, max_files=2000)
    
    # Run baseline pipeline
    print(f"\n2. Running baseline TBM pipeline...")
    start = time.time()
    baseline_preds = run_baseline_pipeline(sequences_df, config)
    baseline_time = time.time() - start
    print(f"   Time: {baseline_time:.1f}s")
    
    # Run hybrid pipeline
    print(f"\n3. Running hybrid pipeline with RNA-FM...")
    start = time.time()
    hybrid_preds = run_hybrid_pipeline(sequences_df, config)
    hybrid_time = time.time() - start
    print(f"   Time: {hybrid_time:.1f}s")
    
    # Compute TM-scores
    print(f"\n4. Computing TM-scores...")
    print("-" * 70)
    print(f"{'Target ID':<20} {'Length':>8} {'Baseline':>12} {'Hybrid':>12} {'Δ':>10}")
    print("-" * 70)
    
    baseline_scores = []
    hybrid_scores = []
    
    for _, row in sequences_df.iterrows():
        target_id = row['target_id']
        seq_len = len(row['sequence'])
        
        if target_id not in ground_truth:
            print(f"{target_id:<20} {seq_len:>8} {'N/A':>12} {'N/A':>12}")
            continue
        
        ref_coords = ground_truth[target_id]
        
        # Get predictions
        baseline_pred = baseline_preds.get(target_id)
        hybrid_pred = hybrid_preds.get(target_id)
        
        if baseline_pred is None or hybrid_pred is None:
            continue
        
        # Ensure same length
        min_len = min(len(ref_coords), len(baseline_pred), len(hybrid_pred))
        ref = ref_coords[:min_len]
        base_pred = baseline_pred[:min_len]
        hyb_pred = hybrid_pred[:min_len]
        
        # Compute TM-scores
        baseline_tm = compute_tm_score(base_pred, ref)
        hybrid_tm = compute_tm_score(hyb_pred, ref)
        delta = hybrid_tm - baseline_tm
        
        baseline_scores.append(baseline_tm)
        hybrid_scores.append(hybrid_tm)
        
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{target_id:<20} {seq_len:>8} {baseline_tm:>12.4f} {hybrid_tm:>12.4f} {delta_str:>10}")
    
    # Summary
    print("-" * 70)
    
    if baseline_scores and hybrid_scores:
        avg_baseline = np.mean(baseline_scores)
        avg_hybrid = np.mean(hybrid_scores)
        avg_delta = avg_hybrid - avg_baseline
        improvement = (avg_delta / max(avg_baseline, 0.0001)) * 100
        
        print(f"\n{'SUMMARY':^70}")
        print("=" * 70)
        print(f"Baseline TBM average TM-score:    {avg_baseline:.4f}")
        print(f"Hybrid + RNA-FM average TM-score: {avg_hybrid:.4f}")
        print(f"Improvement:                      {avg_delta:+.4f} ({improvement:+.1f}%)")
        print()
        print(f"Baseline time: {baseline_time:.1f}s ({baseline_time/len(baseline_scores):.1f}s/target)")
        print(f"Hybrid time:   {hybrid_time:.1f}s ({hybrid_time/len(hybrid_scores):.1f}s/target)")
        print("=" * 70)
    else:
        print("\nNo valid comparisons could be made.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
