#!/usr/bin/env python3
"""
Test enhanced pipeline with validation data.
Runs predictions on a subset and measures accuracy.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


def compute_tm_score(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
    """
    Compute TM-score between predicted and true coordinates.
    Simplified version using superposition.
    """
    if len(pred_coords) != len(true_coords):
        return 0.0
    
    L = len(pred_coords)
    if L == 0:
        return 0.0
    
    # Center both
    pred_centered = pred_coords - np.mean(pred_coords, axis=0)
    true_centered = true_coords - np.mean(true_coords, axis=0)
    
    # Optimal rotation using SVD
    H = pred_centered.T @ true_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    pred_rotated = pred_centered @ R
    
    # Compute RMSD
    distances = np.linalg.norm(pred_rotated - true_centered, axis=1)
    
    # TM-score formula
    d0 = 1.24 * (max(L, 15) - 15) ** (1/3) - 1.8
    d0 = max(d0, 0.5)  # Minimum d0
    
    tm_score = np.sum(1 / (1 + (distances / d0) ** 2)) / L
    
    return float(tm_score)


def run_validation(num_samples: int = 5):
    """Run validation on a few samples."""
    print("=" * 60)
    print("Enhanced RNA-TBM Pipeline Validation")
    print("=" * 60)
    
    # Load validation data
    val_seqs = pd.read_csv(project_root / "data/sequences/validation_sequences.csv")
    val_labels = pd.read_csv(project_root / "data/validation_labels.csv")
    
    print(f"\nTotal validation sequences: {len(val_seqs)}")
    print(f"Testing on first {num_samples} samples\n")
    
    # Import pipeline components
    from rna_tbm.config import EnhancedPipelineConfig, PipelineConfig
    from rna_tbm.enhanced_pipeline import EnhancedTBMPipeline
    from rna_tbm.pipeline import TBMPipeline
    
    # Test both baseline and enhanced
    results = {
        'baseline': {'tm_scores': [], 'times': []},
        'enhanced': {'tm_scores': [], 'times': []},
    }
    
    # Select samples
    samples = val_seqs.head(num_samples)
    
    # Get ground truth labels for these samples
    labels_dict = {}
    for _, row in val_labels.iterrows():
        target_id = row['ID'].split('_')[0]  # e.g., "8ZNQ_1_1" -> "8ZNQ"
        resid = int(row['ID'].split('_')[1])
        if target_id not in labels_dict:
            labels_dict[target_id] = {}
        labels_dict[target_id][resid] = [row['x_1'], row['y_1'], row['z_1']]
    
    # Test baseline TBM
    print("Testing BASELINE TBM pipeline...")
    print("-" * 40)
    
    # Use explicit config with max_files
    baseline_config = PipelineConfig(
        max_files=100,       # Limit for speed, no file size filter
    )
    baseline_pipeline = TBMPipeline(baseline_config)
    
    try:
        baseline_pipeline.load_or_build_database()
        
        for idx, row in samples.iterrows():
            target_id = row['target_id']
            sequence = row['sequence']
            temporal_cutoff = row['temporal_cutoff']
            
            start_time = time.time()
            try:
                pred = baseline_pipeline.predict_single(
                    target_id, sequence, temporal_cutoff, num_models=5
                )
                elapsed = time.time() - start_time
                
                # Get ground truth
                if target_id in labels_dict:
                    true_coords = []
                    for i in range(1, len(sequence) + 1):
                        if i in labels_dict[target_id]:
                            true_coords.append(labels_dict[target_id][i])
                        else:
                            true_coords.append([0, 0, 0])
                    true_coords = np.array(true_coords)
                    
                    tm = compute_tm_score(pred.models[0], true_coords)
                else:
                    tm = 0.0  # No ground truth
                
                results['baseline']['tm_scores'].append(tm)
                results['baseline']['times'].append(elapsed)
                print(f"  {target_id}: {len(sequence):4d}nt, TM={tm:.4f}, time={elapsed:.2f}s")
            except Exception as e:
                print(f"  {target_id}: ERROR - {e}")
                results['baseline']['tm_scores'].append(0.0)
                results['baseline']['times'].append(0.0)
    except Exception as e:
        print(f"Baseline pipeline error: {e}")
    
    # Test enhanced TBM
    print("\nTesting ENHANCED TBM pipeline...")
    print("-" * 40)
    
    enhanced_config = EnhancedPipelineConfig(
        use_msa_covariation=False,  # Skip MSA for speed (no per-target MSA files)
        use_embeddings=True,
        use_torsion_refinement=True,
        use_metal_prediction=True,
        use_functional_homology=False,
    )
    enhanced_pipeline = EnhancedTBMPipeline(enhanced_config)
    
    try:
        enhanced_pipeline.load_or_build_database()
        
        for idx, row in samples.iterrows():
            target_id = row['target_id']
            sequence = row['sequence']
            temporal_cutoff = row['temporal_cutoff']
            
            start_time = time.time()
            try:
                pred = enhanced_pipeline.predict_single(
                    target_id, sequence, temporal_cutoff, num_models=5
                )
                elapsed = time.time() - start_time
                
                # Get ground truth
                if target_id in labels_dict:
                    true_coords = []
                    for i in range(1, len(sequence) + 1):
                        if i in labels_dict[target_id]:
                            true_coords.append(labels_dict[target_id][i])
                        else:
                            true_coords.append([0, 0, 0])
                    true_coords = np.array(true_coords)
                    
                    tm = compute_tm_score(pred.models[0], true_coords)
                else:
                    tm = 0.0
                
                results['enhanced']['tm_scores'].append(tm)
                results['enhanced']['times'].append(elapsed)
                print(f"  {target_id}: {len(sequence):4d}nt, TM={tm:.4f}, time={elapsed:.2f}s")
            except Exception as e:
                print(f"  {target_id}: ERROR - {e}")
                results['enhanced']['tm_scores'].append(0.0)
                results['enhanced']['times'].append(0.0)
    except Exception as e:
        print(f"Enhanced pipeline error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if results['baseline']['tm_scores']:
        baseline_mean = np.mean(results['baseline']['tm_scores'])
        baseline_time = np.sum(results['baseline']['times'])
        print(f"\nBaseline TBM:")
        print(f"  Mean TM-score: {baseline_mean:.4f}")
        print(f"  Total time:    {baseline_time:.2f}s")
    
    if results['enhanced']['tm_scores']:
        enhanced_mean = np.mean(results['enhanced']['tm_scores'])
        enhanced_time = np.sum(results['enhanced']['times'])
        print(f"\nEnhanced TBM:")
        print(f"  Mean TM-score: {enhanced_mean:.4f}")
        print(f"  Total time:    {enhanced_time:.2f}s")
    
    if results['baseline']['tm_scores'] and results['enhanced']['tm_scores']:
        improvement = enhanced_mean - baseline_mean
        print(f"\nImprovement: {improvement:+.4f} TM-score")
    
    return results


if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_validation(num_samples)
