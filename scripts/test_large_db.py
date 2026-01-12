#!/usr/bin/env python3
"""
Test pipeline with larger template database.
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


def compute_tm_score(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
    """Compute TM-score between predicted and true coordinates."""
    # Filter valid coordinates (not -1e18)
    valid_mask = (true_coords[:, 0] > -1e17)
    
    if valid_mask.sum() < 3:
        return 0.0
    
    pred_valid = pred_coords[valid_mask]
    true_valid = true_coords[valid_mask]
    
    L = len(pred_valid)
    if L < 3:
        return 0.0
    
    # Center both
    pred_centered = pred_valid - np.mean(pred_valid, axis=0)
    true_centered = true_valid - np.mean(true_valid, axis=0)
    
    # Optimal rotation using SVD
    H = pred_centered.T @ true_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    pred_rotated = pred_centered @ R
    
    # Compute distances
    distances = np.linalg.norm(pred_rotated - true_centered, axis=1)
    
    # TM-score formula
    d0 = 1.24 * (max(L, 15) - 15) ** (1/3) - 1.8
    d0 = max(d0, 0.5)
    
    tm_score = np.sum(1 / (1 + (distances / d0) ** 2)) / L
    
    return float(tm_score)


def load_ground_truth(labels_file: Path, target_ids: list) -> dict:
    """Load ground truth coordinates for validation targets."""
    print(f"Loading ground truth from {labels_file}...")
    labels_df = pd.read_csv(labels_file)
    
    labels_dict = {}
    for target_id in target_ids:
        # Find rows for this target
        mask = labels_df['ID'].str.startswith(f"{target_id}_")
        target_labels = labels_df[mask]
        
        if len(target_labels) == 0:
            continue
        
        coords_by_resid = {}
        for _, row in target_labels.iterrows():
            resid = row['resid']
            x, y, z = row['x_1'], row['y_1'], row['z_1']
            coords_by_resid[resid] = [x, y, z]
        
        labels_dict[target_id] = coords_by_resid
    
    return labels_dict


def run_validation(num_samples: int = 5, max_templates: int = 2000):
    """Run validation with larger template database."""
    print("=" * 60)
    print("RNA-TBM Pipeline - Large Database Validation")
    print("=" * 60)
    
    # Load data
    val_seqs = pd.read_csv(project_root / "data/sequences/validation_sequences.csv")
    
    print(f"\nTotal validation sequences: {len(val_seqs)}")
    print(f"Testing on {num_samples} samples with {max_templates} templates\n")
    
    # Import pipeline
    from rna_tbm.config import PipelineConfig
    from rna_tbm.pipeline import TBMPipeline
    
    # Create config with larger database
    config = PipelineConfig(
        max_files=max_templates,
        max_file_size_mb=None,  # No size limit
    )
    
    # Create pipeline
    pipeline = TBMPipeline(config)
    
    print("1. Building template database (this may take a few minutes)...")
    start_build = time.time()
    pipeline.load_or_build_database()
    build_time = time.time() - start_build
    print(f"   Database built in {build_time:.1f}s")
    print(f"   Total templates: {len(pipeline.template_db.templates)}")
    
    # Select samples
    samples = val_seqs.head(num_samples)
    
    # Load ground truth
    labels_dict = load_ground_truth(
        project_root / "data/validation_labels.csv",
        samples['target_id'].tolist()
    )
    
    print(f"\n2. Running predictions on {num_samples} targets...")
    print("-" * 50)
    
    results = []
    total_start = time.time()
    
    for idx, row in samples.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        temporal_cutoff = row['temporal_cutoff']
        
        start = time.time()
        try:
            pred = pipeline.predict_single(
                target_id, sequence, temporal_cutoff, num_models=5
            )
            elapsed = time.time() - start
            
            # Get ground truth
            tm_score = 0.0
            if target_id in labels_dict:
                # Build true_coords array
                true_coords = []
                for i in range(1, len(sequence) + 1):
                    if i in labels_dict[target_id]:
                        true_coords.append(labels_dict[target_id][i])
                    else:
                        true_coords.append([-1e18, -1e18, -1e18])
                true_coords = np.array(true_coords)
                
                tm_score = compute_tm_score(pred.models[0], true_coords)
            
            results.append({
                'target_id': target_id,
                'length': len(sequence),
                'tm_score': tm_score,
                'time': elapsed,
                'has_template': pipeline.stats.num_with_templates > idx
            })
            
            print(f"  {target_id}: {len(sequence):4d}nt, TM={tm_score:.4f}, time={elapsed:.2f}s")
            
        except Exception as e:
            print(f"  {target_id}: ERROR - {e}")
            results.append({
                'target_id': target_id,
                'length': len(sequence),
                'tm_score': 0.0,
                'time': 0.0,
                'has_template': False
            })
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    tm_scores = [r['tm_score'] for r in results]
    mean_tm = np.mean(tm_scores) if tm_scores else 0.0
    
    print(f"\nConfiguration:")
    print(f"  Templates: {len(pipeline.template_db.templates)}")
    print(f"  Build time: {build_time:.1f}s")
    
    print(f"\nPerformance:")
    print(f"  Mean TM-score: {mean_tm:.4f}")
    print(f"  Total time: {total_time:.2f}s")
    
    print(f"\nPipeline Stats:")
    print(pipeline.stats.summary())
    
    return results


if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_templates = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    run_validation(num_samples, max_templates)
