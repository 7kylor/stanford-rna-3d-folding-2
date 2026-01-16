#!/usr/bin/env python3
"""
Run Ensemble Pipeline Validation.

Tests the new ensemble pipeline on validation data and computes TM-scores.
"""
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def load_ground_truth(labels_file: str) -> dict:
    """Load ground truth coordinates from labels file."""
    df = pd.read_csv(labels_file)
    ground_truth = {}
    
    for _, row in df.iterrows():
        id_str = row['ID']
        parts = id_str.rsplit('_', 1)
        target_id = parts[0]
        resid = int(parts[1])
        
        if target_id not in ground_truth:
            ground_truth[target_id] = {}
        
        x = row.get('x_1', np.nan)
        y = row.get('y_1', np.nan)
        z = row.get('z_1', np.nan)
        
        if not (isinstance(x, float) and x < -1e10):
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


def main():
    """Run ensemble pipeline validation."""
    from rna_tbm import EnsemblePipeline, EnsembleConfig, compute_tm_score
    from rna_tbm.config import PipelineConfig, PathConfig
    
    print("=" * 60)
    print("Stanford RNA 3D Folding - Ensemble Pipeline Validation")
    print("=" * 60)
    
    # Set up paths
    paths = PathConfig()
    config = PipelineConfig()
    
    # Find files
    val_seq_file = paths.data_dir / "sequences" / "test_sequences.csv"
    if not val_seq_file.exists():
        val_seq_file = paths.data_dir / "validation_sequences.csv"
    
    val_labels_file = paths.data_dir / "validation_labels.csv"
    
    cache_path = paths.output_dir / "template_db.pkl"
    output_path = paths.output_dir / "ensemble_submission.csv"
    
    print(f"\nConfiguration:")
    print(f"  Sequences: {val_seq_file}")
    print(f"  Labels: {val_labels_file}")
    print(f"  Cache: {cache_path}")
    print(f"  Output: {output_path}")
    
    # Load validation sequences
    if not val_seq_file.exists():
        print(f"\nError: Sequence file not found: {val_seq_file}")
        return 1
    
    sequences_df = pd.read_csv(val_seq_file)
    print(f"\nLoaded {len(sequences_df)} sequences")
    
    # Load ground truth if available
    ground_truth = {}
    if val_labels_file.exists():
        ground_truth = load_ground_truth(str(val_labels_file))
        print(f"Loaded {len(ground_truth)} ground truth structures")
    
    # Create ensemble pipeline
    ensemble_config = EnsembleConfig(
        max_templates=10,
        num_candidates=15,
        num_output_models=5,
        use_ml_distances=True,
        num_refinement_steps=100,
    )
    
    pipeline = EnsemblePipeline(config, ensemble_config)
    
    # Load template database
    print("\nLoading template database...")
    pipeline.load_template_database(cache_path=str(cache_path))
    
    # Run predictions
    print("\nRunning ensemble predictions...")
    start_time = time.time()
    # Process each target
    results = []
    
    # Debug: Only run first 3 targets for rapid verification
    targets_to_run = list(sequences_df.iterrows())[:3]
    
    for idx, row in targets_to_run:
        target_id = row['target_id']
        sequence = row['sequence']
        temporal_cutoff = row.get('temporal_cutoff', '2099-12-31')
        
        gt = ground_truth.get(target_id)
        
        print(f"[{idx+1}/{len(sequences_df)}] {target_id} ({len(sequence)}nt)... ", end="", flush=True)
        
        try:
            # Run prediction
            pred = pipeline.predict_single(
                target_id, sequence, temporal_cutoff,
                ground_truth=gt,
            )
            
            # Use the already computed TM-score from the pipeline if available
            tm = pred.tm_scores[0] if (pred.tm_scores and gt is not None) else None
            conf = pred.confidences[0] if pred.confidences else 0.0
            
            if tm is not None:
                print(f"TM={tm:.3f}, conf={conf:.3f}")
            else:
                print(f"conf={conf:.3f}")
                
            results.append({
                'target_id': target_id,
                'length': len(sequence),
                'tm_score': tm,
                'confidence': conf,
            })
            
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("ENSEMBLE VALIDATION RESULTS")
    print("=" * 60)
    
    valid_results = [r for r in results if r['tm_score'] is not None]
    if valid_results:
        tm_scores = [r['tm_score'] for r in valid_results]
        
        print(f"\nTotal targets: {len(results)}")
        print(f"With ground truth: {len(valid_results)}")
        print(f"Average TM-score: {np.mean(tm_scores):.3f}")
        print(f"Median TM-score: {np.median(tm_scores):.3f}")
        print(f"Min TM-score: {np.min(tm_scores):.3f}")
        print(f"Max TM-score: {np.max(tm_scores):.3f}")
        print(f"\nTargets with TM ≥ 0.8: {sum(1 for t in tm_scores if t >= 0.8)}/{len(tm_scores)}")
        print(f"Targets with TM ≥ 0.5: {sum(1 for t in tm_scores if t >= 0.5)}/{len(tm_scores)}")
        print(f"Targets with TM ≥ 0.3: {sum(1 for t in tm_scores if t >= 0.3)}/{len(tm_scores)}")
        
        # Per-target breakdown
        print("\nPer-target breakdown:")
        print("-" * 50)
        print(f"{'Target':<12} {'Length':>8} {'TM-score':>10} {'Confidence':>10}")
        print("-" * 50)
        for r in sorted(valid_results, key=lambda x: x['tm_score'], reverse=True):
            tm_str = f"{r['tm_score']:.3f}"
            status = "✓" if r['tm_score'] >= 0.5 else "✗"
            print(f"{r['target_id']:<12} {r['length']:>8} {tm_str:>10} {r['confidence']:>10.3f} {status}")
    
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/len(results):.1f}s per target)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
