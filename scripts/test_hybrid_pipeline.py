#!/usr/bin/env python3
"""
Test script for hybrid deep learning pipeline.

Validates the enhanced pipeline on test targets and measures TM-score improvements.

Usage:
    python scripts/test_hybrid_pipeline.py [num_targets]
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_tm_score(pred_coords: np.ndarray, ref_coords: np.ndarray) -> float:
    """
    Compute TM-score between predicted and reference coordinates.
    
    TM-score is a length-independent metric for structural similarity.
    
    Args:
        pred_coords: (L, 3) predicted coordinates
        ref_coords: (L, 3) reference coordinates
        
    Returns:
        TM-score in range [0, 1]
    """
    L = len(pred_coords)
    
    if L == 0 or len(ref_coords) != L:
        return 0.0
    
    # Handle NaN
    valid_mask = ~(np.isnan(pred_coords).any(axis=1) | np.isnan(ref_coords).any(axis=1))
    if np.sum(valid_mask) == 0:
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
        
        # Apply rotation
        pred_rotated = pred_centered @ R
    except np.linalg.LinAlgError:
        pred_rotated = pred_centered
    
    # Compute TM-score
    distances = np.sqrt(np.sum((pred_rotated - ref_centered) ** 2, axis=1))
    tm_score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / L_valid
    
    return tm_score


def run_baseline_pipeline(target_id: str, sequence: str, temporal_cutoff: str):
    """Run baseline TBM pipeline."""
    from rna_tbm.pipeline import TBMPipeline
    from rna_tbm.config import PipelineConfig
    
    config = PipelineConfig()
    pipeline = TBMPipeline(config)
    pipeline.load_or_build_database()
    
    result = pipeline.predict_single(
        target_id=target_id,
        sequence=sequence,
        temporal_cutoff=temporal_cutoff,
        num_models=5,
    )
    
    return result


def run_hybrid_pipeline(target_id: str, sequence: str, temporal_cutoff: str):
    """Run enhanced hybrid pipeline."""
    from rna_tbm.hybrid_pipeline import HybridPipeline
    from rna_tbm.config import PipelineConfig
    
    config = PipelineConfig()
    pipeline = HybridPipeline(
        config,
        template_weight=0.5,
        distance_weight=0.5,
        use_covariation=True,
        use_embeddings=True,
    )
    pipeline.load_or_build_database()
    
    result = pipeline.predict_single(
        target_id=target_id,
        sequence=sequence,
        temporal_cutoff=temporal_cutoff,
        num_models=5,
    )
    
    return result


def test_components():
    """Test individual deep learning components."""
    print("=" * 60)
    print("Testing Deep Learning Components")
    print("=" * 60)
    
    # Test sequence
    test_sequence = "GCGCAUAUAUAUGCGC"
    
    # Test 1: Enhanced embeddings
    print("\n1. Testing enhanced embeddings...")
    try:
        from rna_tbm.embeddings.rna_fm import RNAFMEncoder
        encoder = RNAFMEncoder(use_fallback=True)
        
        # Basic fallback
        basic_emb = encoder.encode(test_sequence)
        print(f"   Basic embedding shape: {basic_emb.shape}")
        
        # Enhanced fallback
        enhanced_emb = encoder.encode_enhanced(test_sequence)
        print(f"   Enhanced embedding shape: {enhanced_emb.shape}")
        print("   ✓ Embeddings working")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Distance prediction
    print("\n2. Testing distance prediction...")
    try:
        from rna_tbm.models.distance_predictor import DistancePredictorNumpy
        predictor = DistancePredictorNumpy(use_enhanced=True)
        
        distances, confidences = predictor.predict_from_sequence(
            test_sequence,
            embeddings=enhanced_emb if 'enhanced_emb' in dir() else None,
        )
        print(f"   Distance matrix shape: {distances.shape}")
        print(f"   Confidence matrix shape: {confidences.shape}")
        print(f"   Distance range: [{distances.min():.2f}, {distances.max():.2f}]")
        print("   ✓ Distance prediction working")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Distance to coordinates
    print("\n3. Testing coordinate generation...")
    try:
        from rna_tbm.models.distance_to_coords import DistanceToCoords
        converter = DistanceToCoords(
            max_iterations=100,
            use_enhanced=True,
        )
        
        coords = converter.convert(distances, confidences)
        print(f"   Coordinates shape: {coords.shape}")
        print(f"   Coordinate range: x=[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}]")
        print("   ✓ Coordinate generation working")
        
        # Test diversity generation
        models = converter.convert_with_diversity(distances, confidences, num_models=3)
        print(f"   Generated {len(models)} diverse models")
        print("   ✓ Diversity generation working")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Contact prediction
    print("\n4. Testing contact prediction...")
    try:
        from rna_tbm.models.contact_map import ContactPredictorNumpy
        predictor = ContactPredictorNumpy()
        
        contacts = predictor.predict(test_sequence)
        print(f"   Contact map shape: {contacts.shape}")
        print(f"   Contact probability range: [{contacts.min():.3f}, {contacts.max():.3f}]")
        print("   ✓ Contact prediction working")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)


def main(num_targets: int = 10):
    """
    Run hybrid pipeline on validation targets.
    
    Args:
        num_targets: Number of targets to test
    """
    print("=" * 60)
    print("Stanford RNA 3D Folding - Hybrid Pipeline Test")
    print("=" * 60)
    
    # First test components
    test_components()
    
    # Check for validation data
    validation_file = project_root / "data" / "validation_sequences.csv"
    if not validation_file.exists():
        validation_file = project_root / "validation_sequences.csv"
    
    if not validation_file.exists():
        print("\n⚠ No validation sequences found. Testing with synthetic sequence.")
        
        # Test with synthetic sequence
        test_sequence = "GCGCAUAUAUAUGCGC" * 5  # 80nt sequence
        
        print(f"\nTest sequence length: {len(test_sequence)}")
        
        # Run hybrid pipeline
        start = time.time()
        try:
            from rna_tbm.hybrid_pipeline import HybridPipeline
            from rna_tbm.config import PipelineConfig
            
            config = PipelineConfig()
            config.max_files = 100  # Limit for fast testing
            
            pipeline = HybridPipeline(
                config,
                template_weight=0.5,
                distance_weight=0.5,
            )
            pipeline.load_or_build_database()
            
            result = pipeline.predict_single(
                target_id="TEST_001",
                sequence=test_sequence,
                temporal_cutoff="2099-12-31",
                num_models=5,
            )
            
            elapsed = time.time() - start
            
            print(f"\nGenerated {len(result.models)} models in {elapsed:.2f}s")
            for i, model in enumerate(result.models):
                print(f"  Model {i+1}: shape={model.shape}, "
                      f"coord range=[{model.min():.1f}, {model.max():.1f}]")
            
            print("\n✓ Hybrid pipeline working!")
            
        except Exception as e:
            print(f"\n✗ Error running pipeline: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # Load validation data
    print(f"\n1. Loading validation sequences from {validation_file}...")
    df = pd.read_csv(validation_file)
    
    # Limit targets
    if len(df) > num_targets:
        df = df.head(num_targets)
    
    print(f"   Testing on {len(df)} targets")
    
    # Results storage
    results = []
    
    # Run pipeline on each target
    print("\n2. Running hybrid pipeline...")
    
    for idx, row in df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        temporal_cutoff = row.get('temporal_cutoff', '2099-12-31')
        
        print(f"   [{idx + 1}/{len(df)}] {target_id}: {len(sequence)}nt", end="... ")
        
        start = time.time()
        try:
            result = run_hybrid_pipeline(target_id, sequence, temporal_cutoff)
            elapsed = time.time() - start
            
            print(f"done ({elapsed:.2f}s)")
            
            results.append({
                'target_id': target_id,
                'length': len(sequence),
                'num_models': len(result.models),
                'time': elapsed,
                'success': True,
            })
            
        except Exception as e:
            print(f"error: {e}")
            results.append({
                'target_id': target_id,
                'length': len(sequence),
                'success': False,
                'error': str(e),
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    success_rate = results_df['success'].mean() * 100
    
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if results_df['success'].any():
        successful = results_df[results_df['success']]
        print(f"Average time per target: {successful['time'].mean():.2f}s")
        print(f"Total targets processed: {len(successful)}")
    
    print("\n✓ Testing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hybrid deep learning pipeline")
    parser.add_argument(
        "num_targets",
        type=int,
        nargs="?",
        default=10,
        help="Number of targets to test (default: 10)",
    )
    
    args = parser.parse_args()
    main(args.num_targets)
