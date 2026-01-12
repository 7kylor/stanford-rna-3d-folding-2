#!/usr/bin/env python3
"""
Run TBM pipeline on validation set and generate submission.
This script demonstrates the full pipeline workflow.
"""
import sys
import time
from pathlib import Path

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from rna_tbm import PipelineConfig, PredictionSet, TBMPipeline, write_submission
from rna_tbm.config import PathConfig


def main():
    """Run validation pipeline."""
    print("=" * 60)
    print("Stanford RNA 3D Folding - TBM Validation")
    print("=" * 60)
    
    # Set up paths
    paths = PathConfig()
    
    # Configure for testing (limit files for speed)
    config = PipelineConfig.for_testing()
    
    # Find validation file
    validation_file = paths.validation_sequences_file
    if not validation_file or not validation_file.exists():
        # Try legacy location
        validation_file = PROJECT_ROOT / "validation_sequences.csv"
    
    if not validation_file.exists():
        print(f"Error: Validation file not found")
        print(f"Looked in: {validation_file}")
        return 1
    
    # Set up output
    output_file = PROJECT_ROOT / "validation_submission.csv"
    
    print(f"\nConfiguration:")
    print(f"  PDB directory: {config.paths.pdb_rna_dir}")
    print(f"  Metadata file: {config.paths.release_dates_file}")
    print(f"  Validation file: {validation_file}")
    print(f"  Output file: {output_file}")
    
    # Create and run pipeline
    pipeline = TBMPipeline(config)
    
    try:
        # Load or build database
        print("\n1. Preparing template database...")
        pipeline.load_or_build_database()
        
        # Load validation sequences
        print(f"\n2. Loading validation sequences...")
        val_df = pd.read_csv(validation_file)
        print(f"   Found {len(val_df)} targets")
        
        # Run predictions
        print("\n3. Running predictions...")
        start_time = time.time()
        predictions = pipeline.predict_all(val_df, str(output_file), verbose=True)
        pred_time = time.time() - start_time
        
        # Verify output
        print("\n4. Verifying output...")
        result_df = pd.read_csv(output_file)
        print(f"   Written: {len(result_df)} rows")
        
        # Show sample
        print(f"\n   Sample output:")
        print(result_df.head(3).to_string(index=False))
        
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        print(pipeline.stats.summary())
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
