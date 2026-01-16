
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rna_tbm.template_db import TemplateDB
from rna_tbm import EnsemblePipeline, EnsembleConfig
from rna_tbm.config import PipelineConfig

def debug_8znq():
    config = PipelineConfig()
    ensemble_config = EnsembleConfig(max_templates=10)
    pipeline = EnsemblePipeline(config, ensemble_config)
    
    db_path = "/Users/taher/Projects/stanford-rna-3d-folding-2/output/template_db.pkl"
    pipeline.template_db = TemplateDB.load(db_path)
    
    target_id = "8ZNQ"
    seq_df = pd.read_csv("data/sequences/test_sequences.csv")
    row = seq_df[seq_df['target_id'] == target_id].iloc[0]
    sequence = row['sequence']
    temporal_cutoff = "2024-10-30"
    
    # Check template candidates directly
    print("=== Generating Template Candidates ===")
    template_candidates = pipeline._generate_template_candidates(sequence, temporal_cutoff)
    print(f"Template candidates: {len(template_candidates)}")
    for i, tc in enumerate(template_candidates):
        print(f"  Template {i}: min={tc.min():.1f}, max={tc.max():.1f}, range={tc.max()-tc.min():.1f}")
    
    print("\n=== Generating ML Candidates ===")
    ml_candidates = pipeline._generate_ml_candidates(sequence, None)
    print(f"ML candidates: {len(ml_candidates)}")
    for i, mc in enumerate(ml_candidates):
        print(f"  ML {i}: min={mc.min():.1f}, max={mc.max():.1f}, range={mc.max()-mc.min():.1f}")

if __name__ == "__main__":
    debug_8znq()
