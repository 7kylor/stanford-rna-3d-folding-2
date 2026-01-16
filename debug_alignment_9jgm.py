
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rna_tbm.template_db import TemplateDB

def debug_9jgm_alignment():
    db_path = "/Users/taher/Projects/stanford-rna-3d-folding-2/output/template_db.pkl"
    db = TemplateDB.load(db_path)
    
    # Load sequences
    seq_df = pd.read_csv("data/sequences/test_sequences.csv")
    sequence = seq_df[seq_df['target_id'] == "9JGM"].iloc[0]['sequence']
    
    print(f"Target Sequence: {sequence[:50]}...")
    
    hits = db.search(sequence, max_hits=5)
    print(f"Found {len(hits)} hits.")
    
    for i, hit in enumerate(hits):
        print(f"Hit {i}: {hit.pdb_id}_{hit.chain_id}, Identity: {hit.identity:.3f}, Coverage: {hit.coverage:.3f}")

if __name__ == "__main__":
    debug_9jgm_alignment()
