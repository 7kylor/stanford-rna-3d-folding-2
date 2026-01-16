
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rna_tbm.template_db import TemplateDB

def debug_db_coords():
    db_path = "/Users/taher/Projects/stanford-rna-3d-folding-2/output/template_db.pkl"
    db = TemplateDB.load(db_path)
    
    pdb_id = "8ZNQ" # Or try another one
    hit = None
    # Find 8ZNQ or anything
    for p_id in db.templates:
        if p_id == pdb_id:
            for chain_id, coords in db.templates[p_id].items():
                print(f"Found {p_id}_{chain_id}")
                res0 = coords.residues[0]
                print(f"Residue 1: {res0.x}, {res0.y}, {res0.z}")
                # Print last residue
                resL = coords.residues[-1]
                print(f"Residue {len(coords.residues)}: {resL.x}, {resL.y}, {resL.z}")
                
                # Check for NaNs
                coords_array = np.array([(r.x, r.y, r.z) for r in coords.residues])
                print(f"Mean distance from centroid: {np.mean(np.sqrt(np.sum((coords_array - np.mean(coords_array, axis=0))**2, axis=1)))}")
                break

if __name__ == "__main__":
    debug_db_coords()
