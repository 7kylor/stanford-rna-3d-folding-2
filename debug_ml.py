
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rna_tbm.models.transformer_distance import predict_distances
from rna_tbm.models.distance_geometry import DistanceGeometrySolver

def debug_ml_distances():
    sequence = "ACCGUGACGGGCCUUUUGGCUAUACGCGGU"  # 8ZNQ, 30nt
    
    print("=== Distance Prediction ===")
    distances, confidence = predict_distances(sequence)
    
    print(f"Distance matrix: min={distances.min():.1f}, max={distances.max():.1f}, mean={distances.mean():.1f}")
    print(f"Confidence: min={confidence.min():.3f}, max={confidence.max():.3f}, mean={confidence.mean():.3f}")
    
    print("\n=== MDS Embedding ===")
    solver = DistanceGeometrySolver()
    coords = solver.mds_embed(distances)
    print(f"MDS coords: min={coords.min():.1f}, max={coords.max():.1f}, range={coords.max()-coords.min():.1f}")
    
    print("\n=== Full Solve ===")
    coords2, loss = solver.solve(distances, confidence)
    print(f"Solved coords: min={coords2.min():.1f}, max={coords2.max():.1f}, range={coords2.max()-coords2.min():.1f}")
    print(f"Final loss: {loss:.2f}")

if __name__ == "__main__":
    debug_ml_distances()
