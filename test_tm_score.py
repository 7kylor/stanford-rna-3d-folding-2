
import numpy as np
from rna_tbm.confidence import compute_tm_score

def test_tm():
    # 30nt random structure
    L = 30
    coords1 = np.random.randn(L, 3) * 10
    
    # 1. Perfect match
    tm1 = compute_tm_score(coords1, coords1)
    print(f"Perfect match: TM = {tm1:.3f}")
    
    # 2. Translated match
    coords2 = coords1 + np.array([100, 200, 300])
    tm2 = compute_tm_score(coords1, coords2)
    print(f"Translated match: TM = {tm2:.3f}")
    
    # 3. Rotated match
    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    coords3 = coords1 @ R
    tm3 = compute_tm_score(coords1, coords3)
    print(f"Rotated match: TM = {tm3:.3f}")
    
    # 4. Mirrored match
    coords4 = coords1.copy()
    coords4[:, 0] *= -1
    tm4 = compute_tm_score(coords1, coords4)
    print(f"Mirrored match: TM = {tm4:.3f}")
    
    # 5. Totally different
    coords5 = np.random.randn(L, 3) * 10
    tm5 = compute_tm_score(coords1, coords5)
    print(f"Different match: TM = {tm5:.3f}")

if __name__ == "__main__":
    test_tm()
