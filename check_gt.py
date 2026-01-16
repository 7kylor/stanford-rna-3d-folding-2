
import pandas as pd
import numpy as np

def check_gt_distances():
    df = pd.read_csv("data/validation_labels.csv")
    target_id = "8ZNQ"
    gt_coords = []
    for i in range(1, 31):
        res_row = df[df['ID'] == f"{target_id}_{i}"]
        gt_coords.append([res_row.iloc[0]['x_1'], res_row.iloc[0]['y_1'], res_row.iloc[0]['z_1']])
    
    gt_coords = np.array(gt_coords)
    diffs = np.linalg.norm(np.diff(gt_coords, axis=0), axis=1)
    print(f"GT Consecutive distances: {diffs}")
    print(f"Mean: {np.mean(diffs):.2f}, Max: {np.max(diffs):.2f}, Min: {np.min(diffs):.2f}")

if __name__ == "__main__":
    check_gt_distances()
