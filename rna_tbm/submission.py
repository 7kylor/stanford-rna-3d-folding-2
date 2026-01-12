"""
Submission file generation for Stanford RNA 3D Folding competition.
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .alignment import CoordinateTransferResult
from .gap_filling import clip_coordinates


@dataclass
class PredictionSet:
    """Complete prediction for one target (5 models)."""
    target_id: str
    sequence: str
    # List of 5 coordinate arrays, each shape (N, 3)
    models: List[np.ndarray]
    
    def __post_init__(self):
        if len(self.models) != 5:
            raise ValueError(f"Must have exactly 5 models, got {len(self.models)}")
        for i, model in enumerate(self.models):
            expected_shape = (len(self.sequence), 3)
            if model.shape != expected_shape:
                raise ValueError(
                    f"Model {i} shape mismatch: {model.shape} vs {expected_shape}"
                )


def create_submission_rows(prediction: PredictionSet) -> List[Dict]:
    """
    Convert a PredictionSet to submission CSV rows.
    
    Args:
        prediction: PredictionSet with 5 models
        
    Returns:
        List of dicts, one per residue
    """
    rows = []
    n = len(prediction.sequence)
    
    for i in range(n):
        resid = i + 1  # 1-based
        resname = prediction.sequence[i]
        
        row = {
            'ID': f"{prediction.target_id}_{resid}",
            'resname': resname,
            'resid': resid,
        }
        
        # Add coordinates for each model
        for model_idx in range(5):
            coords = prediction.models[model_idx][i]
            coords = np.clip(coords, -999.999, 9999.999)
            row[f'x_{model_idx + 1}'] = round(float(coords[0]), 3)
            row[f'y_{model_idx + 1}'] = round(float(coords[1]), 3)
            row[f'z_{model_idx + 1}'] = round(float(coords[2]), 3)
        
        rows.append(row)
    
    return rows


def write_submission(
    predictions: List[PredictionSet],
    output_path: str
):
    """
    Write submission.csv file.
    
    Args:
        predictions: List of PredictionSet, one per target
        output_path: Path to output CSV
    """
    all_rows = []
    for pred in predictions:
        all_rows.extend(create_submission_rows(pred))
    
    # Create DataFrame with correct column order
    columns = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


def diversify_models(
    base_coords: np.ndarray,
    num_models: int = 5,
    perturbation_scale: float = 0.1
) -> List[np.ndarray]:
    """
    Create diverse models from a base prediction.
    
    For TBM, we want diversity to capture uncertainty without
    degrading good template predictions.
    
    Args:
        base_coords: Shape (N, 3) base coordinates
        num_models: Number of models to generate
        perturbation_scale: Scale of random perturbations (Å)
        
    Returns:
        List of num_models coordinate arrays
    """
    models = [base_coords.copy()]  # First model is unperturbed
    
    for i in range(1, num_models):
        perturbed = base_coords.copy()
        # Add small Gaussian noise
        noise = np.random.randn(*base_coords.shape) * perturbation_scale * (i / num_models)
        perturbed += noise
        models.append(perturbed)
    
    return models


def create_diverse_predictions(
    target_id: str,
    sequence: str,
    transfer_results: List[CoordinateTransferResult],
    num_models: int = 5
) -> PredictionSet:
    """
    Create diverse 5-model prediction from multiple template transfers.
    
    Strategy:
    - If we have multiple good templates, use them as different models
    - If we have fewer templates, diversify with perturbations
    - Model 1 is always the best template (unperturbed)
    
    Args:
        target_id: Target identifier
        sequence: Target sequence
        transfer_results: List of coordinate transfer results from different templates
        num_models: Number of models to generate (default 5)
        
    Returns:
        PredictionSet with 5 diverse models
    """
    models = []
    
    # Sort templates by mapped_fraction (quality)
    sorted_transfers = sorted(transfer_results, key=lambda t: -t.mapped_fraction)
    
    if len(sorted_transfers) >= num_models:
        # Use top templates directly
        for i in range(num_models):
            models.append(clip_coordinates(sorted_transfers[i].coords))
    else:
        # Use all templates, then fill with perturbations
        for transfer in sorted_transfers:
            models.append(clip_coordinates(transfer.coords))
        
        # Fill remaining with perturbations of best template
        base = sorted_transfers[0].coords
        remaining = num_models - len(models)
        if remaining > 0:
            perturbations = diversify_models(base, remaining + 1, perturbation_scale=0.3)
            for p in perturbations[1:]:  # Skip first (unperturbed)
                models.append(clip_coordinates(p))
    
    # Ensure exactly 5 models
    while len(models) < num_models:
        models.append(models[-1].copy())
    models = models[:num_models]
    
    return PredictionSet(
        target_id=target_id,
        sequence=sequence,
        models=models
    )


def load_test_sequences(csv_path: str) -> pd.DataFrame:
    """Load test_sequences.csv."""
    return pd.read_csv(csv_path)


def parse_stoichiometry(stoich_str: str) -> List[tuple]:
    """
    Parse stoichiometry string like 'A:2;B:1' into list of (chain, count).
    
    Returns:
        List of (chain_id, count) tuples
    """
    if not stoich_str or pd.isna(stoich_str):
        return []
    
    result = []
    for part in stoich_str.split(';'):
        if ':' in part:
            chain, count = part.split(':')
            result.append((chain.strip(), int(count.strip())))
    return result
