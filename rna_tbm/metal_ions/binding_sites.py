"""
High-level metal binding site prediction interface.
Combines model-based and heuristic approaches.
"""

import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from .mgnet import (
    TORCH_AVAILABLE,
    build_graph_from_coords,
    MgBindingMotifs,
    TOTAL_NODE_FEATURES,
)


class MetalBindingPredictor:
    """
    Predict metal ion binding sites in RNA structures.
    Uses MgNet model if available, otherwise falls back to heuristics.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained MgNet weights
            threshold: Probability threshold for binding prediction
            device: Device for inference
        """
        self.threshold = threshold
        self.device = device
        self._model = None
        
        # Try to load model
        if TORCH_AVAILABLE and model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained MgNet model."""
        import torch
        from .mgnet import MgNetPredictor
        
        self._model = MgNetPredictor()
        self._model.load_state_dict(torch.load(model_path, map_location=self.device))
        self._model.to(self.device)
        self._model.eval()
    
    def predict(
        self,
        coords: np.ndarray,
        sequence: str,
        return_probabilities: bool = False,
    ) -> List[int]:
        """
        Predict Mg²⁺ binding sites.
        
        Args:
            coords: (L, 3) C1' coordinates
            sequence: RNA sequence
            return_probabilities: If True, return (sites, probs)
            
        Returns:
            List of residue indices with predicted binding sites
            If return_probabilities=True, also returns probabilities array
        """
        # Get binding probabilities
        probs = self._get_probabilities(coords, sequence)
        
        # Threshold to get binding sites
        binding_sites = [i for i, p in enumerate(probs) if p > self.threshold]
        
        if return_probabilities:
            return binding_sites, probs
        return binding_sites
    
    def _get_probabilities(
        self,
        coords: np.ndarray,
        sequence: str,
    ) -> np.ndarray:
        """Get binding probabilities for all positions."""
        if self._model is not None:
            return self._predict_with_model(coords, sequence)
        else:
            return self._predict_with_heuristics(coords, sequence)
    
    def _predict_with_model(
        self,
        coords: np.ndarray,
        sequence: str,
    ) -> np.ndarray:
        """Predict using trained model."""
        import torch
        
        # Build graph
        graph_data = build_graph_from_coords(coords, sequence)
        
        # Check if torch_geometric is available
        try:
            from torch_geometric.data import Data
            
            data = Data(
                x=torch.from_numpy(graph_data['node_features']).to(self.device),
                edge_index=torch.from_numpy(graph_data['edge_index']).to(self.device),
            )
            
            with torch.no_grad():
                probs = self._model(data)
            
            return probs.cpu().numpy()
            
        except ImportError:
            # Fallback: use node features only
            x = torch.from_numpy(graph_data['node_features']).to(self.device)
            
            with torch.no_grad():
                probs = self._model(x)
            
            return probs.cpu().numpy()
    
    def _predict_with_heuristics(
        self,
        coords: np.ndarray,
        sequence: str,
    ) -> np.ndarray:
        """Predict using heuristic rules."""
        L = len(sequence)
        probs = np.zeros(L)
        
        for i in range(L):
            probs[i] = MgBindingMotifs.score_binding_potential(
                sequence, coords, i
            )
        
        return probs
    
    def predict_top_k(
        self,
        coords: np.ndarray,
        sequence: str,
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Predict top k binding sites.
        
        Args:
            coords: C1' coordinates
            sequence: RNA sequence
            k: Number of sites to return
            
        Returns:
            List of (residue_index, probability) tuples
        """
        probs = self._get_probabilities(coords, sequence)
        
        # Get top k indices
        top_k_indices = np.argsort(probs)[-k:][::-1]
        
        return [(int(i), float(probs[i])) for i in top_k_indices]
    
    def visualize_predictions(
        self,
        coords: np.ndarray,
        sequence: str,
    ) -> str:
        """
        Generate ASCII visualization of binding predictions.
        
        Returns:
            String visualization
        """
        probs = self._get_probabilities(coords, sequence)
        
        lines = []
        lines.append("Mg²⁺ Binding Site Predictions")
        lines.append("=" * 50)
        lines.append(f"Sequence: {sequence}")
        lines.append("")
        
        # Probability bar chart
        lines.append("Binding probability per position:")
        for i, (nt, prob) in enumerate(zip(sequence, probs)):
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            marker = " *" if prob > self.threshold else ""
            lines.append(f"{i+1:4d} {nt} [{bar}] {prob:.2f}{marker}")
        
        # Summary
        sites = [i for i, p in enumerate(probs) if p > self.threshold]
        lines.append("")
        lines.append(f"Predicted binding sites: {len(sites)}")
        if sites:
            lines.append(f"Positions: {', '.join(str(s+1) for s in sites)}")
        
        return "\n".join(lines)


class BindingSiteValidator:
    """
    Validate predicted binding sites against experimental data.
    """
    
    @staticmethod
    def validate_against_pdb(
        predicted_sites: List[int],
        pdb_metal_coords: np.ndarray,
        rna_coords: np.ndarray,
        distance_threshold: float = 5.0,
    ) -> dict:
        """
        Validate predictions against PDB metal positions.
        
        Args:
            predicted_sites: Predicted binding residue indices
            pdb_metal_coords: (M, 3) coordinates of metal ions from PDB
            rna_coords: (L, 3) C1' coordinates
            distance_threshold: Max distance to consider a match
            
        Returns:
            Validation metrics
        """
        if len(pdb_metal_coords) == 0:
            return {
                'true_positives': 0,
                'false_positives': len(predicted_sites),
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }
        
        # Find true binding sites (residues close to metals)
        true_sites = set()
        for metal_coord in pdb_metal_coords:
            distances = np.linalg.norm(rna_coords - metal_coord, axis=1)
            close_residues = np.where(distances < distance_threshold)[0]
            true_sites.update(close_residues)
        
        # Calculate metrics
        predicted_set = set(predicted_sites)
        
        true_positives = len(predicted_set & true_sites)
        false_positives = len(predicted_set - true_sites)
        false_negatives = len(true_sites - predicted_set)
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_sites': list(true_sites),
        }
