"""
Confidence Scoring for RNA 3D Structure Predictions.

Estimates model quality without ground truth using:
- Backbone geometry validity (bond lengths, angles)
- Clash detection
- Distance constraint satisfaction
- Predicted LDDT from distance distributions
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class ConfidenceResult:
    """Result of confidence scoring."""
    overall_score: float  # 0-1, higher is better
    geometry_score: float
    clash_score: float
    distance_score: float
    lddt_score: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'overall': self.overall_score,
            'geometry': self.geometry_score,
            'clash': self.clash_score,
            'distance': self.distance_score,
            'lddt': self.lddt_score,
        }


class ConfidenceScorer:
    """
    Score predicted RNA models by structural quality metrics.
    
    Used for ensemble selection without ground truth.
    
    Metrics:
    - Geometry: Bond lengths should be 5.5-6.5Å for C1'-C1'
    - Clash: No atoms closer than 3.0Å (except adjacent)
    - Distance: Satisfaction of predicted distance constraints
    - LDDT: Local Distance Difference Test (without reference)
    """
    
    # RNA C1' geometry parameters
    IDEAL_C1_DISTANCE = 5.9  # Angstroms
    C1_DISTANCE_TOLERANCE = 1.5  # Acceptable deviation
    MIN_NONBONDED_DISTANCE = 3.0  # Clash threshold
    
    def __init__(
        self,
        geometry_weight: float = 0.3,
        clash_weight: float = 0.3,
        distance_weight: float = 0.2,
        lddt_weight: float = 0.2,
    ):
        """
        Initialize scorer with metric weights.
        
        Args:
            geometry_weight: Weight for backbone geometry score
            clash_weight: Weight for clash score
            distance_weight: Weight for distance constraint score
            lddt_weight: Weight for LDDT score
        """
        # Normalize weights
        total = geometry_weight + clash_weight + distance_weight + lddt_weight
        self.geometry_weight = geometry_weight / total
        self.clash_weight = clash_weight / total
        self.distance_weight = distance_weight / total
        self.lddt_weight = lddt_weight / total
    
    def score(
        self,
        coords: np.ndarray,
        distance_constraints: Optional[np.ndarray] = None,
        constraint_confidence: Optional[np.ndarray] = None,
    ) -> ConfidenceResult:
        """
        Compute overall confidence score for a structure.
        
        Args:
            coords: (L, 3) C1' coordinates
            distance_constraints: Optional (L, L) predicted distances
            constraint_confidence: Optional (L, L) confidence for constraints
            
        Returns:
            ConfidenceResult with all scores
        """
        geometry_score = self._score_geometry(coords)
        clash_score = self._score_clashes(coords)
        
        if distance_constraints is not None:
            distance_score = self._score_distance_constraints(
                coords, distance_constraints, constraint_confidence
            )
        else:
            distance_score = 1.0  # No constraints = neutral
            
        lddt_score = self._compute_lddt_proxy(coords)
        
        overall = (
            self.geometry_weight * geometry_score +
            self.clash_weight * clash_score +
            self.distance_weight * distance_score +
            self.lddt_weight * lddt_score
        )
        
        return ConfidenceResult(
            overall_score=overall,
            geometry_score=geometry_score,
            clash_score=clash_score,
            distance_score=distance_score,
            lddt_score=lddt_score,
        )
    
    def score_batch(
        self,
        coords_list: List[np.ndarray],
        distance_constraints: Optional[np.ndarray] = None,
        constraint_confidence: Optional[np.ndarray] = None,
    ) -> List[ConfidenceResult]:
        """Score multiple models."""
        return [
            self.score(coords, distance_constraints, constraint_confidence)
            for coords in coords_list
        ]
    
    def rank_models(
        self,
        coords_list: List[np.ndarray],
        distance_constraints: Optional[np.ndarray] = None,
        constraint_confidence: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, ConfidenceResult]]:
        """
        Rank models by confidence score.
        
        Returns:
            List of (original_index, ConfidenceResult) sorted by score descending
        """
        results = self.score_batch(coords_list, distance_constraints, constraint_confidence)
        indexed = list(enumerate(results))
        indexed.sort(key=lambda x: x[1].overall_score, reverse=True)
        return indexed
    
    def select_diverse_top_k(
        self,
        coords_list: List[np.ndarray],
        k: int = 5,
        distance_constraints: Optional[np.ndarray] = None,
        constraint_confidence: Optional[np.ndarray] = None,
        diversity_threshold: float = 2.0,  # Min RMSD between selected
    ) -> List[int]:
        """
        Select top-k diverse models.
        
        Uses greedy selection: pick best model, then best remaining
        that is sufficiently different from already selected.
        
        Args:
            coords_list: List of coordinate arrays
            k: Number of models to select
            distance_constraints: Optional constraints for scoring
            constraint_confidence: Optional confidence values
            diversity_threshold: Minimum RMSD between selected models
            
        Returns:
            Indices of selected models
        """
        if len(coords_list) <= k:
            return list(range(len(coords_list)))
        
        ranked = self.rank_models(coords_list, distance_constraints, constraint_confidence)
        
        selected = [ranked[0][0]]  # Best model
        
        for idx, result in ranked[1:]:
            if len(selected) >= k:
                break
            
            # Check diversity against all selected
            is_diverse = True
            for sel_idx in selected:
                rmsd = self._compute_rmsd(coords_list[idx], coords_list[sel_idx])
                if rmsd < diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(idx)
        
        # If not enough diverse models, fill with next best
        for idx, result in ranked:
            if len(selected) >= k:
                break
            if idx not in selected:
                selected.append(idx)
        
        return selected[:k]
    
    def _score_geometry(self, coords: np.ndarray) -> float:
        """
        Score backbone geometry based on C1'-C1' distances.
        
        Ideal consecutive C1' distance is ~5.9Å for RNA.
        """
        L = len(coords)
        if L < 2:
            return 1.0
        
        # Check for NaN
        if np.any(np.isnan(coords)):
            return 0.0
        
        # Consecutive distances
        consecutive_dists = np.linalg.norm(
            coords[1:] - coords[:-1], axis=1
        )
        
        # Score: how close to ideal distance
        deviations = np.abs(consecutive_dists - self.IDEAL_C1_DISTANCE)
        normalized_dev = deviations / self.C1_DISTANCE_TOLERANCE
        
        # Sigmoid-like scoring: small deviation = high score
        scores = np.exp(-normalized_dev ** 2)
        
        return float(np.mean(scores))
    
    def _score_clashes(self, coords: np.ndarray) -> float:
        """
        Score based on absence of steric clashes.
        
        Non-adjacent residues should be > 3.0Å apart.
        """
        L = len(coords)
        if L < 4:
            return 1.0
        
        if np.any(np.isnan(coords)):
            return 0.0
        
        # Compute all pairwise distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        
        # Only check non-adjacent pairs (|i-j| >= 3)
        clash_count = 0
        total_pairs = 0
        
        for i in range(L):
            for j in range(i + 3, L):
                total_pairs += 1
                if distances[i, j] < self.MIN_NONBONDED_DISTANCE:
                    clash_count += 1
        
        if total_pairs == 0:
            return 1.0
        
        # Score: fraction of non-clashing pairs
        clash_fraction = clash_count / total_pairs
        return max(0.0, 1.0 - clash_fraction * 5)  # Penalize clashes heavily
    
    def _score_distance_constraints(
        self,
        coords: np.ndarray,
        constraints: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> float:
        """
        Score satisfaction of distance constraints.
        
        Args:
            coords: (L, 3) coordinates
            constraints: (L, L) predicted distances
            confidence: (L, L) confidence for each constraint
        """
        L = len(coords)
        
        if np.any(np.isnan(coords)):
            return 0.0
        
        # Compute actual distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        actual = np.linalg.norm(diff, axis=2)
        
        # Compute violations
        violations = np.abs(actual - constraints)
        
        if confidence is None:
            confidence = np.ones_like(constraints)
        
        # Only count pairs with separation >= 4 and confidence > 0.5
        mask = np.zeros((L, L), dtype=bool)
        for i in range(L):
            for j in range(i + 4, L):
                if confidence[i, j] > 0.5:
                    mask[i, j] = True
                    mask[j, i] = True
        
        if not np.any(mask):
            return 1.0
        
        # Weighted average violation
        weighted_violations = violations[mask] * confidence[mask]
        mean_violation = np.sum(weighted_violations) / np.sum(confidence[mask])
        
        # Convert to score (violations in Angstroms)
        # 0Å violation = 1.0, 5Å violation = ~0.5, 10Å = ~0.1
        score = np.exp(-mean_violation / 5.0)
        
        return float(score)
    
    def _compute_lddt_proxy(self, coords: np.ndarray) -> float:
        """
        Compute LDDT-like score without reference.
        
        Uses internal consistency: local neighborhoods should have
        similar distance patterns to expected RNA geometry.
        """
        L = len(coords)
        if L < 5:
            return 0.5  # Neutral for short sequences
        
        if np.any(np.isnan(coords)):
            return 0.0
        
        # Compute local distance consistency
        # For each residue, check 4 neighbors on each side
        window = 4
        scores = []
        
        for i in range(window, L - window):
            local_coords = coords[i-window:i+window+1]
            
            # Expected distances for A-form helix
            expected = []
            actual_dists = []
            for j in range(len(local_coords)):
                for k in range(j + 1, len(local_coords)):
                    sep = k - j
                    # Expected C1'-C1' distance based on separation
                    exp_dist = sep * self.IDEAL_C1_DISTANCE * 0.8  # Compaction factor
                    expected.append(exp_dist)
                    actual_dists.append(np.linalg.norm(local_coords[j] - local_coords[k]))
            
            expected = np.array(expected)
            actual_dists = np.array(actual_dists)
            
            # Fraction within tolerance
            tolerance = 2.0
            within = np.sum(np.abs(actual_dists - expected) < tolerance) / len(expected)
            scores.append(within)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _compute_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Compute RMSD between two coordinate sets after alignment."""
        if len(coords1) != len(coords2):
            return float('inf')
        
        # Handle NaN
        valid = ~(np.isnan(coords1).any(axis=1) | np.isnan(coords2).any(axis=1))
        if np.sum(valid) < 3:
            return float('inf')
        
        c1 = coords1[valid]
        c2 = coords2[valid]
        
        # Center
        c1_centered = c1 - c1.mean(axis=0)
        c2_centered = c2 - c2.mean(axis=0)
        
        # Kabsch alignment
        try:
            H = c1_centered.T @ c2_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            c1_rotated = c1_centered @ R
        except np.linalg.LinAlgError:
            c1_rotated = c1_centered
        
        # RMSD
        diff = c1_rotated - c2_centered
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        
        return float(rmsd)


def compute_tm_score(pred_coords: np.ndarray, ref_coords: np.ndarray) -> float:
    """
    Compute TM-score between predicted and reference coordinates.
    
    TM-score is length-normalized and ranges from 0 to 1.
    Scores > 0.5 indicate similar fold, > 0.8 indicate high similarity.
    
    Args:
        pred_coords: (L, 3) predicted C1' coordinates
        ref_coords: (L, 3) reference C1' coordinates
        
    Returns:
        TM-score value
    """
    L = len(pred_coords)
    
    if L == 0 or len(ref_coords) != L:
        return 0.0
    
    # Handle NaN
    valid_mask = ~(np.isnan(pred_coords).any(axis=1) | np.isnan(ref_coords).any(axis=1))
    if np.sum(valid_mask) < 3:
        return 0.0
    
    pred = pred_coords[valid_mask]
    ref = ref_coords[valid_mask]
    L_valid = len(pred)
    
    # d0 normalization factor
    d0 = max(0.5, 1.24 * (L_valid - 15) ** (1/3) - 1.8)
    
    # Center structures
    pred_centered = pred - pred.mean(axis=0)
    ref_centered = ref - ref.mean(axis=0)
    
    # Kabsch algorithm for optimal rotation
    try:
        # Optimal rotation matrix R such that pred @ R ≈ ref
        H = pred_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        
        # Standard Kabsch
        R = U @ Vt
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
            
        pred_rotated = pred_centered @ R
    except np.linalg.LinAlgError:
        pred_rotated = pred_centered
    
    # Compute TM-score
    distances = np.sqrt(np.sum((pred_rotated - ref_centered) ** 2, axis=1))
    tm_score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / L_valid
    
    return float(tm_score)


def batch_tm_scores(
    pred_coords_list: List[np.ndarray],
    ref_coords: np.ndarray,
) -> List[float]:
    """Compute TM-scores for multiple predictions against single reference."""
    return [compute_tm_score(pred, ref_coords) for pred in pred_coords_list]
