"""
Distance Prediction Neural Network.
Predicts inter-residue distances from sequence and MSA features.
Follows the distogram approach used in AlphaFold/RoseTTAFold.
"""

import numpy as np
from typing import Optional, Tuple, List

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Distance bins (Angstroms)
DISTANCE_BINS = np.linspace(2.0, 22.0, 64)
NUM_BINS = len(DISTANCE_BINS) - 1


if TORCH_AVAILABLE:
    class ResBlock2D(nn.Module):
        """2D Residual block for distance map processing."""
        
        def __init__(self, channels: int, dilation: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                channels, channels, 3, 
                padding=dilation, dilation=dilation
            )
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)
            self.activation = nn.ELU(inplace=True)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.activation(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return self.activation(x + residual)
    
    
    class DistancePredictor(nn.Module):
        """
        ResNet-style network for distance prediction.
        
        Takes sequence features and predicts distance probability distributions.
        """
        
        def __init__(
            self,
            input_dim: int = 128,
            hidden_dim: int = 128,
            num_blocks: int = 32,
            num_bins: int = NUM_BINS,
        ):
            """
            Args:
                input_dim: Input feature dimension per position pair
                hidden_dim: Hidden channel dimension  
                num_blocks: Number of residual blocks
                num_bins: Number of distance bins for output
            """
            super().__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_bins = num_bins
            
            # Input projection
            self.input_proj = nn.Conv2d(input_dim, hidden_dim, 1)
            
            # Residual blocks with varying dilation
            self.blocks = nn.ModuleList([
                ResBlock2D(hidden_dim, dilation=(i % 4) + 1)
                for i in range(num_blocks)
            ])
            
            # Output projection to distance bins
            self.output_proj = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ELU(inplace=True),
                nn.Conv2d(hidden_dim, num_bins, 1),
            )
        
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            """
            Predict distance distributions.
            
            Args:
                features: (B, L, L, input_dim) pairwise features
                
            Returns:
                logits: (B, L, L, num_bins) distance bin logits
            """
            # (B, L, L, C) -> (B, C, L, L)
            x = features.permute(0, 3, 1, 2)
            
            x = self.input_proj(x)
            
            for block in self.blocks:
                x = block(x)
            
            x = self.output_proj(x)
            
            # (B, C, L, L) -> (B, L, L, C)
            return x.permute(0, 2, 3, 1)
        
        def predict_distances(
            self, 
            features: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Predict expected distances and uncertainties.
            
            Returns:
                distances: (B, L, L) expected distances
                uncertainties: (B, L, L) prediction uncertainties
            """
            logits = self.forward(features)
            probs = F.softmax(logits, dim=-1)
            
            # Bin centers
            bin_centers = torch.tensor(
                (DISTANCE_BINS[:-1] + DISTANCE_BINS[1:]) / 2,
                device=features.device,
                dtype=features.dtype
            )
            
            # Expected distance
            distances = (probs * bin_centers).sum(dim=-1)
            
            # Uncertainty (variance)
            variance = (probs * (bin_centers - distances.unsqueeze(-1)) ** 2).sum(dim=-1)
            uncertainties = torch.sqrt(variance)
            
            return distances, uncertainties


class DistancePredictorNumpy:
    """
    NumPy-based distance predictor for inference without PyTorch.
    Uses simpler heuristics based on sequence and secondary structure.
    
    Enhanced with:
    - Secondary structure prediction with pseudoknot awareness
    - Improved stem detection algorithm
    - Better integration with covariation and embeddings
    """
    
    # Average C1' distances for base pairs (Angstroms)
    CANONICAL_DISTANCES = {
        ('A', 'U'): 5.4,
        ('U', 'A'): 5.4,
        ('G', 'C'): 5.3,
        ('C', 'G'): 5.3,
        ('G', 'U'): 5.5,
        ('U', 'G'): 5.5,
    }
    
    # Average backbone distances
    BACKBONE_DISTANCE = 5.9  # Average C1'-C1' for adjacent residues
    
    # Structural element distances (typical ranges)
    LOOP_DISTANCE = 12.0    # Typical loop residue distance
    HELIX_DISTANCE = 5.4    # Base pair distance in helix
    STACK_DISTANCE = 3.4    # Base stacking distance
    
    def __init__(self, use_enhanced: bool = True):
        """
        Initialize predictor.
        
        Args:
            use_enhanced: Use enhanced prediction features
        """
        self.use_enhanced = use_enhanced
    
    def predict_from_sequence(
        self, 
        sequence: str,
        covariation_scores: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict distance map from sequence.
        
        Args:
            sequence: RNA sequence
            covariation_scores: Optional (L, L) covariation matrix
            embeddings: Optional (L, D) per-residue embeddings
            
        Returns:
            distances: (L, L) predicted distances
            confidences: (L, L) prediction confidences
        """
        L = len(sequence)
        distances = np.full((L, L), 20.0)  # Default to far distance
        confidences = np.full((L, L), 0.1)  # Low default confidence
        
        # Diagonal (same residue)
        np.fill_diagonal(distances, 0.0)
        np.fill_diagonal(confidences, 1.0)
        
        # Adjacent residues (backbone constraint)
        for i in range(L - 1):
            distances[i, i + 1] = self.BACKBONE_DISTANCE
            distances[i + 1, i] = self.BACKBONE_DISTANCE
            confidences[i, i + 1] = 0.95
            confidences[i + 1, i] = 0.95
        
        # i+2 neighbors (short-range constraint)
        for i in range(L - 2):
            distances[i, i + 2] = 9.0
            distances[i + 2, i] = 9.0
            confidences[i, i + 2] = 0.8
            confidences[i + 2, i] = 0.8
        
        # i+3 neighbors
        for i in range(L - 3):
            distances[i, i + 3] = 11.5
            distances[i + 3, i] = 11.5
            confidences[i, i + 3] = 0.6
            confidences[i + 3, i] = 0.6
        
        # Use covariation for base pair predictions
        if covariation_scores is not None:
            self._apply_covariation_constraints(
                sequence, distances, confidences, covariation_scores
            )
        
        # Use embeddings for distance estimation
        if embeddings is not None and self.use_enhanced:
            self._apply_embedding_constraints(
                distances, confidences, embeddings
            )
        
        # Predict secondary structure and add constraints
        self._add_secondary_structure_constraints(
            sequence, distances, confidences
        )
        
        # Detect and add pseudoknot constraints
        if self.use_enhanced:
            self._add_pseudoknot_constraints(
                sequence, distances, confidences, covariation_scores
            )
        
        return distances, confidences
    
    def predict_with_ss(
        self,
        sequence: str,
        secondary_structure: str,
        covariation_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict distances using explicit secondary structure.
        
        Args:
            sequence: RNA sequence
            secondary_structure: Dot-bracket notation (e.g., "(((...)))")
            covariation_scores: Optional covariation matrix
            
        Returns:
            distances: (L, L) predicted distances
            confidences: (L, L) prediction confidences
        """
        L = len(sequence)
        distances = np.full((L, L), 20.0)
        confidences = np.full((L, L), 0.1)
        
        # Basic constraints
        np.fill_diagonal(distances, 0.0)
        np.fill_diagonal(confidences, 1.0)
        
        for i in range(L - 1):
            distances[i, i + 1] = self.BACKBONE_DISTANCE
            distances[i + 1, i] = self.BACKBONE_DISTANCE
            confidences[i, i + 1] = 0.95
            confidences[i + 1, i] = 0.95
        
        # Parse secondary structure
        base_pairs = self._parse_dot_bracket(secondary_structure)
        
        # Add base pair constraints
        for i, j in base_pairs:
            nt_i = sequence[i].upper()
            nt_j = sequence[j].upper()
            pair = (nt_i, nt_j)
            
            if pair in self.CANONICAL_DISTANCES:
                distances[i, j] = self.CANONICAL_DISTANCES[pair]
                distances[j, i] = distances[i, j]
                confidences[i, j] = 0.9
                confidences[j, i] = 0.9
            else:
                # Non-canonical pair
                distances[i, j] = 6.0
                distances[j, i] = 6.0
                confidences[i, j] = 0.7
                confidences[j, i] = 0.7
        
        # Add loop residue constraints
        self._add_loop_constraints(
            secondary_structure, distances, confidences
        )
        
        # Apply covariation
        if covariation_scores is not None:
            self._apply_covariation_constraints(
                sequence, distances, confidences, covariation_scores
            )
        
        return distances, confidences
    
    def _parse_dot_bracket(self, ss: str) -> List[Tuple[int, int]]:
        """Parse dot-bracket notation to base pairs."""
        pairs = []
        stack = []
        
        for i, char in enumerate(ss):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    pairs.append((j, i))
        
        return pairs
    
    def _apply_covariation_constraints(
        self,
        sequence: str,
        distances: np.ndarray,
        confidences: np.ndarray,
        covariation_scores: np.ndarray,
    ):
        """Apply constraints from covariation analysis."""
        L = len(sequence)
        
        # Find likely base pairs (high covariation)
        # Use multiple thresholds for graded confidence
        thresholds = [
            (99, 0.9, 1.0),   # Top 1% - very high confidence
            (95, 0.8, 0.9),   # Top 5% - high confidence
            (90, 0.7, 0.7),   # Top 10% - medium confidence
        ]
        
        for percentile, conf_base, distance_mult in thresholds:
            threshold = np.percentile(covariation_scores, percentile)
            
            for i in range(L):
                for j in range(i + 4, L):  # Minimum loop size
                    if covariation_scores[i, j] > threshold:
                        nt_i = sequence[i].upper()
                        nt_j = sequence[j].upper()
                        pair = (nt_i, nt_j)
                        
                        # Only update if better confidence
                        if confidences[i, j] < conf_base:
                            if pair in self.CANONICAL_DISTANCES:
                                distances[i, j] = self.CANONICAL_DISTANCES[pair] * distance_mult
                                distances[j, i] = distances[i, j]
                            else:
                                # Possible non-canonical pair
                                distances[i, j] = 7.0 * distance_mult
                                distances[j, i] = distances[i, j]
                            
                            # Scale confidence by covariation strength
                            cov_strength = (covariation_scores[i, j] - threshold) / (
                                np.max(covariation_scores) - threshold + 1e-8
                            )
                            confidences[i, j] = conf_base + 0.1 * cov_strength
                            confidences[j, i] = confidences[i, j]
    
    def _apply_embedding_constraints(
        self,
        distances: np.ndarray,
        confidences: np.ndarray,
        embeddings: np.ndarray,
    ):
        """Apply distance estimates from embedding similarity."""
        L = embeddings.shape[0]
        
        # Compute pairwise embedding similarity
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity = emb_norm @ emb_norm.T
        
        # High similarity suggests structural proximity
        for i in range(L):
            for j in range(i + 5, L):  # Skip nearby residues
                sim = similarity[i, j]
                
                # Only use for low-confidence predictions
                if confidences[i, j] < 0.3 and sim > 0.8:
                    # High similarity -> closer distance
                    estimated_dist = 15.0 - 10.0 * (sim - 0.8) / 0.2
                    estimated_dist = max(8.0, min(15.0, estimated_dist))
                    
                    if estimated_dist < distances[i, j]:
                        distances[i, j] = estimated_dist
                        distances[j, i] = estimated_dist
                        confidences[i, j] = 0.3 + 0.2 * (sim - 0.8) / 0.2
                        confidences[j, i] = confidences[i, j]
    
    def _add_secondary_structure_constraints(
        self,
        sequence: str,
        distances: np.ndarray,
        confidences: np.ndarray,
    ):
        """Add constraints from predicted secondary structure."""
        L = len(sequence)
        
        # Simple base pair prediction based on Watson-Crick complementarity
        # Look for potential stems
        for i in range(L):
            for j in range(i + 4, L):
                nt_i = sequence[i].upper()
                nt_j = sequence[j].upper()
                
                # Check if canonical base pair
                if ((nt_i, nt_j) in self.CANONICAL_DISTANCES and 
                    confidences[i, j] < 0.5):  # Don't override high-confidence
                    
                    # Check for stem potential (multiple adjacent pairs)
                    stem_score = self._compute_stem_score(sequence, i, j)
                    
                    if stem_score >= 2:
                        distances[i, j] = self.CANONICAL_DISTANCES[(nt_i, nt_j)]
                        distances[j, i] = distances[i, j]
                        confidences[i, j] = min(0.85, 0.5 + 0.1 * stem_score)
                        confidences[j, i] = confidences[i, j]
                        
                        # Add stacking constraints for stem residues
                        self._add_stacking_constraints(
                            sequence, i, j, stem_score, distances, confidences
                        )
    
    def _compute_stem_score(
        self, 
        sequence: str, 
        i: int, 
        j: int,
        max_extension: int = 5,
    ) -> int:
        """Compute stem potential score for a base pair."""
        L = len(sequence)
        stem_score = 0
        
        # Check pairs extending in both directions
        for k in range(max_extension):
            if i - k < 0 or j + k >= L:
                break
            
            ni = sequence[i - k].upper()
            nj = sequence[j + k].upper()
            
            if (ni, nj) in self.CANONICAL_DISTANCES:
                stem_score += 1
            else:
                # Allow one mismatch in stem
                if k > 0 and stem_score >= 2:
                    break
        
        return stem_score
    
    def _add_stacking_constraints(
        self,
        sequence: str,
        i: int,
        j: int,
        stem_length: int,
        distances: np.ndarray,
        confidences: np.ndarray,
    ):
        """Add base stacking distance constraints within stems."""
        L = len(sequence)
        
        for k in range(1, min(stem_length, 4)):
            if i - k >= 0 and j + k < L:
                # Stacking between adjacent base pairs
                # i and i-1 should be stacked (~3.4Å apart in helix axis)
                if confidences[i, i - k] < 0.5:
                    distances[i, i - k] = self.STACK_DISTANCE * k
                    distances[i - k, i] = distances[i, i - k]
                    confidences[i, i - k] = 0.5
                    confidences[i - k, i] = 0.5
    
    def _add_loop_constraints(
        self,
        secondary_structure: str,
        distances: np.ndarray,
        confidences: np.ndarray,
    ):
        """Add distance constraints for loop regions."""
        L = len(secondary_structure)
        
        # Find loop regions (consecutive dots)
        in_loop = False
        loop_start = 0
        
        for i in range(L):
            if secondary_structure[i] == '.':
                if not in_loop:
                    in_loop = True
                    loop_start = i
            else:
                if in_loop:
                    # Process loop from loop_start to i-1
                    loop_end = i - 1
                    loop_len = loop_end - loop_start + 1
                    
                    if loop_len >= 3:
                        # Add loop residue distance constraints
                        for li in range(loop_start, loop_end + 1):
                            for lj in range(li + 2, loop_end + 1):
                                sep = lj - li
                                # Loop residues are typically 8-15Å apart
                                est_dist = min(15.0, 8.0 + sep * 1.5)
                                
                                if confidences[li, lj] < 0.4:
                                    distances[li, lj] = est_dist
                                    distances[lj, li] = est_dist
                                    confidences[li, lj] = 0.4
                                    confidences[lj, li] = 0.4
                    
                    in_loop = False
    
    def _add_pseudoknot_constraints(
        self,
        sequence: str,
        distances: np.ndarray,
        confidences: np.ndarray,
        covariation_scores: Optional[np.ndarray] = None,
    ):
        """
        Detect and add constraints for potential pseudoknots.
        
        Pseudoknots are base pairs that cross over each other:
        i < i' < j < j' where (i,j) and (i',j') are both pairs.
        """
        L = len(sequence)
        
        if covariation_scores is None:
            return
        
        # Find high-confidence pairs from existing predictions
        pairs = []
        for i in range(L):
            for j in range(i + 4, L):
                if confidences[i, j] > 0.6:
                    pairs.append((i, j, confidences[i, j]))
        
        # Check for crossing pairs (pseudoknots)
        for idx1, (i, j, conf1) in enumerate(pairs):
            for idx2, (ip, jp, conf2) in enumerate(pairs):
                if idx2 <= idx1:
                    continue
                
                # Check if pairs cross: i < i' < j < j'
                if i < ip < j < jp:
                    # This is a pseudoknot!
                    # Add distance constraint between the crossing regions
                    
                    # The crossing regions should be in proximity
                    # Residues ip to j should be relatively close
                    for k in range(ip, j + 1):
                        for l in range(k + 2, j + 1):
                            if confidences[k, l] < 0.5:
                                distances[k, l] = min(distances[k, l], 12.0)
                                distances[l, k] = distances[k, l]
                                confidences[k, l] = 0.5
                                confidences[l, k] = 0.5
    
    def _detect_pseudoknots(
        self,
        sequence: str,
        covariation_scores: np.ndarray,
        threshold_percentile: float = 95,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect potential pseudoknots from covariation.
        
        Args:
            sequence: RNA sequence
            covariation_scores: (L, L) covariation matrix
            threshold_percentile: Percentile for pair detection
            
        Returns:
            List of ((i, j), (i', j')) crossing pair tuples
        """
        L = len(sequence)
        threshold = np.percentile(covariation_scores, threshold_percentile)
        
        # Find all potential pairs
        pairs = []
        for i in range(L):
            for j in range(i + 4, L):
                if covariation_scores[i, j] > threshold:
                    nt_i = sequence[i].upper()
                    nt_j = sequence[j].upper()
                    if (nt_i, nt_j) in self.CANONICAL_DISTANCES:
                        pairs.append((i, j))
        
        # Find crossing pairs
        pseudoknots = []
        for idx1, (i, j) in enumerate(pairs):
            for idx2, (ip, jp) in enumerate(pairs):
                if idx2 <= idx1:
                    continue
                
                # Check crossing condition
                if i < ip < j < jp:
                    pseudoknots.append(((i, j), (ip, jp)))
                elif ip < i < jp < j:
                    pseudoknots.append(((ip, jp), (i, j)))
        
        return pseudoknots



def build_pairwise_features(
    sequence: str,
    embeddings: Optional[np.ndarray] = None,
    covariation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build pairwise feature tensor for distance prediction.
    
    Args:
        sequence: RNA sequence
        embeddings: (L, D) per-residue embeddings
        covariation: (L, L) covariation scores
        
    Returns:
        features: (L, L, num_features) pairwise features
    """
    L = len(sequence)
    features = []
    
    # 1. One-hot sequence outer product (16 dims)
    nt_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    one_hot = np.zeros((L, 4))
    for i, nt in enumerate(sequence.upper()):
        if nt in nt_map:
            one_hot[i, nt_map[nt]] = 1.0
    
    # Outer product for pair features
    pair_onehot = np.einsum('ia,jb->ijab', one_hot, one_hot).reshape(L, L, 16)
    features.append(pair_onehot)
    
    # 2. Positional features (relative position)
    pos = np.arange(L)
    rel_pos = pos[:, None] - pos[None, :]
    rel_pos_features = np.stack([
        np.abs(rel_pos) / L,  # Normalized distance
        np.sign(rel_pos),     # Direction
        (np.abs(rel_pos) < 5).astype(float),  # Local
        (np.abs(rel_pos) < 10).astype(float), # Medium range
    ], axis=-1)
    features.append(rel_pos_features)
    
    # 3. Embedding outer product (if available)
    if embeddings is not None:
        # Use first few embedding dims for efficiency
        emb_dim = min(32, embeddings.shape[1])
        emb = embeddings[:, :emb_dim]
        emb_pair = np.einsum('id,jd->ijd', emb, emb)
        features.append(emb_pair)
    
    # 4. Covariation features
    if covariation is not None:
        cov_features = np.stack([
            covariation,
            covariation > np.percentile(covariation, 90),
            covariation > np.percentile(covariation, 95),
        ], axis=-1)
        features.append(cov_features)
    
    return np.concatenate(features, axis=-1).astype(np.float32)
