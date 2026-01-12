"""
Contact Map Prediction from Covariation and Embeddings.
Neural network for combining MSA features with sequence embeddings.
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


class ContactPredictorNumpy:
    """
    NumPy-based contact predictor combining covariation and embeddings.
    
    Uses a simple multi-layer perceptron approach with NumPy operations
    for environments without PyTorch.
    """
    
    # Contact threshold (Angstroms)
    CONTACT_THRESHOLD = 8.0
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize predictor.
        
        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate (simulated via noise)
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize random weights (for untrained inference)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights randomly."""
        np.random.seed(42)  # Reproducibility
        self.weights = []
        self.biases = []
        
        # Input dimension: covariation (1) + embedding pair (64) + position (4)
        input_dim = 1 + 64 + 4
        
        for i in range(self.num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = self.hidden_dim
            
            # Xavier initialization
            scale = np.sqrt(2.0 / (in_dim + self.hidden_dim))
            w = np.random.randn(in_dim, self.hidden_dim) * scale
            b = np.zeros(self.hidden_dim)
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Output layer
        scale = np.sqrt(2.0 / (self.hidden_dim + 1))
        self.output_weight = np.random.randn(self.hidden_dim, 1) * scale
        self.output_bias = np.zeros(1)
    
    def predict(
        self,
        sequence: str,
        covariation: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        msa_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict contact probability map.
        
        Args:
            sequence: RNA sequence
            covariation: (L, L) covariation/DCA scores
            embeddings: (L, D) per-residue embeddings
            msa_path: Path to MSA file (for computing covariation)
            
        Returns:
            Contact probability map of shape (L, L)
        """
        L = len(sequence)
        
        # Build pairwise features
        features = self._build_pairwise_features(
            sequence, covariation, embeddings
        )
        
        # Forward pass
        contact_probs = self._forward(features)
        
        # Reshape to (L, L)
        contact_probs = contact_probs.reshape(L, L)
        
        # Symmetrize
        contact_probs = (contact_probs + contact_probs.T) / 2
        
        # Zero diagonal
        np.fill_diagonal(contact_probs, 0)
        
        return contact_probs
    
    def _build_pairwise_features(
        self,
        sequence: str,
        covariation: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build pairwise features for contact prediction."""
        L = len(sequence)
        features_list = []
        
        # 1. Covariation features
        if covariation is not None:
            cov_flat = covariation.reshape(L, L, 1)
        else:
            cov_flat = np.zeros((L, L, 1))
        features_list.append(cov_flat)
        
        # 2. Embedding pairwise features
        if embeddings is not None:
            # Use first 32 dims for efficiency
            emb_dim = min(32, embeddings.shape[1])
            emb = embeddings[:, :emb_dim]
            
            # Outer concatenation [ei || ej]
            emb_i = emb[:, np.newaxis, :]  # (L, 1, D)
            emb_j = emb[np.newaxis, :, :]  # (1, L, D)
            emb_pair = np.concatenate([
                np.broadcast_to(emb_i, (L, L, emb_dim)),
                np.broadcast_to(emb_j, (L, L, emb_dim)),
            ], axis=-1)
            features_list.append(emb_pair)
        else:
            features_list.append(np.zeros((L, L, 64)))
        
        # 3. Position features
        pos = np.arange(L)
        rel_pos = pos[:, None] - pos[None, :]
        pos_features = np.stack([
            np.abs(rel_pos) / L,  # Normalized distance
            (np.abs(rel_pos) < 5).astype(float),  # Local
            (np.abs(rel_pos) < 10).astype(float),  # Medium
            (np.abs(rel_pos) >= 10).astype(float),  # Long-range
        ], axis=-1)
        features_list.append(pos_features)
        
        # Concatenate all features
        features = np.concatenate(features_list, axis=-1)
        
        # Flatten to (L*L, num_features)
        features = features.reshape(L * L, -1)
        
        return features.astype(np.float32)
    
    def _forward(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        x = features
        
        # Hidden layers with ReLU
        for w, b in zip(self.weights, self.biases):
            # Adjust input dimension if needed
            if x.shape[1] != w.shape[0]:
                # Pad or truncate
                if x.shape[1] < w.shape[0]:
                    padding = np.zeros((x.shape[0], w.shape[0] - x.shape[1]))
                    x = np.concatenate([x, padding], axis=1)
                else:
                    x = x[:, :w.shape[0]]
            
            x = x @ w + b
            x = np.maximum(x, 0)  # ReLU
        
        # Output layer with sigmoid
        x = x @ self.output_weight + self.output_bias
        x = 1.0 / (1.0 + np.exp(-x))  # Sigmoid
        
        return x.flatten()
    
    def combine_with_dca(
        self,
        contact_probs: np.ndarray,
        dca_scores: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Combine neural network predictions with DCA scores.
        
        Args:
            contact_probs: (L, L) neural network contact predictions
            dca_scores: (L, L) DCA coupling scores
            alpha: Weight for neural network (1-alpha for DCA)
            
        Returns:
            Combined contact predictions
        """
        L = contact_probs.shape[0]
        
        # Normalize DCA scores to [0, 1]
        mask = ~np.eye(L, dtype=bool)
        if np.sum(mask) > 0:
            dca_min = np.min(dca_scores[mask])
            dca_max = np.max(dca_scores[mask])
            if dca_max > dca_min:
                dca_normalized = (dca_scores - dca_min) / (dca_max - dca_min)
            else:
                dca_normalized = np.zeros_like(dca_scores)
        else:
            dca_normalized = np.zeros_like(dca_scores)
        
        # Weighted combination
        combined = alpha * contact_probs + (1 - alpha) * dca_normalized
        
        # Ensure diagonal is zero
        np.fill_diagonal(combined, 0)
        
        return combined
    
    def predict_with_msa(
        self,
        sequence: str,
        msa_path: str,
        embeddings: Optional[np.ndarray] = None,
        combine_alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Predict contacts using MSA covariation.
        
        Args:
            sequence: RNA sequence
            msa_path: Path to MSA file
            embeddings: Optional per-residue embeddings
            combine_alpha: Weight for neural network predictions
            
        Returns:
            Contact probability map
        """
        # Import here to avoid circular dependencies
        from ..msa.parser import MSAParser
        from ..msa.covariation import CovariationAnalyzer
        
        # Parse MSA
        parser = MSAParser()
        msa = parser.parse(msa_path)
        
        # Compute DCA scores
        analyzer = CovariationAnalyzer()
        dca_scores = analyzer.compute_dca_scores(msa)
        
        # Get neural network predictions
        contact_probs = self.predict(
            sequence, 
            covariation=dca_scores, 
            embeddings=embeddings
        )
        
        # Combine with DCA
        combined = self.combine_with_dca(
            contact_probs, dca_scores, alpha=combine_alpha
        )
        
        return combined


if TORCH_AVAILABLE:
    class ContactPredictorNetwork(nn.Module):
        """
        PyTorch neural network for contact prediction.
        
        Combines covariation and embedding features using a ResNet-style
        architecture for improved accuracy.
        """
        
        def __init__(
            self,
            covariation_dim: int = 1,
            embedding_dim: int = 64,
            hidden_dim: int = 64,
            num_blocks: int = 4,
            dropout: float = 0.1,
        ):
            """
            Initialize network.
            
            Args:
                covariation_dim: Covariation feature dimension
                embedding_dim: Per-residue embedding dimension
                hidden_dim: Hidden layer dimension
                num_blocks: Number of residual blocks
                dropout: Dropout rate
            """
            super().__init__()
            
            # Input: covariation (1) + embedding pairs (2*emb_dim) + position (4)
            input_dim = covariation_dim + 2 * embedding_dim + 4
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # Residual blocks
            self.blocks = nn.ModuleList([
                self._make_block(hidden_dim, dropout)
                for _ in range(num_blocks)
            ])
            
            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        
        def _make_block(self, dim: int, dropout: float) -> nn.Module:
            """Create a residual block."""
            return nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
        
        def forward(
            self,
            covariation: torch.Tensor,
            embeddings: torch.Tensor,
        ) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                covariation: (B, L, L, 1) covariation features
                embeddings: (B, L, D) per-residue embeddings
                
            Returns:
                Contact probabilities (B, L, L)
            """
            B, L, _ = embeddings.shape
            
            # Build pairwise embedding features
            emb_i = embeddings.unsqueeze(2).expand(B, L, L, -1)
            emb_j = embeddings.unsqueeze(1).expand(B, L, L, -1)
            emb_pair = torch.cat([emb_i, emb_j], dim=-1)
            
            # Position features
            device = embeddings.device
            pos = torch.arange(L, device=device)
            rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
            pos_features = torch.stack([
                torch.abs(rel_pos).float() / L,
                (torch.abs(rel_pos) < 5).float(),
                (torch.abs(rel_pos) < 10).float(),
                (torch.abs(rel_pos) >= 10).float(),
            ], dim=-1)
            pos_features = pos_features.unsqueeze(0).expand(B, -1, -1, -1)
            
            # Concatenate all features
            x = torch.cat([covariation, emb_pair, pos_features], dim=-1)
            
            # Flatten for linear layers
            x = x.view(B * L * L, -1)
            
            # Forward through network
            x = self.input_proj(x)
            
            for block in self.blocks:
                x = x + block(x)  # Residual connection
            
            x = self.output_proj(x)
            
            # Reshape and symmetrize
            x = x.view(B, L, L)
            x = (x + x.transpose(1, 2)) / 2
            
            # Zero diagonal
            mask = torch.eye(L, device=device).unsqueeze(0)
            x = x * (1 - mask)
            
            return x


def predict_contacts_combined(
    sequence: str,
    covariation: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
    dca_weight: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to predict contacts and distances.
    
    Args:
        sequence: RNA sequence
        covariation: (L, L) covariation scores
        embeddings: (L, D) per-residue embeddings
        dca_weight: Weight for DCA scores in combination
        
    Returns:
        (contact_probs, distance_predictions): Contact probabilities and distances
    """
    L = len(sequence)
    
    # Initialize predictor
    predictor = ContactPredictorNumpy()
    
    # Get predictions
    contact_probs = predictor.predict(
        sequence,
        covariation=covariation,
        embeddings=embeddings,
    )
    
    # Combine with DCA if provided
    if covariation is not None:
        contact_probs = predictor.combine_with_dca(
            contact_probs, covariation, alpha=1.0 - dca_weight
        )
    
    # Convert to distances
    # Contact (~8Å) vs non-contact (~20Å)
    distances = 20.0 - contact_probs * 12.0  # Range: 8-20Å
    
    # Backbone constraints
    for i in range(L - 1):
        distances[i, i + 1] = 5.9
        distances[i + 1, i] = 5.9
    
    np.fill_diagonal(distances, 0)
    
    return contact_probs, distances
