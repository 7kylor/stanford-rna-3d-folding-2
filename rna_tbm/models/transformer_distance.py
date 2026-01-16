"""
Transformer-based Distance Predictor for RNA Structure.

Implements an Evoformer-inspired architecture with:
- Axial attention for efficiency
- MSA row/column attention
- Triangle multiplicative updates for pairwise consistency
- Distance bin classification
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

# Distance bins (Angstroms) - same as AlphaFold
DISTANCE_BINS = np.linspace(2.0, 22.0, 64)
NUM_BINS = len(DISTANCE_BINS) - 1
BIN_CENTERS = (DISTANCE_BINS[:-1] + DISTANCE_BINS[1:]) / 2


if TORCH_AVAILABLE:
    
    class AxialAttention(nn.Module):
        """
        Axial attention for efficient pairwise representation processing.
        
        Instead of full L² attention, applies attention along rows and columns
        separately, reducing complexity to O(L).
        """
        
        def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = self.head_dim ** -0.5
        
        def forward(self, x: torch.Tensor, axis: int = 0) -> torch.Tensor:
            """
            Apply attention along specified axis.
            
            Args:
                x: (B, L, L, D) pairwise representation
                axis: 0 for row attention, 1 for column attention
                
            Returns:
                Updated representation (B, L, L, D)
            """
            B, L1, L2, D = x.shape
            
            if axis == 1:
                x = x.transpose(1, 2)  # Swap for column attention
            
            # Reshape for attention: (B*L, L, D)
            x_flat = x.reshape(B * L1, L2, D)
            
            Q = self.q_proj(x_flat).view(B * L1, L2, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(x_flat).view(B * L1, L2, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(x_flat).view(B * L1, L2, self.num_heads, self.head_dim).transpose(1, 2)
            
            attn = (Q @ K.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            out = (attn @ V).transpose(1, 2).reshape(B * L1, L2, D)
            out = self.out_proj(out)
            out = out.view(B, L1, L2, D)
            
            if axis == 1:
                out = out.transpose(1, 2)
            
            return out
    
    
    class TriangleMultiplication(nn.Module):
        """
        Triangle multiplicative update for pairwise representation.
        
        Updates pair (i,j) based on intermediate nodes k.
        Enforces transitivity: if i-k and k-j are close, i-j should be close.
        """
        
        def __init__(self, dim: int, hidden_dim: int = None, mode: str = 'outgoing'):
            """
            Args:
                dim: Input/output dimension
                hidden_dim: Hidden dimension for projections
                mode: 'outgoing' (i->k, j->k) or 'incoming' (k->i, k->j)
            """
            super().__init__()
            self.mode = mode
            hidden_dim = hidden_dim or dim
            
            self.layer_norm = nn.LayerNorm(dim)
            self.left_proj = nn.Linear(dim, hidden_dim)
            self.right_proj = nn.Linear(dim, hidden_dim)
            self.left_gate = nn.Linear(dim, hidden_dim)
            self.right_gate = nn.Linear(dim, hidden_dim)
            self.output_gate = nn.Linear(dim, dim)
            self.output_proj = nn.Linear(hidden_dim, dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Apply triangle multiplication.
            
            Args:
                x: (B, L, L, D) pairwise representation
                
            Returns:
                Updated representation (B, L, L, D)
            """
            x_norm = self.layer_norm(x)
            
            left = self.left_proj(x_norm) * torch.sigmoid(self.left_gate(x_norm))
            right = self.right_proj(x_norm) * torch.sigmoid(self.right_gate(x_norm))
            
            if self.mode == 'outgoing':
                # Sum over k: pair(i,k) * pair(j,k)
                out = torch.einsum('bikd,bjkd->bijd', left, right)
            else:
                # Sum over k: pair(k,i) * pair(k,j)
                out = torch.einsum('bkid,bkjd->bijd', left, right)
            
            out = self.output_proj(out) * torch.sigmoid(self.output_gate(x_norm))
            
            return x + out
    
    
    class TransformerBlock(nn.Module):
        """
        Single transformer block with axial attention and triangle updates.
        """
        
        def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            
            self.row_attn = AxialAttention(dim, num_heads, dropout)
            self.col_attn = AxialAttention(dim, num_heads, dropout)
            self.triangle_out = TriangleMultiplication(dim, mode='outgoing')
            self.triangle_in = TriangleMultiplication(dim, mode='incoming')
            
            self.ffn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout),
            )
            
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Axial attention
            x = x + self.dropout(self.row_attn(self.norm1(x), axis=0))
            x = x + self.dropout(self.col_attn(self.norm2(x), axis=1))
            
            # Triangle updates
            x = self.triangle_out(x)
            x = self.triangle_in(x)
            
            # FFN
            x = x + self.ffn(x)
            
            return x
    
    
    class RNADistanceTransformer(nn.Module):
        """
        Transformer model for RNA inter-residue distance prediction.
        
        Architecture:
        1. Sequence embedding (one-hot + positional)
        2. Pairwise features (outer product + relative position)
        3. Transformer blocks with axial attention
        4. Distance bin classification
        
        Inspired by Evoformer from AlphaFold2 and RhoFold+.
        """
        
        def __init__(
            self,
            embed_dim: int = 128,
            num_blocks: int = 4,
            num_heads: int = 8,
            dropout: float = 0.1,
            num_bins: int = NUM_BINS,
            max_len: int = 1024,
        ):
            """
            Initialize model.
            
            Args:
                embed_dim: Hidden dimension
                num_blocks: Number of transformer blocks
                num_heads: Attention heads
                dropout: Dropout rate
                num_bins: Output distance bins
                max_len: Maximum sequence length
            """
            super().__init__()
            
            self.embed_dim = embed_dim
            self.num_bins = num_bins
            
            # Sequence embedding
            self.nucleotide_embed = nn.Embedding(5, embed_dim // 4)  # A, C, G, U, pad
            self.position_embed = nn.Embedding(max_len, embed_dim // 4)
            
            # Project to full dimension
            self.seq_proj = nn.Linear(embed_dim // 2, embed_dim)
            
            # Pairwise feature projection
            # Outer product of sequence features + relative position encoding
            self.pair_proj = nn.Linear(embed_dim * 2 + 64, embed_dim)
            self.rel_pos_embed = nn.Embedding(2 * max_len + 1, 64)
            
            # Transformer blocks
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, dropout)
                for _ in range(num_blocks)
            ])
            
            # Output head
            self.output_norm = nn.LayerNorm(embed_dim)
            self.output_proj = nn.Linear(embed_dim, num_bins)
        
        def forward(
            self,
            sequence: torch.Tensor,
            msa_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Predict distance distributions.
            
            Args:
                sequence: (B, L) nucleotide indices (A=0, C=1, G=2, U=3)
                msa_features: Optional (B, L, D) MSA-derived features
                
            Returns:
                logits: (B, L, L, num_bins) distance bin logits
            """
            B, L = sequence.shape
            device = sequence.device
            
            # Sequence embedding
            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            seq_embed = torch.cat([
                self.nucleotide_embed(sequence),
                self.position_embed(positions),
            ], dim=-1)
            seq_embed = self.seq_proj(seq_embed)
            
            # Build pairwise representation
            # Outer product: (B, L, L, 2*D)
            pair_features = torch.cat([
                seq_embed.unsqueeze(2).expand(-1, -1, L, -1),
                seq_embed.unsqueeze(1).expand(-1, L, -1, -1),
            ], dim=-1)
            
            # Relative position encoding
            rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)
            rel_pos = rel_pos + L  # Shift to positive
            rel_pos = rel_pos.clamp(0, 2 * L)
            rel_pos_embed = self.rel_pos_embed(rel_pos)
            
            # Combine features
            pair_repr = torch.cat([pair_features, rel_pos_embed], dim=-1)
            pair_repr = self.pair_proj(pair_repr)
            
            # Transformer blocks
            for block in self.blocks:
                pair_repr = block(pair_repr)
            
            # Output
            pair_repr = self.output_norm(pair_repr)
            logits = self.output_proj(pair_repr)
            
            # Symmetrize
            logits = (logits + logits.transpose(1, 2)) / 2
            
            return logits
        
        def predict_distances(
            self,
            sequence: torch.Tensor,
            msa_features: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Predict expected distances and confidence.
            
            Returns:
                distances: (B, L, L) expected distances
                confidence: (B, L, L) prediction confidence
            """
            logits = self.forward(sequence, msa_features)
            probs = F.softmax(logits, dim=-1)
            
            # Expected distance
            bin_centers = torch.tensor(BIN_CENTERS, device=logits.device, dtype=logits.dtype)
            distances = (probs * bin_centers).sum(dim=-1)
            
            # Confidence: entropy-based
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
            max_entropy = np.log(self.num_bins)
            confidence = 1.0 - entropy / max_entropy
            
            return distances, confidence


class TransformerDistanceNumpy:
    """
    NumPy fallback for distance prediction without PyTorch.
    
    Uses simpler neural network approximation with learned features.
    Still effective for generating distance constraints.
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        """
        Initialize with random weights (for fallback use).
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Nucleotide embedding
        self.nuc_embed = np.random.randn(5, embed_dim) * 0.1
        
        # Simple MLP weights
        self.W1 = np.random.randn(embed_dim * 2, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, NUM_BINS) * 0.1
        self.b3 = np.zeros(NUM_BINS)
        
        # RNA-specific distance priors
        self._setup_priors()
    
    def _setup_priors(self):
        """Set up RNA-specific distance priors."""
        # Base pairing distances
        self.basepair_distance = 8.0  # C1'-C1' for Watson-Crick
        self.stack_distance = 5.9  # Adjacent C1'-C1'
        
        # Complementary pairs
        self.wc_pairs = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')}
        self.wobble_pairs = {('G', 'U'), ('U', 'G')}
    
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """One-hot encode sequence."""
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
        indices = [nuc_map.get(n.upper(), 4) for n in sequence]
        return self.nuc_embed[indices]
    
    def _predict_secondary_structure(self, sequence: str) -> str:
        """
        Simple Nussinov algorithm for maximum base pairing.
        Returns a dot-bracket string.
        """
        L = len(sequence)
        if L > 300: # Limit for performance
            return "." * L
            
        # DP table for max pairs
        dp = np.zeros((L, L), dtype=np.int32)
        
        def can_pair(a, b):
            # Watson-Crick and Wobble
            pair = {a, b}
            return (pair == {'A', 'U'} or pair == {'G', 'C'} or pair == {'G', 'U'})

        # Min hair-pin loop size 3
        for k in range(3, L):
            for i in range(L - k):
                j = i + k
                options = [dp[i + 1, j], dp[i, j - 1], dp[i + 1, j - 1] + (1 if can_pair(sequence[i], sequence[j]) else 0)]
                # Bifurcation
                for m in range(i + 3, j - 3):
                    options.append(dp[i, m] + dp[m + 1, j])
                dp[i, j] = max(options)
        
        # Traceback to get dot-bracket
        structure = ["."] * L
        def traceback(i, j):
            if i >= j - 3:
                return
            if dp[i, j] == dp[i + 1, j]:
                traceback(i + 1, j)
            elif dp[i, j] == dp[i, j - 1]:
                traceback(i, j - 1)
            elif dp[i, j] == dp[i + 1, j - 1] + (1 if can_pair(sequence[i], sequence[j]) else 0):
                if can_pair(sequence[i], sequence[j]):
                    structure[i] = "("
                    structure[j] = ")"
                    traceback(i + 1, j - 1)
                else:
                    traceback(i + 1, j - 1)
            else:
                for m in range(i + 3, j - 3):
                    if dp[i, j] == dp[i, m] + dp[m + 1, j]:
                        traceback(i, m)
                        traceback(m + 1, j)
                        break
        
        traceback(0, L - 1)
        return "".join(structure)

    def predict_from_sequence(
        self,
        sequence: str,
        secondary_structure: Optional[str] = None,
        covariation_scores: Optional[np.ndarray] = None,
        msa_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict distance matrix from sequence with structural priors."""
        L = len(sequence)
        sequence = sequence.upper().replace('T', 'U')
        
        # Use simple Nussinov if no secondary structure provided
        if secondary_structure is None:
            secondary_structure = self._predict_secondary_structure(sequence)
            print(f" Predicted SS: {secondary_structure[:50]}...")
        
        # Get sequence embedding
        seq_embed = self._encode_sequence(sequence)
        
        # Build pairwise features
        pair_features = np.concatenate([
            np.repeat(seq_embed[:, np.newaxis, :], L, axis=1),
            np.repeat(seq_embed[np.newaxis, :, :], L, axis=0),
        ], axis=-1)
        
        # MLP forward pass
        h = pair_features.reshape(L * L, -1)
        h = np.maximum(0, h @ self.W1 + self.b1)  # ReLU
        h = np.maximum(0, h @ self.W2 + self.b2)
        logits = h @ self.W3 + self.b3
        logits = logits.reshape(L, L, NUM_BINS)
        
        # Add prior information
        distances, confidence = self._apply_priors(
            logits, sequence, secondary_structure, covariation_scores
        )
        
        return distances, confidence
    
    def _apply_priors(
        self,
        logits: np.ndarray,
        sequence: str,
        secondary_structure: Optional[str],
        covariation_scores: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply RNA structural priors to predictions."""
        L = len(sequence)
        
        # Start with expected distances from logits
        probs = self._softmax(logits, axis=-1)
        distances = np.sum(probs * BIN_CENTERS, axis=-1)
        
        # Confidence from probability concentration
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
        confidence = 1.0 - entropy / np.log(NUM_BINS)
        
        # 1. Apply Polymer Prior: d_ij ~ sqrt(|i-j|) * 5.0
        # This provides a much better baseline than random/constant distances
        indices = np.arange(L)
        separation = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
        polymer_prior = np.sqrt(separation + 1) * 3.5 + 2.0
        
        # Blend ML prediction with polymer prior (ML is weak without training)
        distances = 0.2 * distances + 0.8 * polymer_prior
        
        # 2. Apply consecutive distance constraint (Very High Confidence)
        for i in range(L - 1):
            distances[i, i + 1] = self.stack_distance
            distances[i + 1, i] = self.stack_distance
            confidence[i, i + 1] = 1.0
            confidence[i + 1, i] = 1.0
        
        # 3. Apply secondary structure if available
        if secondary_structure:
            pairs = self._parse_secondary_structure(secondary_structure)
            for i, j in pairs:
                distances[i, j] = self.basepair_distance
                distances[j, i] = self.basepair_distance
                confidence[i, j] = 0.9
                confidence[j, i] = 0.9
        
        # 4. Apply covariation information
        if covariation_scores is not None and L > 4:
            # High covariation suggests contact
            high_cov_mask = covariation_scores > np.percentile(covariation_scores, 95)
            for i in range(L):
                for j in range(i + 4, L):
                    if high_cov_mask[i, j]:
                        # Likely base pair or contact
                        distances[i, j] = min(distances[i, j], self.basepair_distance + 2)
                        distances[j, i] = distances[i, j]
                        confidence[i, j] = max(confidence[i, j], 0.6)
                        confidence[j, i] = confidence[i, j]
        
        # Ensure proper range
        distances = np.clip(distances, 3.0, 100.0)
        np.fill_diagonal(distances, 0)
        
        # Symmetrize
        distances = (distances + distances.T) / 2
        confidence = (confidence + confidence.T) / 2
        
        return distances, confidence
    
    def _parse_secondary_structure(self, ss: str) -> List[Tuple[int, int]]:
        """Parse dot-bracket notation to pairs."""
        pairs = []
        stack = []
        
        for i, char in enumerate(ss):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                j = stack.pop()
                pairs.append((j, i))
        
        return pairs
    
    def _can_basepair(self, nt1: str, nt2: str) -> bool:
        """Check if nucleotides can form base pair."""
        return (nt1, nt2) in self.wc_pairs or (nt1, nt2) in self.wobble_pairs
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def predict_distances(
    sequence: str,
    secondary_structure: Optional[str] = None,
    covariation_scores: Optional[np.ndarray] = None,
    use_torch: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict inter-residue distances.
    
    Uses PyTorch model if available, otherwise NumPy fallback.
    
    Args:
        sequence: RNA sequence
        secondary_structure: Optional dot-bracket notation
        covariation_scores: Optional covariation matrix
        use_torch: Try to use PyTorch model
        
    Returns:
        distances: (L, L) predicted distances
        confidence: (L, L) prediction confidence
    """
    if use_torch and TORCH_AVAILABLE:
        try:
            model = RNADistanceTransformer()
            model.eval()
            
            nuc_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
            seq_tensor = torch.tensor([[nuc_map.get(n.upper(), 4) for n in sequence]])
            
            with torch.no_grad():
                distances, confidence = model.predict_distances(seq_tensor)
            
            return distances[0].numpy(), confidence[0].numpy()
        except Exception:
            pass  # Fall back to NumPy
    
    # NumPy fallback
    predictor = TransformerDistanceNumpy()
    return predictor.predict_from_sequence(sequence, secondary_structure, covariation_scores)
