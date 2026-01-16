"""
TriFold-Lite: Geometric Modules for RNA Structure Prediction

This module implements the three breakthrough components:
1. TriangleUpdate - Lightweight geometric consistency enforcement
2. DifferentiableDistanceGeometry (DDG) - Iterative coordinate refinement
3. ConfidenceHead - Prediction uncertainty estimation

These modules transform a standard ResNet2D into a geometry-aware predictor.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# Component 1: Triangle Update Module
# =============================================================================

class TriangleUpdate(nn.Module):
    """
    Lightweight triangle consistency update.
    
    For each pair (i,j), gather information from intermediate nodes k
    to ensure geometric consistency: d(i,j) ≈ path through k.
    
    This is a simplified version of Triangular Attention that:
    - Uses sampling instead of full attention (O(L²K) vs O(L³))
    - Maintains geometric intuition without expensive softmax
    - Can be dropped into existing architectures easily
    
    Args:
        dim: Feature dimension
        num_samples: Number of intermediate nodes to sample (K)
    """
    
    def __init__(self, dim: int, num_samples: int = 32):
        super().__init__()
        self.dim = dim
        self.num_samples = num_samples
        
        # Projections for outgoing and incoming edges
        self.proj_out = nn.Linear(dim, dim)  # i -> k edges
        self.proj_in = nn.Linear(dim, dim)   # k -> j edges
        
        # Gating mechanism
        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, L, L, dim) pairwise representations
            
        Returns:
            Updated pair representations with triangle consistency
        """
        B, L, _, dim = pair_repr.shape
        
        # Sample K intermediate nodes (random subset for efficiency)
        K = min(self.num_samples, L)
        if self.training:
            indices = torch.randperm(L, device=pair_repr.device)[:K]
        else:
            # Deterministic sampling during eval (evenly spaced)
            indices = torch.linspace(0, L-1, K, device=pair_repr.device).long()
        
        # Outgoing edges: pair[i, k] for sampled k
        # Shape: (B, L, K, dim)
        out_edges = pair_repr[:, :, indices, :]
        out_aggregated = self.proj_out(out_edges).mean(dim=2)  # (B, L, dim)
        
        # Incoming edges: pair[k, j] for sampled k
        # Shape: (B, K, L, dim)
        in_edges = pair_repr[:, indices, :, :]
        in_aggregated = self.proj_in(in_edges).mean(dim=1)  # (B, L, dim)
        
        # Broadcast to pairwise
        out_broadcast = out_aggregated.unsqueeze(2).expand(-1, -1, L, -1)  # (B, L, L, dim)
        in_broadcast = in_aggregated.unsqueeze(1).expand(-1, L, -1, -1)   # (B, L, L, dim)
        
        # Compute update with gating
        combined = torch.cat([out_broadcast, in_broadcast], dim=-1)  # (B, L, L, 2*dim)
        gate = self.gate_proj(combined)  # (B, L, L, dim)
        
        # Triangle-consistent update
        update = out_broadcast + in_broadcast  # Approximate path through k
        
        # Gated residual connection
        output = pair_repr + gate * (update - pair_repr)
        
        return self.layer_norm(output)


class TriangleMultiplication(nn.Module):
    """
    Triangle multiplication update (from AlphaFold2/3).
    
    More parameter-efficient than TriangleUpdate but similar effect.
    Computes: output[i,j] = sum_k (proj_left(pair[i,k]) * proj_right(pair[k,j]))
    """
    
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        
        self.proj_left = nn.Linear(dim, hidden_dim)
        self.proj_right = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)
        
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, L, L, dim)
        Returns:
            Updated (B, L, L, dim)
        """
        # Project
        left = self.proj_left(pair_repr)   # (B, L, L, hidden)
        right = self.proj_right(pair_repr) # (B, L, L, hidden)
        
        # Triangle multiplication: sum over k
        # left[i,k] * right[k,j] -> output[i,j]
        # This is a batched matrix multiplication over the hidden dimension
        # left: (B, L, L, H) -> (B, L, H, L) for matmul
        left_t = left.permute(0, 1, 3, 2)  # (B, L, H, L)
        # right: (B, L, L, H) -> (B, L, L, H)
        
        # Efficient: use einsum
        # output[b,i,j,h] = sum_k left[b,i,k,h] * right[b,k,j,h]
        output = torch.einsum('bikh,bkjh->bijh', left, right)  # (B, L, L, hidden)
        
        output = self.proj_out(output)  # (B, L, L, dim)
        
        # Gated residual
        gate = self.gate(pair_repr)
        output = pair_repr + gate * output
        
        return self.layer_norm(output)


# =============================================================================
# Component 2: Differentiable Distance Geometry
# =============================================================================

class DifferentiableDistanceGeometry(nn.Module):
    """
    Iterative coordinate refinement from predicted distances.
    
    Instead of one-shot MDS, this module:
    1. Initializes coordinates via fast MDS
    2. Iteratively refines to minimize distance mismatches
    3. Uses confidence weights to prioritize reliable predictions
    
    Key insight: MDS treats all distances equally, but our predictions
    have varying confidence. DDG weights by confidence.
    """
    
    def __init__(
        self, 
        num_iterations: int = 50,
        lr: float = 0.1,
        use_confidence: bool = True
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.lr = lr
        self.use_confidence = use_confidence
        
    def forward(
        self, 
        pred_distances: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred_distances: (L, L) predicted distance matrix
            confidence: (L, L) confidence weights (higher = more reliable)
            mask: (L, L) valid distance mask
            
        Returns:
            coords: (L, 3) refined 3D coordinates
        """
        L = pred_distances.shape[0]
        device = pred_distances.device
        
        # Default confidence and mask
        if confidence is None:
            confidence = torch.ones(L, L, device=device)
        if mask is None:
            mask = torch.ones(L, L, device=device)
            
        # Symmetrize
        pred_distances = (pred_distances + pred_distances.T) / 2
        confidence = (confidence + confidence.T) / 2
        
        # Initialize via MDS
        coords = self._mds_initialize(pred_distances, mask)
        
        # Make coordinates learnable for refinement
        coords = coords.clone().detach().requires_grad_(True)
        
        # Optimizer for refinement
        optimizer = torch.optim.Adam([coords], lr=self.lr)
        
        # Refinement loop
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Compute current pairwise distances
            diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (L, L, 3)
            current_dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # (L, L)
            
            # Weighted distance loss
            weight = confidence * mask
            loss = (weight * (current_dist - pred_distances) ** 2).sum()
            loss = loss / (weight.sum() + 1e-8)
            
            # Add regularization for smoothness
            if L > 2:
                # Encourage sequential neighbors to be close
                neighbor_dist = torch.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(-1) + 1e-8)
                reg_loss = ((neighbor_dist - 5.9) ** 2).mean()  # Target ~5.9Å for C1'-C1'
                loss = loss + 0.1 * reg_loss
            
            loss.backward()
            optimizer.step()
            
        return coords.detach()
    
    def _mds_initialize(self, distances: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Classical MDS initialization."""
        L = distances.shape[0]
        device = distances.device
        
        # Handle missing values
        D = distances.clone()
        D[mask == 0] = 0
        
        # Center the distance matrix
        D_sq = D ** 2
        H = torch.eye(L, device=device) - torch.ones(L, L, device=device) / L
        B = -0.5 * H @ D_sq @ H
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(B)
            
            # Take top 3 positive eigenvalues
            idx = torch.argsort(eigenvalues, descending=True)[:3]
            top_evals = eigenvalues[idx]
            top_evecs = eigenvectors[:, idx]
            
            # Construct coordinates
            coords = top_evecs * torch.sqrt(torch.clamp(top_evals, min=1e-10))
            
        except:
            # Fallback to random initialization
            coords = torch.randn(L, 3, device=device) * 5.0
            
        return coords


def ddg_refine_coordinates(
    pred_distances: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    num_iterations: int = 100,
    lr: float = 0.1
) -> np.ndarray:
    """
    Convenience function for coordinate refinement.
    
    Args:
        pred_distances: (L, L) numpy distance matrix
        confidence: (L, L) optional confidence weights
        num_iterations: Refinement iterations
        lr: Learning rate
        
    Returns:
        coords: (L, 3) numpy coordinate array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dist_tensor = torch.from_numpy(pred_distances).float().to(device)
    conf_tensor = None
    if confidence is not None:
        conf_tensor = torch.from_numpy(confidence).float().to(device)
        
    ddg = DifferentiableDistanceGeometry(num_iterations=num_iterations, lr=lr)
    coords = ddg(dist_tensor, conf_tensor)
    
    return coords.cpu().numpy()


# =============================================================================
# Component 3: Confidence Estimation
# =============================================================================

class ConfidenceHead(nn.Module):
    """
    Predict confidence/uncertainty for distance predictions.
    
    High confidence regions should be weighted more in DDG.
    Low confidence regions may have larger errors.
    
    Architecture: Small CNN that takes distance logits and produces
    per-pair confidence scores.
    """
    
    def __init__(self, num_bins: int = 63, hidden_dim: int = 32):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(num_bins, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, distance_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distance_logits: (B, L, L, num_bins) from distance predictor
            
        Returns:
            confidence: (B, L, L) confidence scores in [0, 1]
        """
        # Permute for conv2d: (B, num_bins, L, L)
        x = distance_logits.permute(0, 3, 1, 2)
        
        # Predict confidence
        conf = self.net(x)  # (B, 1, L, L)
        
        return conf.squeeze(1)  # (B, L, L)


# =============================================================================
# Enhanced Distance Predictor with Triangle Updates
# =============================================================================

class ResBlock2D(nn.Module):
    """Standard 2D ResBlock with dilated convolutions."""
    
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.act = nn.ELU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class TriFoldLite(nn.Module):
    """
    TriFold-Lite: Enhanced distance predictor with geometric consistency.
    
    Architecture:
    1. Input projection
    2. ResNet2D blocks (standard convolutions)
    3. Triangle Update layers (geometric consistency)
    4. Distance prediction head
    5. Confidence estimation head
    
    Key innovation: Interleave conv blocks with triangle updates
    to maintain geometric consistency throughout the network.
    """
    
    def __init__(
        self,
        input_dim: int = 41,
        hidden_dim: int = 64,
        num_conv_blocks: int = 12,
        num_triangle_layers: int = 4,
        num_bins: int = 63,
        triangle_samples: int = 32,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        
        # Input projection
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, 1)
        
        # Interleaved conv and triangle blocks
        self.blocks = nn.ModuleList()
        self.block_types = []  # Track block types separately
        
        dilations = [1, 2, 4, 8, 16]
        blocks_per_triangle = num_conv_blocks // (num_triangle_layers + 1)
        
        block_idx = 0
        for i in range(num_triangle_layers + 1):
            # Add conv blocks
            for j in range(blocks_per_triangle):
                dilation = dilations[block_idx % len(dilations)]
                self.blocks.append(ResBlock2D(hidden_dim, dilation))
                self.block_types.append('conv')
                block_idx += 1
                
            # Add triangle update (except after last group)
            if i < num_triangle_layers:
                self.blocks.append(TriangleUpdate(hidden_dim, triangle_samples))
                self.block_types.append('triangle')
        
        # Distance prediction head
        self.distance_head = nn.Conv2d(hidden_dim, num_bins, 3, padding=1)
        
        # Confidence head
        self.confidence_head = ConfidenceHead(num_bins, hidden_dim=32)
        
        # DDG module (not learnable, just for inference)
        self.ddg = DifferentiableDistanceGeometry(num_iterations=50, lr=0.1)
        
    def forward(
        self, 
        features: torch.Tensor,
        return_confidence: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: (B, L, L, input_dim) pairwise features
            return_confidence: Whether to compute confidence
            
        Returns:
            distance_logits: (B, L, L, num_bins)
            confidence: (B, L, L) if return_confidence else None
        """
        B, L, _, _ = features.shape
        
        # Permute for conv: (B, input_dim, L, L)
        x = features.permute(0, 3, 1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Process through blocks
        for block_type, block in zip(self.block_types, self.blocks):
            if block_type == 'conv':
                x = block(x)
            else:  # triangle
                # Permute for triangle update: (B, L, L, dim)
                x = x.permute(0, 2, 3, 1)
                x = block(x)
                # Permute back: (B, dim, L, L)
                x = x.permute(0, 3, 1, 2)
        
        # Distance prediction
        distance_logits = self.distance_head(x)  # (B, num_bins, L, L)
        distance_logits = distance_logits.permute(0, 2, 3, 1)  # (B, L, L, num_bins)
        
        # Confidence estimation
        confidence = None
        if return_confidence:
            confidence = self.confidence_head(distance_logits)
            
        return distance_logits, confidence
    
    def predict_distances(
        self, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict expected distances and confidence.
        
        Returns:
            distances: (B, L, L) expected distances in Angstroms
            confidence: (B, L, L) confidence scores
        """
        logits, confidence = self.forward(features, return_confidence=True)
        
        # Convert logits to expected distance
        probs = F.softmax(logits, dim=-1)  # (B, L, L, num_bins)
        
        # Bin centers
        bin_edges = torch.linspace(2.0, 22.0, self.num_bins + 1, device=logits.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # (num_bins,)
        
        # Expected distance
        distances = (probs * bin_centers).sum(dim=-1)  # (B, L, L)
        
        return distances, confidence
    
    def predict_structure(
        self, 
        features: torch.Tensor,
        refine_iterations: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full pipeline: features -> distances -> 3D coordinates.
        
        Returns:
            coords: (B, L, 3) 3D coordinates
            distances: (B, L, L) predicted distances
            confidence: (B, L, L) confidence scores
        """
        with torch.no_grad():
            distances, confidence = self.predict_distances(features)
        
        B, L, _ = distances.shape
        all_coords = []
        
        for b in range(B):
            # Refine coordinates for this sample (detached tensors)
            self.ddg.num_iterations = refine_iterations
            coords = self.ddg(distances[b].detach(), confidence[b].detach())
            all_coords.append(coords)
            
        coords = torch.stack(all_coords, dim=0)  # (B, L, 3)
        
        return coords, distances, confidence


# =============================================================================
# Utility: Convert existing checkpoint to TriFold-Lite
# =============================================================================

def upgrade_checkpoint_to_trifold(
    old_checkpoint_path: str,
    new_checkpoint_path: str,
    num_triangle_layers: int = 4
):
    """
    Load a trained DistancePredictor checkpoint and convert to TriFold-Lite.
    
    The conv blocks are preserved, triangle layers are initialized fresh.
    """
    import pickle
    
    old_ckpt = torch.load(old_checkpoint_path, map_location='cpu')
    old_state = old_ckpt['model_state_dict']
    old_config = old_ckpt.get('config', {})
    
    # Create new model
    new_model = TriFoldLite(
        input_dim=old_config.get('input_dim', 41),
        hidden_dim=old_config.get('hidden_dim', 64),
        num_conv_blocks=old_config.get('num_blocks', 16),
        num_triangle_layers=num_triangle_layers,
        num_bins=old_config.get('num_bins', 63),
    )
    
    # Copy matching weights
    new_state = new_model.state_dict()
    
    # Map old block names to new ones
    # This is a simplified version - may need adjustment based on exact naming
    copied = 0
    for key in new_state.keys():
        if key in old_state and new_state[key].shape == old_state[key].shape:
            new_state[key] = old_state[key]
            copied += 1
            
    print(f"Copied {copied}/{len(new_state)} parameters from old checkpoint")
    
    # Save new checkpoint
    new_ckpt = {
        'model_state_dict': new_state,
        'config': {
            'input_dim': old_config.get('input_dim', 41),
            'hidden_dim': old_config.get('hidden_dim', 64),
            'num_conv_blocks': old_config.get('num_blocks', 16),
            'num_triangle_layers': num_triangle_layers,
            'num_bins': old_config.get('num_bins', 63),
        },
        'upgraded_from': old_checkpoint_path,
    }
    torch.save(new_ckpt, new_checkpoint_path)
    print(f"Saved upgraded checkpoint to {new_checkpoint_path}")
    
    return new_model


if __name__ == "__main__":
    # Test the modules
    print("Testing TriFold-Lite components...")
    
    # Test Triangle Update
    print("\n1. Testing TriangleUpdate...")
    tri = TriangleUpdate(dim=64, num_samples=16)
    x = torch.randn(2, 50, 50, 64)
    y = tri(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    
    # Test DDG
    print("\n2. Testing DifferentiableDistanceGeometry...")
    ddg = DifferentiableDistanceGeometry(num_iterations=20)
    dist = torch.rand(30, 30) * 20
    dist = (dist + dist.T) / 2
    coords = ddg(dist)
    print(f"   Distance matrix: {dist.shape} -> Coordinates: {coords.shape}")
    
    # Test full model
    print("\n3. Testing TriFoldLite...")
    model = TriFoldLite(
        input_dim=41,
        hidden_dim=64,
        num_conv_blocks=8,
        num_triangle_layers=2,
    )
    features = torch.randn(2, 40, 40, 41)
    logits, conf = model(features)
    print(f"   Features: {features.shape}")
    print(f"   Distance logits: {logits.shape}")
    print(f"   Confidence: {conf.shape}")
    
    # Test structure prediction
    print("\n4. Testing structure prediction...")
    coords, distances, confidence = model.predict_structure(features, refine_iterations=20)
    print(f"   Coordinates: {coords.shape}")
    print(f"   Distances: {distances.shape}")
    print(f"   Confidence: {confidence.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n   Total parameters: {n_params:,}")
    
    print("\n✓ All tests passed!")
