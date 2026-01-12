"""
MgNet-style Graph Convolutional Network for metal ion binding prediction.
Predicts Mg²⁺ binding probability at each RNA residue position.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Feature dimensions
NUM_NUCLEOTIDE_TYPES = 4  # A, C, G, U
NUM_GEOMETRIC_FEATURES = 12  # Distance, angles, etc.
TOTAL_NODE_FEATURES = NUM_NUCLEOTIDE_TYPES + NUM_GEOMETRIC_FEATURES


if TORCH_AVAILABLE:
    try:
        from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
        from torch_geometric.data import Data, Batch
        TORCH_GEOMETRIC_AVAILABLE = True
    except ImportError:
        TORCH_GEOMETRIC_AVAILABLE = False
    
    
    if TORCH_GEOMETRIC_AVAILABLE:
        class MgNetPredictor(nn.Module):
            """
            Graph Convolutional Network for Mg²⁺ binding site prediction.
            
            Uses RNA structure as a graph where:
            - Nodes: Residues with nucleotide type and geometric features
            - Edges: Spatial proximity (distance < cutoff)
            """
            
            def __init__(
                self,
                node_features: int = TOTAL_NODE_FEATURES,
                hidden_dim: int = 64,
                num_layers: int = 4,
                dropout: float = 0.2,
                edge_cutoff: float = 10.0,
            ):
                """
                Initialize MgNet.
                
                Args:
                    node_features: Number of input features per node
                    hidden_dim: Hidden layer dimension
                    num_layers: Number of GCN layers
                    dropout: Dropout rate
                    edge_cutoff: Distance cutoff for edges (Å)
                """
                super().__init__()
                
                self.edge_cutoff = edge_cutoff
                self.dropout = dropout
                
                # Node encoder
                self.node_encoder = nn.Sequential(
                    nn.Linear(node_features, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                
                # Graph convolution layers
                self.conv_layers = nn.ModuleList()
                self.batch_norms = nn.ModuleList()
                
                for _ in range(num_layers):
                    self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                
                # Binding probability head
                self.binding_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )
            
            def forward(self, data: Data) -> torch.Tensor:
                """
                Predict Mg²⁺ binding probabilities.
                
                Args:
                    data: PyG Data object with x (node features) and edge_index
                    
                Returns:
                    binding_probs: (N,) binding probability per residue
                """
                x = self.node_encoder(data.x)
                
                # Graph convolutions with residual connections
                for conv, bn in zip(self.conv_layers, self.batch_norms):
                    x_new = conv(x, data.edge_index)
                    x_new = bn(x_new)
                    x_new = F.relu(x_new)
                    x_new = F.dropout(x_new, p=self.dropout, training=self.training)
                    x = x + x_new  # Residual connection
                
                # Predict binding probability
                binding_probs = self.binding_head(x).squeeze(-1)
                
                return binding_probs
            
            def predict_batch(self, batch: Batch) -> torch.Tensor:
                """Predict on batch of graphs."""
                return self.forward(batch)
        
        
        class MgNetWithAttention(nn.Module):
            """
            MgNet variant using Graph Attention Networks.
            May capture binding patterns more accurately.
            """
            
            def __init__(
                self,
                node_features: int = TOTAL_NODE_FEATURES,
                hidden_dim: int = 64,
                num_layers: int = 4,
                num_heads: int = 4,
                dropout: float = 0.2,
            ):
                super().__init__()
                
                self.node_encoder = nn.Linear(node_features, hidden_dim)
                
                # GAT layers
                self.gat_layers = nn.ModuleList()
                for i in range(num_layers):
                    # Last layer: single head, no concat
                    if i == num_layers - 1:
                        self.gat_layers.append(
                            GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
                        )
                    else:
                        self.gat_layers.append(
                            GATConv(hidden_dim, hidden_dim // num_heads, 
                                   heads=num_heads, dropout=dropout)
                        )
                
                self.binding_head = nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
            
            def forward(self, data: Data) -> torch.Tensor:
                x = F.relu(self.node_encoder(data.x))
                
                for gat in self.gat_layers:
                    x = F.relu(gat(x, data.edge_index))
                
                return self.binding_head(x).squeeze(-1)
    
    else:
        # Fallback without torch_geometric
        class MgNetPredictor(nn.Module):
            """Simple MLP fallback when torch_geometric not available."""
            
            def __init__(
                self,
                node_features: int = TOTAL_NODE_FEATURES,
                hidden_dim: int = 64,
                num_layers: int = 4,
                dropout: float = 0.2,
                **kwargs,
            ):
                super().__init__()
                
                layers = [nn.Linear(node_features, hidden_dim), nn.ReLU()]
                for _ in range(num_layers - 1):
                    layers.extend([
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ])
                layers.append(nn.Linear(hidden_dim, 1))
                layers.append(nn.Sigmoid())
                
                self.mlp = nn.Sequential(*layers)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Predict binding probabilities from node features."""
                return self.mlp(x).squeeze(-1)
        
        class MgNetWithAttention(MgNetPredictor):
            """Alias for fallback."""
            pass

else:
    # Placeholders when PyTorch not available
    TORCH_GEOMETRIC_AVAILABLE = False
    
    class MgNetPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for MgNetPredictor")
    
    class MgNetWithAttention:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for MgNetWithAttention")


def build_graph_from_coords(
    coords: np.ndarray,
    sequence: str,
    edge_cutoff: float = 10.0,
) -> Dict:
    """
    Build graph representation from RNA coordinates.
    
    Args:
        coords: (L, 3) C1' coordinates
        sequence: RNA sequence
        edge_cutoff: Distance cutoff for edges (Å)
        
    Returns:
        Dictionary with node_features and edge_index
    """
    L = len(sequence)
    
    # Build node features
    node_features = np.zeros((L, TOTAL_NODE_FEATURES))
    
    # One-hot nucleotide type (first 4 features)
    nt_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    for i, nt in enumerate(sequence.upper()):
        if nt in nt_map:
            node_features[i, nt_map[nt]] = 1.0
    
    # Geometric features (features 4-15)
    for i in range(L):
        # Distance to neighbors
        if i > 0:
            node_features[i, 4] = np.linalg.norm(coords[i] - coords[i-1])
        if i < L - 1:
            node_features[i, 5] = np.linalg.norm(coords[i] - coords[i+1])
        
        # Local density (number of atoms within 8Å)
        distances = np.linalg.norm(coords - coords[i], axis=1)
        node_features[i, 6] = np.sum(distances < 8.0)
        
        # Local curvature (if enough neighbors)
        if 1 < i < L - 2:
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            node_features[i, 7] = cos_angle
        
        # Position encoding
        node_features[i, 8] = i / L  # Relative position
        node_features[i, 9] = 1.0 if i < 5 else 0.0  # 5' end
        node_features[i, 10] = 1.0 if i >= L - 5 else 0.0  # 3' end
        
        # Coordinates (normalized)
        node_features[i, 11:14] = coords[i] / 100.0  # Scale down
    
    # Build edges based on distance
    edge_list = []
    for i in range(L):
        for j in range(i + 1, L):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < edge_cutoff:
                edge_list.append([i, j])
                edge_list.append([j, i])  # Undirected
    
    # Add sequential edges
    for i in range(L - 1):
        if [i, i+1] not in edge_list:
            edge_list.append([i, i+1])
            edge_list.append([i+1, i])
    
    edge_index = np.array(edge_list).T if edge_list else np.zeros((2, 0), dtype=np.int64)
    
    return {
        'node_features': node_features.astype(np.float32),
        'edge_index': edge_index.astype(np.int64),
    }


class MgBindingMotifs:
    """
    Known Mg²⁺ binding motifs in RNA.
    Used for heuristic-based prediction when no model available.
    """
    
    # Common Mg binding sequence contexts (approximate)
    # These are simplified patterns, real binding is structure-dependent
    BINDING_PATTERNS = [
        'GG',    # G-rich regions often coordinate Mg
        'GGG',
        'GA',
        'AG',
        'GNRA',  # GNRA tetraloop (G-N-R-A)
        'UNCG',  # UNCG tetraloop
    ]
    
    # Structural features associated with binding
    # Based on analysis of experimental structures
    @staticmethod
    def score_binding_potential(
        sequence: str,
        coords: np.ndarray,
        position: int,
    ) -> float:
        """
        Score binding potential at a position using heuristics.
        
        Args:
            sequence: RNA sequence
            coords: C1' coordinates
            position: Position to score
            
        Returns:
            Binding potential score (0-1)
        """
        score = 0.0
        L = len(sequence)
        
        # Nucleotide preference (G > A > U > C for Mg binding)
        nt = sequence[position].upper()
        nt_scores = {'G': 0.4, 'A': 0.25, 'U': 0.15, 'C': 0.1}
        score += nt_scores.get(nt, 0.0)
        
        # Context: G-rich regions
        start = max(0, position - 2)
        end = min(L, position + 3)
        context = sequence[start:end].upper()
        g_count = context.count('G')
        score += 0.1 * g_count
        
        # Structural: regions with unusual geometry (loops, turns)
        if 1 < position < L - 2:
            v1 = coords[position] - coords[position-1]
            v2 = coords[position+1] - coords[position]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            # Sharp turns have lower cos_angle
            if cos_angle < 0.5:
                score += 0.2
        
        return min(score, 1.0)
