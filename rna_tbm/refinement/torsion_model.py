"""
TorRNA-style torsion angle prediction model.
Uses RNA-FM embeddings as input and predicts 7 backbone torsion angles.
"""

import numpy as np
from typing import List, Optional, Tuple, Union

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class nn:
        class Module:
            pass


# Torsion angle names
TORSION_NAMES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']
NUM_TORSIONS = 7


if TORCH_AVAILABLE:
    class TorsionDecoder(nn.Module):
        """
        Transformer decoder for torsion angle prediction.
        
        Takes RNA-FM embeddings (L, 640) and predicts 7 torsion angles per residue.
        Uses sin/cos parameterization for angular outputs.
        """
        
        def __init__(
            self,
            input_dim: int = 640,
            hidden_dim: int = 512,
            num_layers: int = 3,
            num_heads: int = 8,
            num_torsions: int = 7,
            dropout: float = 0.1,
        ):
            """
            Initialize decoder.
            
            Args:
                input_dim: Input embedding dimension (640 for RNA-FM)
                hidden_dim: Hidden layer dimension
                num_layers: Number of transformer layers
                num_heads: Number of attention heads
                num_torsions: Number of torsion angles to predict
                dropout: Dropout rate
            """
            super().__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_torsions = num_torsions
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # Positional encoding
            self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            # Separate prediction heads for each torsion angle
            # Output sin and cos for each angle
            self.torsion_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 2),  # sin, cos
                )
                for _ in range(num_torsions)
            ])
        
        def forward(
            self,
            embeddings: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Predict torsion angles from embeddings.
            
            Args:
                embeddings: (B, L, 640) RNA-FM embeddings
                mask: Optional (B, L) padding mask
                
            Returns:
                torsions: (B, L, 7) torsion angles in radians [-π, π]
            """
            # Project input
            x = self.input_proj(embeddings)
            
            # Add positional encoding
            x = self.pos_encoding(x)
            
            # Transformer processing
            if mask is not None:
                x = self.transformer(x, src_key_padding_mask=mask)
            else:
                x = self.transformer(x)
            
            # Predict torsion angles
            angles = []
            for head in self.torsion_heads:
                sincos = head(x)  # (B, L, 2)
                # Convert sin/cos to angle
                angle = torch.atan2(sincos[..., 0], sincos[..., 1])
                angles.append(angle)
            
            # Stack: (B, L, 7)
            return torch.stack(angles, dim=-1)
        
        def predict_with_uncertainty(
            self,
            embeddings: torch.Tensor,
            num_samples: int = 10,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Predict with uncertainty estimation using dropout.
            
            Args:
                embeddings: (B, L, 640) embeddings
                num_samples: Number of forward passes
                
            Returns:
                mean_angles: (B, L, 7) mean predictions
                std_angles: (B, L, 7) standard deviations
            """
            self.train()  # Enable dropout
            
            samples = []
            for _ in range(num_samples):
                angles = self.forward(embeddings)
                samples.append(angles)
            
            samples = torch.stack(samples, dim=0)  # (S, B, L, 7)
            
            # Compute circular mean and std
            sin_samples = torch.sin(samples)
            cos_samples = torch.cos(samples)
            
            mean_sin = sin_samples.mean(dim=0)
            mean_cos = cos_samples.mean(dim=0)
            mean_angles = torch.atan2(mean_sin, mean_cos)
            
            # Circular standard deviation
            R = torch.sqrt(mean_sin**2 + mean_cos**2)
            std_angles = torch.sqrt(-2 * torch.log(R + 1e-8))
            
            self.eval()
            return mean_angles, std_angles
    
    
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding."""
        
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding to input."""
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
    
    
    class TorRNAModel(nn.Module):
        """
        Full TorRNA model combining RNA-FM encoder and torsion decoder.
        
        RNA-FM encoder is frozen, only decoder is trained.
        """
        
        def __init__(
            self,
            encoder_path: Optional[str] = None,
            decoder_config: Optional[dict] = None,
            device: str = "cpu",
        ):
            """
            Initialize model.
            
            Args:
                encoder_path: Path to RNA-FM weights
                decoder_config: Configuration for decoder
                device: Device for computation
            """
            super().__init__()
            
            self.device = device
            
            # Initialize encoder (RNA-FM or fallback)
            from ..embeddings.rna_fm import RNAFMEncoder
            self.encoder = RNAFMEncoder(encoder_path, device=device)
            
            # Initialize decoder
            config = decoder_config or {}
            self.decoder = TorsionDecoder(**config)
            self.decoder = self.decoder.to(device)
        
        def forward(
            self,
            sequences: Union[str, List[str]],
        ) -> torch.Tensor:
            """
            Predict torsion angles from sequences.
            
            Args:
                sequences: Single sequence or list of sequences
                
            Returns:
                torsions: (B, L, 7) or (L, 7) torsion angles
            """
            if isinstance(sequences, str):
                sequences = [sequences]
                single_input = True
            else:
                single_input = False
            
            # Get embeddings
            embeddings_list = self.encoder.encode_batch(sequences)
            
            # Pad to same length
            max_len = max(e.shape[0] for e in embeddings_list)
            batch_embeddings = np.zeros((len(sequences), max_len, 640), dtype=np.float32)
            mask = np.ones((len(sequences), max_len), dtype=bool)
            
            for i, emb in enumerate(embeddings_list):
                batch_embeddings[i, :len(emb)] = emb
                mask[i, :len(emb)] = False
            
            # Convert to tensors
            batch_embeddings = torch.from_numpy(batch_embeddings).to(self.device)
            mask = torch.from_numpy(mask).to(self.device)
            
            # Predict torsions
            torsions = self.decoder(batch_embeddings, mask)
            
            if single_input:
                return torsions[0, :len(sequences[0])]
            return torsions
        
        def load_decoder_weights(self, path: str):
            """Load pre-trained decoder weights."""
            self.decoder.load_state_dict(torch.load(path, map_location=self.device))
        
        def save_decoder_weights(self, path: str):
            """Save decoder weights."""
            torch.save(self.decoder.state_dict(), path)
    
    
    class AngularLoss(nn.Module):
        """
        Loss function for angular predictions.
        Handles the circular nature of angles.
        """
        
        def __init__(self, reduction: str = 'mean'):
            super().__init__()
            self.reduction = reduction
        
        def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Compute angular loss.
            
            Args:
                pred: Predicted angles (B, L, 7)
                target: Target angles (B, L, 7)
                mask: Optional mask for valid positions
                
            Returns:
                Loss value
            """
            # Compute angular difference
            diff = pred - target
            
            # Wrap to [-π, π]
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            
            # Squared angular distance
            loss = diff ** 2
            
            if mask is not None:
                loss = loss * mask.unsqueeze(-1)
            
            if self.reduction == 'mean':
                if mask is not None:
                    return loss.sum() / (mask.sum() * pred.size(-1) + 1e-8)
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss

else:
    # Fallback when PyTorch is not available
    class TorsionDecoder:
        """Placeholder when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for TorsionDecoder")
    
    class TorRNAModel:
        """Placeholder when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for TorRNAModel")
    
    class AngularLoss:
        """Placeholder when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for AngularLoss")


# NumPy-based fallback for inference without PyTorch
class TorsionPredictorNumpy:
    """
    Simple torsion predictor using NumPy.
    Uses heuristics based on sequence and secondary structure.
    """
    
    # Ideal torsion angles for A-form RNA (in radians)
    A_FORM_TORSIONS = {
        'alpha': np.radians(-68),
        'beta': np.radians(178),
        'gamma': np.radians(54),
        'delta': np.radians(82),
        'epsilon': np.radians(-153),
        'zeta': np.radians(-71),
        'chi': np.radians(-158),  # anti for purines
    }
    
    def predict(self, sequence: str) -> np.ndarray:
        """
        Predict torsion angles using A-form heuristics.
        
        Args:
            sequence: RNA sequence
            
        Returns:
            Torsion angles (L, 7)
        """
        L = len(sequence)
        torsions = np.zeros((L, NUM_TORSIONS))
        
        for i in range(L):
            for j, name in enumerate(TORSION_NAMES):
                torsions[i, j] = self.A_FORM_TORSIONS[name]
            
            # Adjust chi angle for purines vs pyrimidines
            nt = sequence[i].upper()
            if nt in ['A', 'G']:  # Purines
                torsions[i, 6] = np.radians(-158)  # anti
            else:  # Pyrimidines
                torsions[i, 6] = np.radians(-152)  # anti
        
        # Add small random perturbation for diversity
        torsions += np.random.randn(L, NUM_TORSIONS) * np.radians(5)
        
        return torsions
