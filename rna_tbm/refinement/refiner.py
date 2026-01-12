"""
Structure refinement pipeline.
Combines torsion prediction with coordinate refinement.
"""

import numpy as np
from typing import Optional, Union, List
from pathlib import Path

from .torsion_model import TORCH_AVAILABLE, TORSION_NAMES, NUM_TORSIONS
from .torsion_to_coords import TorsionToCoords


class StructureRefiner:
    """
    High-level interface for structure refinement.
    
    Uses torsion angle prediction to refine template-based coordinates.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_fallback: bool = True,
    ):
        """
        Initialize refiner.
        
        Args:
            model_path: Path to trained torsion model weights
            device: Device for inference
            use_fallback: Use NumPy fallback if PyTorch unavailable
        """
        self.device = device
        self.use_fallback = use_fallback
        self.torsion_converter = TorsionToCoords()
        
        # Initialize model
        self._model = None
        self._fallback_predictor = None
        
        if TORCH_AVAILABLE and model_path and Path(model_path).exists():
            self._load_model(model_path)
        elif use_fallback:
            from .torsion_model import TorsionPredictorNumpy
            self._fallback_predictor = TorsionPredictorNumpy()
    
    def _load_model(self, model_path: str):
        """Load trained torsion model."""
        from .torsion_model import TorRNAModel
        
        self._model = TorRNAModel(device=self.device)
        self._model.load_decoder_weights(model_path)
        self._model.decoder.eval()
    
    def refine(
        self,
        coords: np.ndarray,
        sequence: str,
        embeddings: Optional[np.ndarray] = None,
        num_iterations: int = 1,
    ) -> np.ndarray:
        """
        Refine structure using torsion angle prediction.
        
        Args:
            coords: Initial C1' coordinates (L, 3)
            sequence: RNA sequence
            embeddings: Optional pre-computed embeddings
            num_iterations: Number of refinement iterations
            
        Returns:
            Refined coordinates (L, 3)
        """
        refined = coords.copy()
        
        for _ in range(num_iterations):
            # Predict torsion angles
            torsions = self.predict_torsions(sequence, embeddings)
            
            # Apply torsion refinement
            refined = self.torsion_converter.apply_torsions(
                refined,
                torsions,
                sequence,
            )
        
        return refined
    
    def predict_torsions(
        self,
        sequence: str,
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict torsion angles for sequence.
        
        Args:
            sequence: RNA sequence
            embeddings: Optional pre-computed embeddings
            
        Returns:
            Torsion angles (L, 7)
        """
        if self._model is not None:
            return self._predict_with_model(sequence, embeddings)
        elif self._fallback_predictor is not None:
            return self._fallback_predictor.predict(sequence)
        else:
            # Return ideal A-form torsions
            return self._get_ideal_torsions(len(sequence))
    
    def _predict_with_model(
        self,
        sequence: str,
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using trained model."""
        import torch
        
        if embeddings is not None:
            # Use provided embeddings
            emb_tensor = torch.from_numpy(embeddings).unsqueeze(0).to(self.device)
            with torch.no_grad():
                torsions = self._model.decoder(emb_tensor)
            return torsions[0].cpu().numpy()
        else:
            # Generate embeddings and predict
            with torch.no_grad():
                torsions = self._model([sequence])
            return torsions.cpu().numpy()
    
    def _get_ideal_torsions(self, length: int) -> np.ndarray:
        """Get ideal A-form RNA torsion angles."""
        # A-form RNA torsion angles in radians
        ideal = np.array([
            np.radians(-68),   # alpha
            np.radians(178),   # beta
            np.radians(54),    # gamma
            np.radians(82),    # delta
            np.radians(-153),  # epsilon
            np.radians(-71),   # zeta
            np.radians(-158),  # chi
        ])
        
        return np.tile(ideal, (length, 1))
    
    def refine_batch(
        self,
        coords_list: List[np.ndarray],
        sequences: List[str],
    ) -> List[np.ndarray]:
        """
        Refine multiple structures.
        
        Args:
            coords_list: List of coordinate arrays
            sequences: List of sequences
            
        Returns:
            List of refined coordinate arrays
        """
        return [
            self.refine(coords, seq)
            for coords, seq in zip(coords_list, sequences)
        ]


class RefinementTrainer:
    """
    Trainer for torsion prediction model.
    Uses self-supervised learning on PDB structures.
    """
    
    def __init__(
        self,
        model: 'TorRNAModel',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        """
        Initialize trainer.
        
        Args:
            model: TorRNAModel to train
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for training")
        
        import torch
        from .torsion_model import AngularLoss
        
        self.model = model
        self.loss_fn = AngularLoss()
        
        # Only train decoder, encoder is frozen
        self.optimizer = torch.optim.AdamW(
            model.decoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
    
    def train_step(
        self,
        sequences: List[str],
        target_torsions: 'torch.Tensor',
    ) -> float:
        """
        Single training step.
        
        Args:
            sequences: Batch of sequences
            target_torsions: Ground truth torsion angles
            
        Returns:
            Loss value
        """
        import torch
        
        self.model.decoder.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_torsions = self.model(sequences)
        
        # Compute loss
        loss = self.loss_fn(pred_torsions, target_torsions)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(
        self,
        sequences: List[str],
        target_torsions: 'torch.Tensor',
    ) -> dict:
        """
        Validate model.
        
        Args:
            sequences: Validation sequences
            target_torsions: Ground truth torsion angles
            
        Returns:
            Validation metrics
        """
        import torch
        
        self.model.decoder.eval()
        
        with torch.no_grad():
            pred_torsions = self.model(sequences)
            loss = self.loss_fn(pred_torsions, target_torsions)
            
            # Compute per-angle MAE
            diff = pred_torsions - target_torsions
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            mae_per_angle = torch.abs(diff).mean(dim=(0, 1))
        
        # Update scheduler
        self.scheduler.step(loss)
        
        return {
            'loss': loss.item(),
            'mae_radians': mae_per_angle.cpu().numpy(),
            'mae_degrees': np.degrees(mae_per_angle.cpu().numpy()),
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        """Save training checkpoint."""
        import torch
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
        }, path)
    
    def load_checkpoint(self, path: str) -> dict:
        """Load training checkpoint."""
        import torch
        
        checkpoint = torch.load(path, map_location=self.model.device)
        self.model.decoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint['metrics'],
        }
