#!/usr/bin/env python3
"""
Phase 2: Train ML Distance Predictor

=============================================================================
WHY THIS SCRIPT EXISTS
=============================================================================
Phase 1 template-based modeling only achieved TM=0.036 because 27/28 test
sequences have no close templates. This ML training:
1. Learns general RNA folding principles from 4,572 training structures
2. Can predict distances for novel sequences without templates
3. Uses distogram (distance distribution) for better calibration

=============================================================================
WHAT THIS SCRIPT DOES
=============================================================================
1. Loads prepared training data (features, distance matrices)
2. Trains DistancePredictor model using distogram loss
3. Validates on held-out structures every epoch
4. Saves best checkpoint based on validation loss
5. Logs all progress with detailed WHAT/WHY/HOW annotations

=============================================================================
HOW IT WORKS
=============================================================================
Training Loop:
1. Sample batch of (features, distances) pairs
2. Forward pass: model(features) → predicted distance distribution
3. Backward pass: gradient of cross-entropy loss
4. Optimizer step: update model weights

Key Hyperparameters:
- Batch size: 4 (memory-limited due to L×L distance matrices)
- Learning rate: 1e-4 with cosine annealing
- Epochs: 50 (with early stopping patience=10)
- Max sequence length: 500 (longer sequences cropped randomly)
"""

import sys
import time
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not available. Cannot train ML model.")
    print("Install with: pip install torch")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def log_change(what: str, why: str, how: str) -> None:
    """Log a change in detailed format."""
    logger.info("=" * 60)
    logger.info(f"WHAT: {what}")
    logger.info(f"WHY:  {why}")
    logger.info(f"HOW:  {how}")
    logger.info("=" * 60)


class RNADistanceDataset(Dataset):
    """
    PyTorch Dataset for RNA distance prediction.
    
    WHAT: Wraps prepared data for DataLoader
    WHY:  Enables batching, shuffling, and efficient data loading
    HOW:  __getitem__ returns (features, distance_bins, mask) tuple
    """
    
    def __init__(
        self,
        features: List[np.ndarray],
        distances: List[np.ndarray],
        max_length: int = 500,
        num_bins: int = 63,
    ):
        """
        Args:
            features: List of (L, 20) feature arrays
            distances: List of (L, L) distance matrices
            max_length: Maximum sequence length (longer sequences cropped)
            num_bins: Number of distance bins (default 63)
        """
        self.features = features
        self.distances = distances
        self.max_length = max_length
        self.num_bins = num_bins
        
        # Distance bins (2-22 Angstroms)
        self.bin_edges = np.linspace(2.0, 22.0, num_bins + 1)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            features: (L, L, d_pair) pairwise features
            distance_bins: (L, L) discretized distance labels
            mask: (L, L) valid position mask
        """
        feat = self.features[idx]  # (L, 20)
        dist = self.distances[idx]  # (L, L)
        
        L = feat.shape[0]
        
        # Crop if too long
        if L > self.max_length:
            start = np.random.randint(0, L - self.max_length)
            feat = feat[start:start + self.max_length]
            dist = dist[start:start + self.max_length, start:start + self.max_length]
            L = self.max_length
        
        # Build pairwise features
        pair_feat = self._build_pair_features(feat)  # (L, L, d_pair)
        
        # Discretize distances into bins
        dist_bins = np.digitize(dist, self.bin_edges) - 1
        dist_bins = np.clip(dist_bins, 0, self.num_bins - 1)
        
        # Create mask (valid = 1, invalid = 0)
        # Invalid: NaN, inf, or very large distances
        mask = np.ones((L, L), dtype=np.float32)
        mask[np.isnan(dist) | np.isinf(dist) | (dist > 100)] = 0
        
        return (
            torch.from_numpy(pair_feat).float(),
            torch.from_numpy(dist_bins).long(),
            torch.from_numpy(mask).float(),
        )
    
    def _build_pair_features(self, feat: np.ndarray) -> np.ndarray:
        """
        Build pairwise features from per-residue features.
        
        WHAT: Combine features of residue pairs
        WHY:  Distance D[i,j] depends on both residues i and j
        HOW:  Concatenate [feat[i], feat[j], |i-j|/L]
        """
        L, d = feat.shape
        
        # Expand to (L, 1, d) and (1, L, d)
        row = feat[:, np.newaxis, :]
        col = feat[np.newaxis, :, :]
        
        # Tile to (L, L, d)
        row = np.tile(row, (1, L, 1))
        col = np.tile(col, (L, 1, 1))
        
        # Sequence separation
        positions = np.arange(L)
        seq_sep = np.abs(positions[:, np.newaxis] - positions[np.newaxis, :])
        seq_sep = seq_sep[:, :, np.newaxis].astype(np.float32) / L
        
        # Concatenate: (L, L, 2*d + 1)
        pair_feat = np.concatenate([row, col, seq_sep], axis=-1)
        
        return pair_feat.astype(np.float32)


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    
    WHAT: Pad sequences to same length within batch
    WHY:  PyTorch requires same-size tensors in batch
    HOW:  Pad with zeros, track valid positions in mask
    """
    features, dist_bins, masks = zip(*batch)
    
    # Find max length in batch
    max_len = max(f.shape[0] for f in features)
    d_pair = features[0].shape[2]
    
    B = len(batch)
    
    # Pad to max length
    padded_feat = torch.zeros(B, max_len, max_len, d_pair)
    padded_bins = torch.zeros(B, max_len, max_len, dtype=torch.long)
    padded_mask = torch.zeros(B, max_len, max_len)
    
    for i, (f, d, m) in enumerate(zip(features, dist_bins, masks)):
        L = f.shape[0]
        padded_feat[i, :L, :L, :] = f
        padded_bins[i, :L, :L] = d
        padded_mask[i, :L, :L] = m
    
    return padded_feat, padded_bins, padded_mask


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train for one epoch.
    
    WHAT: Single pass through training data
    WHY:  Update model weights on all training examples
    HOW:  For each batch: forward → loss → backward → optimizer step
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    for features, dist_bins, mask in train_loader:
        features = features.to(device)
        dist_bins = dist_bins.to(device)
        mask = mask.to(device)
        
        # Forward pass
        logits = model(features)  # (B, L, L, num_bins)
        
        # Cross-entropy loss for distogram
        B, L, _, num_bins = logits.shape
        logits_flat = logits.reshape(-1, num_bins)  # (B*L*L, num_bins)
        targets_flat = dist_bins.reshape(-1)  # (B*L*L,)
        mask_flat = mask.reshape(-1)  # (B*L*L,)
        
        loss_per_elem = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = (loss_per_elem * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validate model.
    
    WHAT: Evaluate model on validation set
    WHY:  Monitor generalization, prevent overfitting
    HOW:  Forward pass only (no gradients), compute loss + RMSE
    
    Returns:
        (val_loss, distance_rmse)
    """
    model.eval()
    total_loss = 0
    total_rmse = 0
    n_batches = 0
    
    # Bin centers for expected distance (63 bins from 2-22 Angstroms)
    # Model outputs 63 bins, so we need 63 centers
    bin_edges = np.linspace(2.0, 22.0, 64)  # 64 edges = 63 bins
    bin_centers = torch.tensor(
        (bin_edges[:-1] + bin_edges[1:]) / 2,  # 63 centers
        dtype=torch.float32
    )
    
    with torch.no_grad():
        for features, dist_bins, mask in val_loader:
            features = features.to(device)
            dist_bins = dist_bins.to(device)
            mask = mask.to(device)
            
            # Forward pass
            logits = model(features)
            probs = F.softmax(logits, dim=-1)
            
            # Loss
            B, L, _, num_bins = logits.shape
            logits_flat = logits.reshape(-1, num_bins)
            targets_flat = dist_bins.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            loss_per_elem = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            loss = (loss_per_elem * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            
            # Expected distance
            bin_centers_gpu = bin_centers.to(device)
            pred_dist = (probs * bin_centers_gpu).sum(dim=-1)  # (B, L, L)
            
            # True distance from bin centers
            true_dist = bin_centers_gpu[dist_bins.clamp(0, len(bin_centers_gpu) - 1)]
            
            # RMSE
            sq_err = ((pred_dist - true_dist) ** 2) * mask
            rmse = torch.sqrt(sq_err.sum() / (mask.sum() + 1e-8))
            
            total_loss += loss.item()
            total_rmse += rmse.item()
            n_batches += 1
    
    return total_loss / n_batches, total_rmse / n_batches


def main():
    """Main training function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train RNA distance predictor')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 2: TRAIN ML DISTANCE PREDICTOR")
    print("=" * 70)
    print()
    
    # Configuration
    config = {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'epochs': 50,
        'patience': 10,
        'max_length': 300,  # Crop sequences longer than this
        'hidden_dim': 64,   # Smaller model for faster training
        'num_blocks': 16,   # Fewer blocks for efficiency
    }
    
    # Paths
    data_dir = PROJECT_ROOT / "output" / "training_data"
    checkpoint_dir = PROJECT_ROOT / "output" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_distance_predictor.pt"
    
    # Check for resume
    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    
    if args.resume and checkpoint_path.exists():
        log_change(
            what="Resuming training from checkpoint",
            why="Continue from saved state to avoid retraining",
            how=f"Loading model and optimizer from {checkpoint_path}"
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        config = checkpoint.get('config', config)  # Use saved config if available
        logger.info(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        
        # Load history if exists
        history_path = checkpoint_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            logger.info(f"Loaded training history with {len(history['train_loss'])} epochs")
    else:
        log_change(
            what="Training distance predictor model",
            why="Learn RNA distance patterns from 4,572 training structures",
            how=f"ResNet2D with {config['num_blocks']} blocks, batch_size={config['batch_size']}"
        )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load data
    log_change(
        what="Loading prepared training data",
        why="Use pre-computed features and distance matrices",
        how="Load pickle files from output/training_data/"
    )
    
    with open(data_dir / "train_features.pkl", 'rb') as f:
        train_features = pickle.load(f)
    with open(data_dir / "train_distances.pkl", 'rb') as f:
        train_distances = pickle.load(f)
    with open(data_dir / "val_features.pkl", 'rb') as f:
        val_features = pickle.load(f)
    with open(data_dir / "val_distances.pkl", 'rb') as f:
        val_distances = pickle.load(f)
    
    logger.info(f"Train: {len(train_features)} structures")
    logger.info(f"Val: {len(val_features)} structures")
    
    # Create datasets
    train_dataset = RNADistanceDataset(
        train_features, train_distances,
        max_length=config['max_length']
    )
    val_dataset = RNADistanceDataset(
        val_features, val_distances,
        max_length=config['max_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Create model
    log_change(
        what="Creating DistancePredictor model",
        why="ResNet2D with dilated convolutions captures long-range dependencies",
        how=f"input_dim=41 (2*20+1), hidden_dim={config['hidden_dim']}, {config['num_blocks']} blocks"
    )
    
    from rna_tbm.models.distance_predictor import DistancePredictor
    
    # Input dim: 2*20 (pair features) + 1 (seq separation) = 41
    model = DistancePredictor(
        input_dim=41,
        hidden_dim=config['hidden_dim'],
        num_blocks=config['num_blocks'],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
    )
    
    # Load checkpoint weights if resuming
    if args.resume and checkpoint_path.exists():
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Advance scheduler to correct position
        for _ in range(start_epoch):
            scheduler.step()
        logger.info(f"Loaded model weights and optimizer state from epoch {start_epoch - 1}")
    
    # Training loop
    log_change(
        what="Starting training loop",
        why="Learn distance prediction from training data",
        how=f"Epochs {start_epoch + 1}-{config['epochs']} with early stopping (patience={config['patience']})"
    )
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_rmse = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log progress
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_rmse={val_rmse:.2f}Å | lr={lr:.2e} | {epoch_time:.1f}s"
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'config': config,
            }
            torch.save(checkpoint, checkpoint_dir / "best_distance_predictor.pt")
            logger.info(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save training history
    with open(checkpoint_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Summary
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation RMSE: {min(history['val_rmse']):.2f} Å")
    print(f"Checkpoint saved to: {checkpoint_dir}/best_distance_predictor.pt")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
