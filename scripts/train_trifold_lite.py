#!/usr/bin/env python3
"""
Train TriFold-Lite: Enhanced RNA Distance Predictor

This script trains the TriFold-Lite model which adds:
1. Triangle Update modules for geometric consistency
2. Confidence estimation for DDG weighting
3. Improved architecture with interleaved triangle/conv blocks

Can resume from existing DistancePredictor checkpoint or train fresh.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rna_tbm.models.trifold_lite import TriFoldLite, TriangleUpdate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def log_section(title: str):
    """Log a section header."""
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


class RNADistanceDataset(Dataset):
    """PyTorch Dataset for RNA distance prediction."""
    
    def __init__(
        self,
        features: List[np.ndarray],
        distances: List[np.ndarray],
        max_length: int = 300,
        num_bins: int = 63,
    ):
        self.features = features
        self.distances = distances
        self.max_length = max_length
        self.num_bins = num_bins
        self.bin_edges = np.linspace(2.0, 22.0, num_bins + 1)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.features[idx]
        dist = self.distances[idx]
        L = feat.shape[0]
        
        # Crop if too long
        if L > self.max_length:
            start = np.random.randint(0, L - self.max_length)
            feat = feat[start:start + self.max_length]
            dist = dist[start:start + self.max_length, start:start + self.max_length]
            L = self.max_length
        
        # Build pairwise features
        pair_feat = self._build_pair_features(feat)
        
        # Discretize distances
        dist_bins = np.digitize(dist, self.bin_edges) - 1
        dist_bins = np.clip(dist_bins, 0, self.num_bins - 1)
        
        # Mask
        mask = np.ones((L, L), dtype=np.float32)
        mask[np.isnan(dist) | np.isinf(dist) | (dist > 100)] = 0
        
        return (
            torch.from_numpy(pair_feat).float(),
            torch.from_numpy(dist_bins).long(),
            torch.from_numpy(mask).float(),
        )
    
    def _build_pair_features(self, feat: np.ndarray) -> np.ndarray:
        L, d = feat.shape
        row = np.tile(feat[:, np.newaxis, :], (1, L, 1))
        col = np.tile(feat[np.newaxis, :, :], (L, 1, 1))
        
        positions = np.arange(L)
        seq_sep = np.abs(positions[:, np.newaxis] - positions[np.newaxis, :])
        seq_sep = seq_sep[:, :, np.newaxis].astype(np.float32) / L
        
        return np.concatenate([row, col, seq_sep], axis=-1).astype(np.float32)


def collate_fn(batch):
    """Custom collate for variable-length sequences."""
    features, dist_bins, masks = zip(*batch)
    max_len = max(f.shape[0] for f in features)
    d_pair = features[0].shape[2]
    B = len(batch)
    
    padded_feat = torch.zeros(B, max_len, max_len, d_pair)
    padded_bins = torch.zeros(B, max_len, max_len, dtype=torch.long)
    padded_mask = torch.zeros(B, max_len, max_len)
    
    for i, (f, d, m) in enumerate(zip(features, dist_bins, masks)):
        L = f.shape[0]
        padded_feat[i, :L, :L, :] = f
        padded_bins[i, :L, :L] = d
        padded_mask[i, :L, :L] = m
    
    return padded_feat, padded_bins, padded_mask


def train_epoch(model, loader, optimizer, device, num_bins):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_conf_loss = 0
    n_batches = 0
    
    for features, dist_bins, mask in loader:
        features = features.to(device)
        dist_bins = dist_bins.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, confidence = model(features, return_confidence=True)
        
        # Distance prediction loss (cross-entropy)
        B, L, _, num_bins_out = logits.shape
        logits_flat = logits.reshape(-1, num_bins_out)
        targets_flat = dist_bins.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        loss_per_elem = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        dist_loss = (loss_per_elem * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        
        # Confidence calibration loss
        # High confidence should correlate with low error
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            bin_edges = torch.linspace(2.0, 22.0, num_bins + 1, device=device)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            pred_dist = (probs * bin_centers).sum(dim=-1)
            true_dist = bin_centers[dist_bins.clamp(0, num_bins - 1)]
            error = torch.abs(pred_dist - true_dist)
        
        # Confidence should be high when error is low
        # Target: confidence = 1 - normalized_error
        normalized_error = error / 10.0  # Normalize by ~10Å
        target_conf = torch.clamp(1.0 - normalized_error, 0.0, 1.0)
        conf_loss = F.mse_loss(confidence * mask, target_conf * mask)
        
        # Combined loss
        loss = dist_loss + 0.1 * conf_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += dist_loss.item()
        total_conf_loss += conf_loss.item()
        n_batches += 1
    
    return total_loss / n_batches, total_conf_loss / n_batches


def validate(model, loader, device, num_bins):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_rmse = 0
    n_batches = 0
    
    bin_edges = np.linspace(2.0, 22.0, num_bins + 1)
    bin_centers = torch.tensor(
        (bin_edges[:-1] + bin_edges[1:]) / 2,
        dtype=torch.float32,
        device=device
    )
    
    with torch.no_grad():
        for features, dist_bins, mask in loader:
            features = features.to(device)
            dist_bins = dist_bins.to(device)
            mask = mask.to(device)
            
            logits, confidence = model(features, return_confidence=True)
            probs = F.softmax(logits, dim=-1)
            
            # Loss
            B, L, _, num_bins_out = logits.shape
            logits_flat = logits.reshape(-1, num_bins_out)
            targets_flat = dist_bins.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            loss_per_elem = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            loss = (loss_per_elem * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            
            # RMSE
            pred_dist = (probs * bin_centers).sum(dim=-1)
            true_dist = bin_centers[dist_bins.clamp(0, len(bin_centers) - 1)]
            
            sq_err = ((pred_dist - true_dist) ** 2) * mask
            rmse = torch.sqrt(sq_err.sum() / (mask.sum() + 1e-8))
            
            total_loss += loss.item()
            total_rmse += rmse.item()
            n_batches += 1
    
    return total_loss / n_batches, total_rmse / n_batches


def main():
    parser = argparse.ArgumentParser(description='Train TriFold-Lite')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--from-distpredictor', type=str, default=None,
                        help='Initialize from DistancePredictor checkpoint')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-conv-blocks', type=int, default=12)
    parser.add_argument('--num-triangle-layers', type=int, default=4)
    args = parser.parse_args()
    
    log_section("TriFold-Lite Training")
    
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'patience': 10,
        'max_length': 300,
        'hidden_dim': args.hidden_dim,
        'num_conv_blocks': args.num_conv_blocks,
        'num_triangle_layers': args.num_triangle_layers,
        'input_dim': 41,
        'num_bins': 63,
    }
    
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Paths
    data_dir = PROJECT_ROOT / "output" / "training_data"
    checkpoint_dir = PROJECT_ROOT / "output" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "trifold_lite.pt"
    
    # Load data
    log_section("Loading Training Data")
    
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
    
    # Datasets
    train_dataset = RNADistanceDataset(
        train_features, train_distances, max_length=config['max_length']
    )
    val_dataset = RNADistanceDataset(
        val_features, val_distances, max_length=config['max_length']
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    # Model
    log_section("Creating TriFold-Lite Model")
    
    model = TriFoldLite(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_conv_blocks=config['num_conv_blocks'],
        num_triangle_layers=config['num_triangle_layers'],
        num_bins=config['num_bins'],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Resume or initialize
    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'conf_loss': []}
    
    if args.resume and checkpoint_path.exists():
        log_section("Resuming from Checkpoint")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        history = ckpt.get('history', history)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        
        for _ in range(start_epoch):
            scheduler.step()
            
    elif args.from_distpredictor:
        log_section("Initializing from DistancePredictor")
        old_ckpt = torch.load(args.from_distpredictor, map_location='cpu')
        old_state = old_ckpt['model_state_dict']
        
        # Try to copy matching weights
        new_state = model.state_dict()
        copied = 0
        for key in new_state.keys():
            # Map old keys to new (input_proj, blocks.X.*, distance_head)
            old_key = key
            if key in old_state and new_state[key].shape == old_state[key].shape:
                new_state[key] = old_state[key]
                copied += 1
        
        model.load_state_dict(new_state)
        logger.info(f"Copied {copied}/{len(new_state)} weights from DistancePredictor")
    
    # Training loop
    log_section(f"Training (Epochs {start_epoch+1}-{config['epochs']})")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()
        
        train_loss, conf_loss = train_epoch(model, train_loader, optimizer, device, config['num_bins'])
        val_loss, val_rmse = validate(model, val_loader, device, config['num_bins'])
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"rmse={val_rmse:.2f}Å | conf={conf_loss:.4f} | "
            f"lr={lr:.2e} | {epoch_time:.1f}s"
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['conf_loss'].append(conf_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'config': config,
                'history': history,
            }, checkpoint_path)
            logger.info(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Summary
    log_section("Training Complete")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation RMSE: {min(history['val_rmse']):.2f}Å")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
