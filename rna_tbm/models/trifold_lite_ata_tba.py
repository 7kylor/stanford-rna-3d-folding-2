"""
TriFold-Lite ATA+TBA
--------------------
One-file implementation of Anchor Triangle Attention (ATA) with
Template-Biased Attention (TBA). This keeps the original TriFold-Lite
pairwise conv backbone and adds a lightweight attention mechanism that
scales as O(L^2 * K) with K anchors.

Design goals:
- Simple, fast, and geometry-aware.
- Template-aware biasing without full attention cost.
- Compatible with Python 3.14+ and uv workflows.

Speed notes:
- Uses small anchor sets and fused einsum patterns.
- Works on CPU, CUDA, or Apple MPS (M4 Max).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_runtime() -> None:
    """Set safe performance flags for modern PyTorch builds."""
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def select_anchors(
    length: int,
    num_anchors: int,
    mode: str = "stride",
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Anchor selection for ATA.

    mode="stride" yields deterministic anchors.
    mode="random" yields stochastic anchors for training regularization.
    """
    if length <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    k = max(1, min(num_anchors, length))

    if mode == "random":
        return torch.randperm(length, device=device, generator=generator)[:k]

    # Default: stride-based anchors for stable inference.
    stride = max(1, length // k)
    idx = torch.arange(0, length, step=stride, device=device, dtype=torch.long)[:k]
    if idx.numel() < k:
        pad = torch.full((k - idx.numel(),), length - 1, device=device, dtype=torch.long)
        idx = torch.cat([idx, pad], dim=0)
    return idx


@dataclass
class ATATBAConfig:
    input_dim: int = 41
    hidden_dim: int = 64
    num_bins: int = 63
    num_conv_blocks: int = 12
    attn_dim: int = 32
    num_anchors: int = 16
    attn_every: int = 2
    num_attn_blocks: Optional[int] = None
    anchor_mode: str = "stride"
    anchor_dropout: float = 0.0
    attn_dropout: float = 0.0
    template_sigma: float = 12.0
    template_bias_mode: str = "consistency"  # "consistency" or "path"
    symmetrize_pairs: bool = True
    use_metric_repair: bool = True
    repair_tau: float = 0.15


class ResBlock2D(nn.Module):
    """Standard 2D residual block with dilated convolutions."""

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


class ConfidenceHead(nn.Module):
    """Small CNN head that predicts per-pair confidence."""

    def __init__(self, num_bins: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_bins, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, distance_logits: torch.Tensor) -> torch.Tensor:
        # distance_logits: (B, L, L, num_bins)
        x = distance_logits.permute(0, 3, 1, 2)  # (B, num_bins, L, L)
        conf = self.net(x)
        return conf.squeeze(1)


class TemplateGate(nn.Module):
    """
    Predicts a gate value in [0,1] based on template coverage and quality.
    This keeps template bias active only when template information is reliable.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        template_dist: torch.Tensor,
        template_quality: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Coverage is the fraction of non-zero template distances.
        mask = (template_dist > 0).float()
        coverage = mask.mean(dim=(1, 2))
        if template_quality is None:
            template_quality = coverage
        length = template_dist.shape[-1]
        length_norm = torch.full_like(coverage, float(length) / 512.0)
        feats = torch.stack([coverage, template_quality, length_norm], dim=-1)
        gate = self.net(feats)
        return gate.view(-1, 1, 1, 1)


class AnchorTriangleAttention(nn.Module):
    """
    Anchor Triangle Attention (ATA) with optional Template-Biased Attention (TBA).

    ATA:
      For each pair (i, j), attend over anchor nodes k using pair features
      from (i, k) and (k, j). This approximates triangle attention at
      O(L^2 * K) cost.

    TBA:
      Adds a bias term derived from template distances to prioritize anchors
      consistent with known template geometry.
    """

    def __init__(
        self,
        dim: int,
        attn_dim: int,
        num_anchors: int,
        anchor_mode: str,
        attn_dropout: float,
        template_sigma: float,
        template_bias_mode: str,
    ):
        super().__init__()
        self.q_proj = nn.Linear(dim, attn_dim, bias=False)
        self.left_proj = nn.Linear(dim, attn_dim, bias=False)
        self.right_proj = nn.Linear(dim, attn_dim, bias=False)
        self.v_left = nn.Linear(dim, attn_dim, bias=False)
        self.v_right = nn.Linear(dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(attn_dropout)

        self.num_anchors = num_anchors
        self.anchor_mode = anchor_mode
        self.template_sigma = template_sigma
        self.template_bias_mode = template_bias_mode
        self.template_gate = TemplateGate()

    def forward(
        self,
        pair_repr: torch.Tensor,
        template_dist: Optional[torch.Tensor] = None,
        template_quality: Optional[torch.Tensor] = None,
        anchor_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, length, _, dim = pair_repr.shape
        if anchor_idx is None:
            anchor_idx = select_anchors(
                length,
                self.num_anchors,
                mode=self.anchor_mode,
                device=pair_repr.device,
            )
        k = anchor_idx.numel()

        # Query from (i, j), keys from (i, k) and (k, j).
        q = self.q_proj(pair_repr)  # (B, L, L, A)
        left = self.left_proj(pair_repr[:, :, anchor_idx, :])  # (B, L, K, A)
        right = self.right_proj(pair_repr[:, anchor_idx, :, :])  # (B, K, L, A)

        # scores_left[i,j,k] = dot(q[i,j], left[i,k])
        scores = torch.einsum("b i j a, b i k a -> b i j k", q, left)
        scores = scores + torch.einsum("b i j a, b k j a -> b i j k", q, right)
        scores = scores / math.sqrt(left.shape[-1])

        if template_dist is not None:
            # Template bias: prefer anchors whose path matches template geometry.
            t_left = template_dist[:, :, anchor_idx]  # (B, L, K)
            t_right = template_dist[:, anchor_idx, :]  # (B, K, L)
            t_right = t_right.permute(0, 2, 1)  # (B, L, K) for j
            t_sum = t_left.unsqueeze(2) + t_right.unsqueeze(1)  # (B, L, L, K)

            if self.template_bias_mode == "consistency":
                t_direct = template_dist.unsqueeze(-1)  # (B, L, L, 1)
                bias = -torch.abs(t_sum - t_direct) / self.template_sigma
            else:
                bias = -t_sum / self.template_sigma

            gate = self.template_gate(template_dist, template_quality)
            scores = scores + gate * bias

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Values from (i, k) and (k, j), then aggregate by attention.
        v_left = self.v_left(pair_repr[:, :, anchor_idx, :])  # (B, L, K, A)
        v_right = self.v_right(pair_repr[:, anchor_idx, :, :])  # (B, K, L, A)
        update = torch.einsum("b i j k, b i k a -> b i j a", attn, v_left)
        update = update + torch.einsum("b i j k, b k j a -> b i j a", attn, v_right)
        update = self.out_proj(update)  # (B, L, L, C)

        gate = torch.sigmoid(self.gate(pair_repr))
        return pair_repr + gate * update


class MetricRepair(nn.Module):
    """
    Soft metric repair using anchor paths. This reduces triangle inequality
    violations without a full O(L^3) projection.
    """

    def __init__(self, num_anchors: int, tau: float = 0.15, anchor_mode: str = "stride"):
        super().__init__()
        self.num_anchors = num_anchors
        self.tau = tau
        self.anchor_mode = anchor_mode

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = distances.shape
        anchor_idx = select_anchors(
            length,
            self.num_anchors,
            mode=self.anchor_mode,
            device=distances.device,
        )
        if anchor_idx.numel() == 0:
            return distances

        left = distances[:, :, anchor_idx]  # (B, L, K)
        right = distances[:, anchor_idx, :]  # (B, K, L)
        right = right.permute(0, 2, 1)  # (B, L, K)
        path = left.unsqueeze(2) + right.unsqueeze(1)  # (B, L, L, K)

        values = torch.cat([distances.unsqueeze(-1), path], dim=-1)  # (B, L, L, K+1)
        repaired = -self.tau * torch.logsumexp(-values / self.tau, dim=-1)

        # Enforce symmetry and zero diagonal for stability.
        repaired = 0.5 * (repaired + repaired.transpose(1, 2))
        repaired = repaired - torch.diag_embed(torch.diagonal(repaired, dim1=1, dim2=2))
        return repaired


class TriFoldLiteATATBA(nn.Module):
    """
    TriFold-Lite with ATA+TBA:
    - Conv backbone for local structure
    - Anchor triangle attention for geometry
    - Template-biased attention for template-rich targets
    - Optional metric repair for triangle consistency
    """

    def __init__(self, config: ATATBAConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Conv2d(config.input_dim, config.hidden_dim, 1)
        self.distance_head = nn.Conv2d(config.hidden_dim, config.num_bins, 3, padding=1)
        self.confidence_head = ConfidenceHead(config.num_bins, hidden_dim=32)

        self.attn_block = AnchorTriangleAttention(
            dim=config.hidden_dim,
            attn_dim=config.attn_dim,
            num_anchors=config.num_anchors,
            anchor_mode=config.anchor_mode,
            attn_dropout=config.attn_dropout,
            template_sigma=config.template_sigma,
            template_bias_mode=config.template_bias_mode,
        )

        self.metric_repair = MetricRepair(
            num_anchors=config.num_anchors,
            tau=config.repair_tau,
            anchor_mode=config.anchor_mode,
        )

        # Build interleaved conv and attention blocks.
        self.blocks = nn.ModuleList()
        self.block_types: list[str] = []
        dilations = [1, 2, 4, 8, 16]

        num_attn_blocks = config.num_attn_blocks
        if num_attn_blocks is None:
            num_attn_blocks = max(1, config.num_conv_blocks // max(1, config.attn_every))

        attn_added = 0
        for i in range(config.num_conv_blocks):
            dilation = dilations[i % len(dilations)]
            self.blocks.append(ResBlock2D(config.hidden_dim, dilation))
            self.block_types.append("conv")

            if (i + 1) % config.attn_every == 0 and attn_added < num_attn_blocks:
                self.blocks.append(self.attn_block)
                self.block_types.append("attn")
                attn_added += 1

    def forward(
        self,
        features: torch.Tensor,
        return_confidence: bool = True,
        template_dist: Optional[torch.Tensor] = None,
        template_quality: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: (B, L, L, input_dim)
            template_dist: (B, L, L) template distances (optional)
            template_quality: (B,) or (B, 1) scalar quality score in [0,1]
        """
        bsz, length, _, _ = features.shape
        x = features.permute(0, 3, 1, 2)  # (B, input_dim, L, L)
        x = self.input_proj(x)

        # Optional symmetrize to stabilize pair representation.
        if self.config.symmetrize_pairs:
            x = 0.5 * (x + x.transpose(2, 3))

        anchor_idx = select_anchors(
            length,
            self.config.num_anchors,
            mode=self.config.anchor_mode,
            device=features.device,
        )

        for block_type, block in zip(self.block_types, self.blocks):
            if block_type == "conv":
                x = block(x)
            else:
                # Attention operates on (B, L, L, C).
                pair = x.permute(0, 2, 3, 1)
                pair = block(
                    pair,
                    template_dist=template_dist,
                    template_quality=template_quality,
                    anchor_idx=anchor_idx,
                )
                if self.config.symmetrize_pairs:
                    pair = 0.5 * (pair + pair.transpose(1, 2))
                x = pair.permute(0, 3, 1, 2)

        logits = self.distance_head(x)  # (B, num_bins, L, L)
        logits = logits.permute(0, 2, 3, 1)  # (B, L, L, num_bins)

        confidence = None
        if return_confidence:
            confidence = self.confidence_head(logits)
        return logits, confidence

    def predict_distances(
        self,
        features: torch.Tensor,
        template_dist: Optional[torch.Tensor] = None,
        template_quality: Optional[torch.Tensor] = None,
        apply_metric_repair: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert distogram logits to expected distances.
        Optionally apply metric repair for triangle consistency.
        """
        logits, confidence = self.forward(
            features,
            return_confidence=True,
            template_dist=template_dist,
            template_quality=template_quality,
        )

        probs = F.softmax(logits, dim=-1)
        bin_edges = torch.linspace(2.0, 22.0, self.config.num_bins + 1, device=logits.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        distances = (probs * bin_centers).sum(dim=-1)

        if apply_metric_repair is None:
            apply_metric_repair = self.config.use_metric_repair
        if apply_metric_repair:
            distances = self.metric_repair(distances)
        return distances, confidence


if __name__ == "__main__":
    configure_runtime()

    # Quick smoke test with random inputs.
    config = ATATBAConfig()
    model = TriFoldLiteATATBA(config)
    features = torch.randn(2, 64, 64, config.input_dim)
    template_dist = torch.rand(2, 64, 64) * 20.0
    logits, conf = model(features, template_dist=template_dist)
    distances, confidence = model.predict_distances(features, template_dist=template_dist)

    print("logits:", logits.shape)
    print("conf:", conf.shape if conf is not None else None)
    print("distances:", distances.shape)
    print("confidence:", confidence.shape)
