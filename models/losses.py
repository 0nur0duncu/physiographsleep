"""Multi-task loss functions.

Combines:
  - Focal loss (class-balanced) for main stage classification
  - BCE for boundary detection
  - CE for prev/next stage prediction
  - BCE for N1-aux

Weight strategies (configured via LossConfig.weight_strategy):
  - "none": Uniform weights (default).
  - "inverse_freq": Static 1/count weights.
  - "class_balanced": Cui et al. CVPR 2019 effective-number weights.
  - "adaptive_f1": SeriesSleepNet (Frontiers 2023) dynamic per-epoch
    F1-based weights.  First `adaptive_warmup` epochs use uniform weights,
    then W_i = 1 − log_K(CF_i)^γ where CF_i is class i's train F1.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.train_config import LossConfig


# ------------------------------------------------------------------
# Weight computation helpers
# ------------------------------------------------------------------

def compute_inverse_freq_weights(class_counts: np.ndarray) -> np.ndarray:
    """Classic inverse-frequency weights: w_i = 1/n_i, normalised ×K."""
    weights = 1.0 / (class_counts.astype(np.float64) + 1e-6)
    weights = weights / weights.sum() * len(class_counts)
    return weights.astype(np.float32)


def compute_class_balanced_weights(
    class_counts: np.ndarray, beta: float = 0.9999,
) -> np.ndarray:
    """Effective-number class-balanced weights (Cui et al. CVPR 2019).

    weight_i = (1 − β) / (1 − β^n_i)
    Normalised so weights sum to num_classes.
    """
    effective_num = 1.0 - np.power(beta, class_counts.astype(np.float64))
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * len(class_counts)
    return weights.astype(np.float32)


def compute_adaptive_f1_weights(
    per_class_f1: np.ndarray,
    K: float = 10.0,
    gamma: float = 3.0,
) -> np.ndarray:
    """Adaptive F1-based weights (SeriesSleepNet, Frontiers 2023).

    W_i = 1 − log_K(CF_i)^γ

    Low F1 → high weight, high F1 → weight near 1.  Gradient-safe:
    CF_i is clamped to [1e-4, 1.0].
    """
    f1 = np.clip(per_class_f1, 1e-4, 1.0).astype(np.float64)
    log_base = math.log(K)
    log_f1 = np.log(f1) / log_base          # log_K(f1)
    weights = 1.0 - np.power(log_f1, gamma)  # log_f1 is ≤0, so log_f1^γ ≤0 → 1-neg = ≥1
    # Normalise to mean=1 so total loss magnitude is stable
    weights = weights / (weights.mean() + 1e-8)
    return weights.astype(np.float32)


class FocalLoss(nn.Module):
    """Focal loss with optional class weights and label smoothing.

    Supports both hard integer targets and soft probability targets
    (e.g. produced by Mixup). For soft targets we use the standard
    cross-entropy form −Σ y_soft · log_softmax(logits) and reweight
    sample-wise by (1 − p_t)^γ where p_t is the model's confidence on
    the soft-label argmax — a faithful focal extension that keeps the
    same family of class weights.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def update_class_weights(self, new_weights: torch.Tensor) -> None:
        """Replace class weights in-place (used by adaptive strategies)."""
        device = self.weight.device if self.weight is not None else new_weights.device
        self.weight = new_weights.to(device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Hard integer targets → standard focal CE
        if targets.dim() == 1 or targets.dtype in (torch.int64, torch.int32, torch.long):
            ce = F.cross_entropy(
                logits, targets,
                weight=self.weight,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.gamma) * ce
            return focal.mean()

        # Soft targets (B, K) — Mixup path
        log_probs = F.log_softmax(logits, dim=-1)
        if self.label_smoothing > 0:
            n_classes = targets.shape[-1]
            targets = (
                targets * (1.0 - self.label_smoothing)
                + self.label_smoothing / n_classes
            )
        if self.weight is not None:
            log_probs = log_probs * self.weight.unsqueeze(0)
        per_sample_ce = -(targets * log_probs).sum(dim=-1)  # (B,)
        # Focal modulation on the *target-weighted* confidence
        with torch.no_grad():
            probs = log_probs.exp()
            pt = (targets * probs).sum(dim=-1).clamp(min=1e-6, max=1.0)
        focal = ((1 - pt) ** self.gamma) * per_sample_ce
        return focal.mean()


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss for PhysioGraphSleep."""

    def __init__(
        self,
        config: LossConfig,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config

        self.focal = FocalLoss(
            gamma=config.focal_gamma,
            weight=class_weights,
            label_smoothing=config.label_smoothing,
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def update_focal_weights(self, new_weights: torch.Tensor) -> None:
        """Update focal loss class weights (called by trainer for adaptive)."""
        self.focal.update_class_weights(new_weights)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute weighted multi-task loss for the center epoch.

        If `targets["label_soft"]` is present (Mixup path), the stage
        focal loss consumes the soft target distribution; auxiliary
        heads still use the original hard center labels.
        """
        losses = {}
        stage_target = targets.get("label_soft", targets["label"])
        losses["stage"] = self.focal(predictions["stage"], stage_target)
        losses["boundary"] = self.bce(predictions["boundary"], targets["boundary"])
        losses["prev"] = self.ce(predictions["prev"], targets["prev_label"])
        losses["next"] = self.ce(predictions["next"], targets["next_label"])
        losses["n1"] = self.bce(predictions["n1"], targets["n1_label"])

        total = (
            self.config.stage_weight * losses["stage"]
            + self.config.boundary_weight * losses["boundary"]
            + self.config.prev_stage_weight * losses["prev"]
            + self.config.next_stage_weight * losses["next"]
            + self.config.n1_aux_weight * losses["n1"]
        )
        losses["total"] = total
        return losses
