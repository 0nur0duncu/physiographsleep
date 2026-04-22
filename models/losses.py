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
    """Adaptive F1-based class weights.

    Original SeriesSleepNet formula (Frontiers 2023) was
        W_i = 1 - log_K(CF_i)^gamma
    which in practice yields nearly uniform weights: for CF=0.50 and
    (K=10, gamma=3) the weight is only 1.027, i.e. < 3% spread between
    the worst and the best class. The signal is too weak to break
    minority-class plateaus.

    We instead use
        W_i = 1.0 + K * (1 - CF_i) ** gamma
    which is monotone and interpretable: CF=0.50 -> 1 + 10*0.125 = 2.25,
    CF=0.95 -> 1 + 10*0.000125 = 1.001. Low F1 classes get a meaningful
    boost, high F1 classes keep weight ~1.

    April 22 2026 — CRITICAL FIX. The previous implementation divided
    weights by their mean so the post-normalisation mean was exactly
    1.0. In practice this meant: whenever ONE class was much harder
    than the rest (e.g. N1 F1=0.44 with raw weight ~4.9 while the
    other four classes sat at raw weight 1.0-1.7), the mean climbed
    to ~2.0 and the four easy classes got divided DOWN to weights
    0.5-0.8 — i.e. their gradient signal was HALVED. Observed effect
    on run jzaveo41: Val MF1 peaked at epoch 2 (the first adaptive
    epoch, 0.7604) and regressed to 0.69 on epoch 3 when mean-norm
    kicked in; N3 F1 fell from 0.886 to 0.880, Wake F1 0.831 -> 0.772.
    All non-N1 classes lost performance because they were being
    under-trained. The fix is to remove the mean division entirely so
    the raw formula (guaranteed >= 1.0) flows through, then clamp the
    maximum to avoid runaway minority-class weights when F1 is near 0.
    Every class now keeps at least a full gradient signal (w >= 1);
    only the hard class receives a genuine boost on top.
    """
    f1 = np.clip(per_class_f1, 0.0, 1.0).astype(np.float64)
    weights = 1.0 + K * np.power(1.0 - f1, gamma)
    weights = np.clip(weights, 1.0, 5.0)
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
        per_sample_ce = -(targets * log_probs).sum(dim=-1)  # (B,)
        if self.weight is not None:
            # PyTorch cross_entropy(weight=w) semantics for hard targets
            # is loss_i = w[target_i] * CE_i. Its soft-target analogue is
            # a per-sample weight equal to sum_k target[i,k] * w[k].
            # Previous implementation scaled log_probs by w across all
            # classes, which merely shifts the prediction distribution
            # and does NOT reweight minority classes — a silent bug that
            # neutralised class weights whenever Mixup was active.
            w = self.weight.to(log_probs.device)
            sample_w = (targets * w.unsqueeze(0)).sum(dim=-1)       # (B,)
            per_sample_ce = per_sample_ce * sample_w
        # Focal modulation on the target-weighted confidence
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

        # Deep-supervision for GNN branch when fusion is active.
        # Without this, detach_gnn_for_lambda=True severs all stage-loss
        # gradient to the GNN encoder, causing the branch to collapse.
        if "stage_gnn" in predictions and self.config.gnn_stage_weight > 0:
            losses["stage_gnn"] = self.focal(predictions["stage_gnn"], stage_target)

        total = (
            self.config.stage_weight * losses["stage"]
            + self.config.boundary_weight * losses["boundary"]
            + self.config.prev_stage_weight * losses["prev"]
            + self.config.next_stage_weight * losses["next"]
            + self.config.n1_aux_weight * losses["n1"]
        )
        if "stage_gnn" in losses:
            total = total + self.config.gnn_stage_weight * losses["stage_gnn"]
        losses["total"] = total
        return losses
