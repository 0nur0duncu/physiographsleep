"""Multi-task loss functions.

Combines:
  - Focal loss (class-balanced) for main stage classification
  - BCE for boundary detection
  - CE for prev/next stage prediction
  - BCE for N1-aux
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.train_config import LossConfig


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
