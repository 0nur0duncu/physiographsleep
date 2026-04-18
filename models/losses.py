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
    """Focal loss with optional class weights and label smoothing."""

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
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
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

    def update_adaptive_weights(
        self,
        class_f1: np.ndarray,
        K: float = 10.0,
        gamma: float = 3.0,
        eps: float = 1e-4,
        min_weight: float = 0.5,
        max_weight: float = 5.0,
        ema_momentum: float = 0.7,
    ) -> None:
        """Update focal loss class weights based on per-class F1 scores.

        Adaptive weighting inspired by SeriesSleepNet (Lee et al., 2023):
            W_i = K / (CF_i + eps)^gamma

        Stabilized with:
          - per-class min/max clipping (defaults 0.5 .. 5.0) to prevent the
            K / F1^gamma blow-up when a class F1 is near zero,
          - re-normalization to sum=num_classes after clipping,
          - EMA smoothing against current weights to damp epoch-to-epoch
            fluctuation (momentum=0.7 keeps 70% of previous weights).
        """
        f1_clipped = np.clip(class_f1, eps, None)
        adaptive = K / np.power(f1_clipped, gamma)
        adaptive = adaptive / adaptive.sum() * len(adaptive)
        adaptive = np.clip(adaptive, min_weight, max_weight)
        adaptive = adaptive / adaptive.sum() * len(adaptive)
        new_w = torch.tensor(adaptive, dtype=torch.float32)

        if self.focal.weight is not None:
            prev = self.focal.weight.detach().to(new_w.device)
            blended = ema_momentum * prev + (1.0 - ema_momentum) * new_w
            blended = blended / blended.sum() * len(blended)
            self.focal.weight.copy_(blended.to(self.focal.weight.device))
        else:
            device = next(self.parameters()).device
            self.focal.register_buffer("weight", new_w.to(device))

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses = {}

        losses["stage"] = self.focal(predictions["stage"], targets["label"])
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
