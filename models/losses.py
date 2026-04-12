"""Multi-task loss functions.

Combines:
  - Focal loss (class-balanced) for main stage classification
  - BCE for boundary detection
  - CE for prev/next stage prediction
  - BCE for N1-aux
"""

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
        """
        Args:
            logits: (B, C) raw logits
            targets: (B,) integer labels

        Returns:
            scalar loss
        """
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

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict from MultiTaskHeads
            targets: dict with label, boundary, prev_label, next_label, n1_label

        Returns:
            dict with individual losses and total
        """
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
