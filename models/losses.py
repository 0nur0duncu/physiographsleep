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
    """Focal loss with optional class weights, per-class gamma, and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        per_class_gamma: dict[int, float] | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.per_class_gamma = per_class_gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)

        if self.per_class_gamma:
            gamma = torch.full_like(ce, self.gamma)
            for cls_id, cls_gamma in self.per_class_gamma.items():
                gamma[targets == cls_id] = cls_gamma
            focal = ((1 - pt) ** gamma) * ce
        else:
            focal = ((1 - pt) ** self.gamma) * ce

        return focal.mean()


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss for PhysioGraphSleep."""

    def __init__(
        self,
        config: LossConfig,
        class_weights: torch.Tensor | None = None,
        per_class_gamma: dict[int, float] | None = None,
    ):
        super().__init__()
        self.config = config

        self.focal = FocalLoss(
            gamma=config.focal_gamma,
            weight=class_weights,
            label_smoothing=config.label_smoothing,
            per_class_gamma=per_class_gamma,
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def update_adaptive_weights(
        self,
        class_f1: np.ndarray,
        K: float = 10.0,
        gamma: float = 3.0,
        eps: float = 1e-4,
    ) -> None:
        """Update focal loss class weights based on per-class F1 scores.

        Adaptive weighting inspired by SeriesSleepNet (Lee et al., 2023):
        W_i = K / (CF_i + eps)^gamma
        """
        f1_clipped = np.clip(class_f1, eps, None)
        adaptive = K / np.power(f1_clipped, gamma)
        adaptive = adaptive / adaptive.sum() * len(adaptive)
        weight_tensor = torch.tensor(adaptive, dtype=torch.float32)
        if self.focal.weight is not None:
            self.focal.weight.copy_(weight_tensor.to(self.focal.weight.device))
        else:
            device = next(self.parameters()).device
            self.focal.register_buffer("weight", weight_tensor.to(device))

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
