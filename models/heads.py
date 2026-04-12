"""Multi-task classification heads.

Heads:
  1. Stage head     — 5-class sleep stage classification
  2. Boundary head  — binary (transition epoch or not)
  3. Prev-stage head — 5-class previous epoch stage
  4. Next-stage head — 5-class next epoch stage
  5. N1-aux head    — binary (N1 vs non-N1)
"""

import torch
import torch.nn as nn

from ..configs.model_config import HeadsConfig


class ClassificationHead(nn.Module):
    """Simple classification head: LayerNorm → Dropout → Linear."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiTaskHeads(nn.Module):
    """All prediction heads for PhysioGraphSleep."""

    def __init__(self, config: HeadsConfig):
        super().__init__()
        d = config.input_dim

        self.stage_head = ClassificationHead(d, config.num_classes, config.dropout)
        self.boundary_head = ClassificationHead(d, 1, config.dropout)
        self.prev_head = ClassificationHead(d, config.num_classes, config.dropout)
        self.next_head = ClassificationHead(d, config.num_classes, config.dropout)
        self.n1_head = ClassificationHead(d, 1, config.dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, D) center epoch features

        Returns:
            dict with keys: stage, boundary, prev, next, n1
        """
        return {
            "stage": self.stage_head(x),        # (B, 5)
            "boundary": self.boundary_head(x).squeeze(-1),  # (B,)
            "prev": self.prev_head(x),           # (B, 5)
            "next": self.next_head(x),           # (B, 5)
            "n1": self.n1_head(x).squeeze(-1),   # (B,)
        }
