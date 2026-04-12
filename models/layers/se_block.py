"""Squeeze-and-Excitation block for channel recalibration."""

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """SE block: global pool → squeeze → excite → scale."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T) re-weighted
        """
        scale = self.pool(x).squeeze(-1)  # (B, C)
        scale = self.fc(scale).unsqueeze(-1)  # (B, C, 1)
        return x * scale
