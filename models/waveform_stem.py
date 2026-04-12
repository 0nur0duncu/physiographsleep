"""WaveformStem — multi-scale CNN for extracting patch morphology tokens.

Architecture:
  Input(1, 3000) → Conv1d(1→32, k=25, s=2) → GroupNorm → GELU
  → 3 parallel DSConv branches (k=25/75/225)
  → Concat → 1×1 projection → 96ch
  → AdaptiveAvgPool → 6 patch tokens
  Output: (B, 6, 96)
"""

import torch
import torch.nn as nn

from ..configs.model_config import WaveformStemConfig
from .layers import DSConvBranch


class WaveformStem(nn.Module):
    """Multi-scale waveform feature extractor producing patch tokens."""

    def __init__(self, config: WaveformStemConfig):
        super().__init__()
        self.config = config

        # Initial convolution: downsample raw signal
        self.initial = nn.Sequential(
            nn.Conv1d(
                config.in_channels,
                config.base_channels,
                kernel_size=config.initial_kernel,
                stride=config.initial_stride,
                padding=config.initial_kernel // 2,
                bias=False,
            ),
            nn.GroupNorm(8, config.base_channels),
            nn.GELU(),
        )

        # Multi-scale parallel branches
        self.branches = nn.ModuleList([
            DSConvBranch(
                channels=config.base_channels,
                kernel_size=k,
                se_reduction=config.se_reduction,
                dropout=config.dropout,
            )
            for k in config.kernel_sizes
        ])

        # Merge: concat all branches → 1×1 conv → embed_dim
        total_ch = config.base_channels * len(config.kernel_sizes)
        self.projection = nn.Sequential(
            nn.Conv1d(total_ch, config.embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, config.embed_dim),
            nn.GELU(),
        )

        # Pool to fixed number of patch tokens
        self.pool = nn.AdaptiveAvgPool1d(config.num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T) raw EEG signal, T=3000

        Returns:
            patch_tokens: (B, num_patches, embed_dim) = (B, 6, 96)
        """
        x = self.initial(x)  # (B, 32, ~1500)

        branch_outs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outs, dim=1)  # (B, 96, ~1500)

        x = self.projection(x)  # (B, 96, ~1500)
        x = self.pool(x)        # (B, 96, 6)

        return x.transpose(1, 2)  # (B, 6, 96)
