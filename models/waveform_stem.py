"""WaveformStem — multi-scale CNN for extracting patch morphology tokens.

Architecture (1-channel):
  Input(1, 3000) → Conv1d(1→32, k=25, s=2) → GroupNorm → GELU
  → 3 parallel DSConv branches (k=25/75/225)
  → Concat → 1×1 projection → 96ch
  → AdaptiveAvgPool → 6 patch tokens
  Output: (B, 6, 96)

Architecture (C-channel, C≥2):
  Input(C, 3000) → depthwise Conv1d(groups=C) so each channel has its
  own `base_channels` filters (e.g. C=2 → 64 total) → 1×1 Conv1d to
  merge back to `base_channels`. This avoids the "forced compromise"
  failure mode where a single Conv1d with in=C learns blended filters
  that work poorly on either modality (empirically: EEG+EOG stem MF1
  collapsed to 0.06 vs 0.52 single-channel in April 2026 diagnostic).
  Rest of the pipeline is identical to the 1-channel case.
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

        in_ch = config.in_channels
        base = config.base_channels
        if in_ch == 1:
            # Classic single-channel path (preserves pre-2ch checkpoints).
            self.initial = nn.Sequential(
                nn.Conv1d(
                    in_ch, base,
                    kernel_size=config.initial_kernel,
                    stride=config.initial_stride,
                    padding=config.initial_kernel // 2,
                    bias=False,
                ),
                nn.GroupNorm(8, base),
                nn.GELU(),
            )
        else:
            # Multi-channel path: per-channel filter bank (groups=in_ch)
            # followed by a point-wise mix. Each channel gets `base`
            # dedicated filters → total in_ch*base intermediate maps →
            # 1×1 conv merges them back to `base` so downstream layers
            # see the same shape as the 1ch path.
            self.initial = nn.Sequential(
                nn.Conv1d(
                    in_ch, base * in_ch,
                    kernel_size=config.initial_kernel,
                    stride=config.initial_stride,
                    padding=config.initial_kernel // 2,
                    groups=in_ch,
                    bias=False,
                ),
                nn.GroupNorm(8, base * in_ch),
                nn.GELU(),
                nn.Conv1d(base * in_ch, base, kernel_size=1, bias=False),
                nn.GroupNorm(8, base),
                nn.GELU(),
            )

        # Multi-scale parallel branches (identical for 1ch / 2ch since
        # channel axis has been collapsed to `base` by the initial block).
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
            x: (B, C, T) raw signal, T=3000; C matches config.in_channels.

        Returns:
            patch_tokens: (B, num_patches, embed_dim) = (B, 6, 96)
        """
        x = self.initial(x)  # (B, base, ~1500)

        branch_outs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outs, dim=1)  # (B, base*len(branches), ~1500)

        x = self.projection(x)  # (B, embed_dim, ~1500)
        x = self.pool(x)        # (B, embed_dim, num_patches)

        return x.transpose(1, 2)  # (B, num_patches, embed_dim)
