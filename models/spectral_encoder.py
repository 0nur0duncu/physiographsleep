"""SpectralTokenEncoder — per-band MLP producing spectral tokens.

For each of 5 frequency bands, projects aggregated patch-level features
into a 96-dim token via a small MLP.

Input: (B, 5, 42) — 5 bands, 42 features per band (6 patches × 7 features)
Output: (B, 5, 96) — 5 spectral tokens
"""

import torch
import torch.nn as nn

from ..configs.model_config import SpectralEncoderConfig


class BandMLP(nn.Module):
    """Shared MLP architecture for a single band."""

    def __init__(self, in_features: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralTokenEncoder(nn.Module):
    """Encode spectral features into per-band tokens."""

    def __init__(self, config: SpectralEncoderConfig):
        super().__init__()
        self.band_mlps = nn.ModuleList([
            BandMLP(
                in_features=config.features_per_band,
                hidden_dim=config.hidden_dim,
                out_dim=config.embed_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_bands)
        ])

    def forward(self, spectral_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral_features: (B, 5, 42) — pre-extracted spectral features

        Returns:
            band_tokens: (B, 5, 96)
        """
        tokens = []
        for i, mlp in enumerate(self.band_mlps):
            band_feat = spectral_features[:, i, :]  # (B, 42)
            tokens.append(mlp(band_feat))            # (B, 96)

        return torch.stack(tokens, dim=1)  # (B, 5, 96)
