"""PhysioGraphSleep — full model assembly.

Pipeline per epoch:
  raw_signal (1, 3000) → WaveformStem → patch_tokens (6, 96)
  spectral_features (5, 42) → SpectralTokenEncoder → band_tokens (5, 96)
  (patch_tokens, band_tokens) → HeteroGraphEncoder → epoch_embedding (128)

Pipeline per sequence:
  epoch_embeddings (L, 128) → SequenceTransitionDecoder → seq_features (L, 160)
  center_features (160) → MultiTaskHeads → predictions
"""

import torch
import torch.nn as nn

from ..configs.model_config import ModelConfig
from .waveform_stem import WaveformStem
from .spectral_encoder import SpectralTokenEncoder
from .hetero_graph import HeteroGraphEncoder
from .sequence_decoder import SequenceTransitionDecoder
from .heads import MultiTaskHeads


class PhysioGraphSleep(nn.Module):
    """Full PhysioGraphSleep model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.waveform_stem = WaveformStem(config.waveform)
        self.spectral_encoder = SpectralTokenEncoder(config.spectral)
        self.graph_encoder = HeteroGraphEncoder(config.graph)
        self.sequence_decoder = SequenceTransitionDecoder(config.decoder)
        self.heads = MultiTaskHeads(config.heads)

    def encode_epoch(
        self,
        signal: torch.Tensor,
        spectral_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a single epoch to an embedding.

        Args:
            signal: (B, 1, T) raw EEG
            spectral_features: (B, 5, 42)

        Returns:
            epoch_embedding: (B, 128)
        """
        patch_tokens = self.waveform_stem(signal)          # (B, 6, 96)
        band_tokens = self.spectral_encoder(spectral_features)  # (B, 5, 96)
        epoch_emb = self.graph_encoder(patch_tokens, band_tokens)  # (B, 128)
        return epoch_emb

    def forward(
        self,
        signals: torch.Tensor,
        spectral_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass over a sequence of epochs.

        Args:
            signals: (B, L, C, T) — sequence of epochs
            spectral_features: (B, L, 5, 42)
            mask: (B, L) — sequence validity mask

        Returns:
            predictions: dict from MultiTaskHeads for center epoch
        """
        B, L, C, T = signals.shape

        # Encode each epoch independently
        epoch_embeddings = []
        for t in range(L):
            sig_t = signals[:, t, :, :]          # (B, C, T)
            spec_t = spectral_features[:, t, :, :]  # (B, 5, 42)
            emb_t = self.encode_epoch(sig_t, spec_t)  # (B, 128)
            epoch_embeddings.append(emb_t)

        epoch_embeddings = torch.stack(epoch_embeddings, dim=1)  # (B, L, 128)

        # Decode sequence
        seq_features = self.sequence_decoder(epoch_embeddings, mask)  # (B, L, 160)

        # Extract center epoch features
        center_idx = L // 2
        center_features = seq_features[:, center_idx, :]  # (B, 160)

        # Multi-task predictions
        predictions = self.heads(center_features)
        return predictions

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per component."""
        components = {
            "waveform_stem": self.waveform_stem,
            "spectral_encoder": self.spectral_encoder,
            "graph_encoder": self.graph_encoder,
            "sequence_decoder": self.sequence_decoder,
            "heads": self.heads,
        }
        counts = {}
        total = 0
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            counts[name] = n
            total += n
        counts["total"] = total
        return counts
