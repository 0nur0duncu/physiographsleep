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
from .fusion import build_fusion


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

        # Optional λ-interpolation fusion (scGraPhT IEEE TSIPN 2025 Eq. 1)
        self.transformer_classifier, self.fusion = build_fusion(
            config.fusion, config.waveform, config.heads,
        )
        self.fusion_enabled = self.fusion is not None

    def encode_epoch(
        self,
        signal: torch.Tensor,
        spectral_features: torch.Tensor,
        return_patch_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a single epoch to an embedding.

        Args:
            signal: (B, 1, T) raw EEG
            spectral_features: (B, 5, 42)
            return_patch_tokens: also return raw patch tokens for fusion.

        Returns:
            epoch_embedding: (B, 128)
            (optional) patch_tokens: (B, num_patches, embed_dim)
        """
        patch_tokens = self.waveform_stem(signal)          # (B, 6, 96)
        band_tokens = self.spectral_encoder(spectral_features)  # (B, 5, 96)
        epoch_emb = self.graph_encoder(patch_tokens, band_tokens)  # (B, 128)
        if return_patch_tokens:
            return epoch_emb, patch_tokens
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

        # Encode ALL epochs in one batched pass (B*L instead of looping L times)
        flat_sig = signals.reshape(B * L, C, T)              # (B*L, C, T)
        flat_spec = spectral_features.reshape(B * L, *spectral_features.shape[2:])  # (B*L, 5, 42)
        if self.fusion_enabled:
            flat_emb, flat_patch = self.encode_epoch(
                flat_sig, flat_spec, return_patch_tokens=True,
            )
        else:
            flat_emb = self.encode_epoch(flat_sig, flat_spec)
        epoch_embeddings = flat_emb.reshape(B, L, -1)         # (B, L, 128)

        # Decode sequence
        seq_features = self.sequence_decoder(epoch_embeddings, mask)  # (B, L, 160)

        # Extract center epoch features
        center_idx = L // 2
        center_features = seq_features[:, center_idx, :]  # (B, 160)

        # Multi-task predictions
        predictions = self.heads(center_features)

        if self.fusion_enabled:
            # Center-epoch patch tokens for the waveform-only branch
            patch_tokens = flat_patch.reshape(B, L, *flat_patch.shape[1:])
            center_patch = patch_tokens[:, center_idx]  # (B, num_patches, embed_dim)
            trans_logits = self.transformer_classifier(center_patch)  # (B, K)

            gnn_logits = predictions["stage"]
            if self.config.fusion.detach_gnn_for_lambda:
                # scGraPhT _EL_ semantics: keep GNN gradients clean.
                gnn_logits_for_fusion = gnn_logits.detach()
            else:
                gnn_logits_for_fusion = gnn_logits

            fused = self.fusion(trans_logits, gnn_logits_for_fusion)
            predictions["stage_gnn"] = gnn_logits
            predictions["stage_transformer"] = trans_logits
            predictions["stage"] = fused  # main metric uses fused logits
            predictions["lambda"] = self.fusion.lambda_value.detach()

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
        if self.fusion_enabled:
            components["transformer_classifier"] = self.transformer_classifier
            components["fusion"] = self.fusion
        counts = {}
        total = 0
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            counts[name] = n
            total += n
        counts["total"] = total
        return counts
