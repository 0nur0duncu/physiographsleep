"""Self-supervised pretraining for PhysioGraphSleep.

Closes the gap to L-SeqSleepNet (MF1 0.829) which pretrains on unlabeled
epochs (`fair_comparison_analysis.md`). Strategy: *masked epoch
reconstruction* with the WaveformStem + HeteroGraphEncoder as encoder.

Pipeline:
  1. Take a sequence of (B, L, C, T) raw epochs (labels not used).
  2. Randomly mask ~15% of patch tokens AFTER WaveformStem (BERT-style).
  3. HeteroGraphEncoder + sequence_decoder process the masked tokens.
  4. A small reconstruction head re-predicts the masked patch token
     activations (MSE loss).
  5. Optionally combine with a contrastive objective on epoch
     embeddings (positive = same subject adjacent epochs, negative =
     random different-subject epochs).

This module ships a *reusable* pretraining harness; the trainer code
remains identical to the supervised path so the encoder weights produced
here can be loaded straight into `Trainer` via standard PyTorch
`load_state_dict(strict=False)`.

Usage:
    python -m physiographsleep.scripts.run_pretrain \
        --epochs 20 --batch-size 64 --output checkpoints/pretrain.pt
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PretrainConfig:
    mask_ratio: float = 0.15
    contrastive_weight: float = 0.0   # set >0 to enable contrastive head
    contrastive_temperature: float = 0.07


class PretrainingHead(nn.Module):
    """Reconstructs masked patch tokens from contextualised epoch
    representations.

    Input  : context_embedding (B, L, D_ctx) — from sequence decoder
    Output : (B, L, num_patches, embed_dim) — re-predicted patch tokens
    """

    def __init__(self, ctx_dim: int, num_patches: int, embed_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        hidden = max(ctx_dim, num_patches * embed_dim // 2)
        self.proj = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_patches * embed_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        B, L, _ = context.shape
        out = self.proj(context)
        return out.view(B, L, self.num_patches, self.embed_dim)


class MaskedPatchPretrainer(nn.Module):
    """Wraps PhysioGraphSleep encoder for masked-patch pretraining.

    Reuses the supervised model's WaveformStem, SpectralEncoder,
    HeteroGraphEncoder, and SequenceTransitionDecoder. Adds a tiny
    reconstruction head and (optionally) a projection head for SimCLR-
    style contrastive learning between adjacent epochs.
    """

    def __init__(
        self,
        encoder,                         # PhysioGraphSleep instance
        cfg: PretrainConfig,
    ):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg

        wave = encoder.config.waveform
        decoder_dim = encoder.config.decoder.output_dim
        self.recon_head = PretrainingHead(
            ctx_dim=decoder_dim,
            num_patches=wave.num_patches,
            embed_dim=wave.embed_dim,
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, wave.embed_dim) * 0.02)

        if cfg.contrastive_weight > 0:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim),
                nn.GELU(),
                nn.Linear(decoder_dim, 64),
            )
        else:
            self.contrastive_proj = None

    # ------------------------------------------------------------------
    def _encode_with_masking(
        self, signals: torch.Tensor, spectral: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the encoder with random patch-token masking.

        Returns:
            seq_features: (B, L, D_ctx) — sequence-decoder output
            original_patches: (B, L, P, E) — clean patch tokens (target)
            mask: (B, L, P) bool — True where token was masked
        """
        B, L, C, T = signals.shape
        flat_sig = signals.reshape(B * L, C, T)
        flat_spec = spectral.reshape(B * L, *spectral.shape[2:])

        # WaveformStem + SpectralEncoder
        patch = self.encoder.waveform_stem(flat_sig)               # (B*L, P, E)
        band = self.encoder.spectral_encoder(flat_spec)            # (B*L, 5, E)

        original_patches = patch.detach().clone()

        # Random mask
        BL, P, E = patch.shape
        mask = torch.rand(BL, P, device=patch.device) < self.cfg.mask_ratio
        if mask.any():
            patch = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.expand(BL, P, E),
                patch,
            )

        # Graph + sequence
        epoch_emb = self.encoder.graph_encoder(patch, band)        # (B*L, 128)
        epoch_emb = epoch_emb.view(B, L, -1)
        seq_features = self.encoder.sequence_decoder(epoch_emb)     # (B, L, D_ctx)

        return (
            seq_features,
            original_patches.view(B, L, P, E),
            mask.view(B, L, P),
        )

    # ------------------------------------------------------------------
    def forward(
        self, signals: torch.Tensor, spectral: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        seq_features, target_patches, mask = self._encode_with_masking(signals, spectral)

        recon = self.recon_head(seq_features)                       # (B, L, P, E)

        if mask.any():
            mask_f = mask.unsqueeze(-1).expand_as(recon).float()
            recon_loss = (
                ((recon - target_patches) ** 2) * mask_f
            ).sum() / (mask_f.sum() + 1e-8)
        else:
            recon_loss = (recon - target_patches).pow(2).mean() * 0.0

        out = {"recon_loss": recon_loss, "total": recon_loss}

        if self.contrastive_proj is not None:
            # Anchor: center epoch; positive: any neighbour with mask=1
            B, L, _ = seq_features.shape
            center = seq_features[:, L // 2]
            positive = seq_features[:, L // 2 - 1] if L >= 3 else center

            z_a = F.normalize(self.contrastive_proj(center), dim=-1)
            z_p = F.normalize(self.contrastive_proj(positive), dim=-1)
            logits = z_a @ z_p.t() / self.cfg.contrastive_temperature
            labels = torch.arange(B, device=z_a.device)
            contrastive = F.cross_entropy(logits, labels)
            out["contrastive_loss"] = contrastive
            out["total"] = recon_loss + self.cfg.contrastive_weight * contrastive

        return out
