"""Grad-CAM-style band/patch attention extraction.

Inspired by scGraPhT (Koç et al., IEEE TSIPN 2025) Fig. 4: PCA of
Grad-CAM importance profiles to show which signal facets each branch
exploits. Here we run the same idea for sleep staging:

  • For a given target stage class k, compute the gradient of logit_k
    with respect to *band* and *patch* token activations entering the
    HeteroGraphEncoder.
  • Importance(node) = ReLU( Σ_d (∂logit_k / ∂token_d) · token_d )
  • Aggregate per-class importance over a validation set to produce
    band/patch heatmaps — useful for thesis figures of the form
    "the model relies on δ-band for N3 and θ-band for REM".

Usage
-----
    cam = BandAttentionCAM(model)
    band_imp, patch_imp = cam.run(loader, device, target_class=4, max_batches=20)
    # band_imp: (5,) — one per band, normalised to sum 1
    # patch_imp: (6,) — one per patch
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CAMResult:
    target_class: int
    band_importance: np.ndarray   # (5,)
    patch_importance: np.ndarray  # (6,)
    n_samples: int


class BandAttentionCAM:
    """Lightweight Grad-CAM hook for the WaveformStem and SpectralEncoder
    output token tensors entering the graph encoder.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._patch_tokens: torch.Tensor | None = None
        self._band_tokens: torch.Tensor | None = None
        self._handles: list = []

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------
    def _hook_patch(self, _, __, output):
        # Stem output is (B, num_patches, embed_dim); detach=False so we
        # can backprop through it.
        output.retain_grad()
        self._patch_tokens = output

    def _hook_band(self, _, __, output):
        output.retain_grad()
        self._band_tokens = output

    def _attach(self) -> None:
        if not self._handles:
            self._handles.append(
                self.model.waveform_stem.register_forward_hook(self._hook_patch)
            )
            self._handles.append(
                self.model.spectral_encoder.register_forward_hook(self._hook_band)
            )

    def _detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def run(
        self,
        loader,
        device: torch.device,
        target_class: int,
        max_batches: int | None = None,
        spectral_extractor=None,
    ) -> CAMResult:
        """Aggregate Grad-CAM importance for `target_class` over `loader`."""
        self.model.eval()
        self._attach()

        # We need gradients but no optimizer step
        for p in self.model.parameters():
            p.requires_grad_(True)

        band_acc = np.zeros(0, dtype=np.float64)
        patch_acc = np.zeros(0, dtype=np.float64)
        n = 0

        try:
            for bi, batch in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break

                signals = batch["signal"].to(device)
                if "spectral" in batch:
                    spectral = batch["spectral"].to(device)
                else:
                    if spectral_extractor is None:
                        raise ValueError(
                            "Need either a 'spectral' batch key or a "
                            "spectral_extractor argument."
                        )
                    spectral = spectral_extractor(signals).to(device)
                labels = batch["label"].to(device)

                # Only consider samples of the target class
                mask = labels == target_class
                if not mask.any():
                    continue

                self.model.zero_grad(set_to_none=True)
                preds = self.model(signals, spectral, batch.get("mask"))
                logits = preds["stage"][mask]
                target_logit = logits[:, target_class].sum()
                target_logit.backward()

                # Patch tokens shape (B*L, num_patches, D); only center epoch
                # contributed via the heads, but every epoch gradient flows
                # through the GRU+attention. Use ALL tokens for stability.
                ptoks = self._patch_tokens
                btoks = self._band_tokens
                if ptoks is None or btoks is None:
                    continue

                pgrad = ptoks.grad
                bgrad = btoks.grad
                if pgrad is None or bgrad is None:
                    continue

                # Importance(node) = ReLU( Σ_d grad_d · activation_d )
                p_imp = (pgrad * ptoks).sum(dim=-1).clamp(min=0)  # (B*L, num_patches)
                b_imp = (bgrad * btoks).sum(dim=-1).clamp(min=0)  # (B*L, num_bands)

                p_mean = p_imp.detach().mean(dim=0).cpu().numpy()
                b_mean = b_imp.detach().mean(dim=0).cpu().numpy()

                if patch_acc.size == 0:
                    patch_acc = np.zeros_like(p_mean, dtype=np.float64)
                    band_acc = np.zeros_like(b_mean, dtype=np.float64)
                patch_acc += p_mean
                band_acc += b_mean
                n += int(mask.sum().item())

        finally:
            self._detach()

        # Normalise so each profile sums to 1 (relative importance)
        if patch_acc.sum() > 0:
            patch_acc = patch_acc / patch_acc.sum()
        if band_acc.sum() > 0:
            band_acc = band_acc / band_acc.sum()

        return CAMResult(
            target_class=target_class,
            band_importance=band_acc,
            patch_importance=patch_acc,
            n_samples=n,
        )
