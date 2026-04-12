"""Evaluator — compute metrics on a dataset split."""

import torch
import numpy as np
from torch.utils.data import DataLoader

from ..evaluation.metrics import MetricsCalculator


class Evaluator:
    """Run model evaluation and compute metrics."""

    def __init__(self, device: torch.device):
        self.device = device
        self.metrics = MetricsCalculator()

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        spectral_extractor=None,
    ) -> dict[str, float]:
        """Run evaluation on a dataloader.

        Args:
            model: trained model in eval mode
            dataloader: validation or test DataLoader
            spectral_extractor: SpectralFeatureExtractor instance

        Returns:
            metrics dict with accuracy, macro_f1, kappa, per-class F1
        """
        model.eval()
        all_preds = []
        all_labels = []

        for batch in dataloader:
            signals = batch["signal"].to(self.device)       # (B, L, C, T)
            labels = batch["label"].to(self.device)          # (B,)

            # Use pre-computed spectral features if available
            if "spectral" in batch:
                spectral = batch["spectral"].to(self.device)
            else:
                B, L, C, T = signals.shape
                spectral = self._extract_spectral_batch(signals, spectral_extractor)

            outputs = model(signals, spectral)
            preds = outputs["stage"].argmax(dim=-1)  # (B,)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        return self.metrics.compute_all(all_labels, all_preds)

    @staticmethod
    def _extract_spectral_batch(
        signals: torch.Tensor,
        extractor,
    ) -> torch.Tensor:
        """Extract spectral features for a batch of sequences.

        Args:
            signals: (B, L, C, T)
            extractor: SpectralFeatureExtractor or None

        Returns:
            spectral: (B, L, 5, 42)
        """
        B, L, C, T = signals.shape
        device = signals.device

        if extractor is None:
            return torch.zeros(B, L, 5, 42, device=device)

        sig_np = signals[:, :, 0, :].cpu().numpy()  # (B, L, T)
        specs = []
        for b in range(B):
            seq_specs = []
            for t in range(L):
                feat = extractor.extract_epoch(sig_np[b, t])  # (5, 42)
                seq_specs.append(feat)
            specs.append(np.stack(seq_specs))
        specs = np.stack(specs)  # (B, L, 5, 42)
        return torch.from_numpy(specs).float().to(device)
