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
        return_logits: bool = False,
    ) -> dict[str, float] | tuple[dict[str, float], np.ndarray, np.ndarray]:
        """Run evaluation on a dataloader.

        Args:
            model: trained model in eval mode
            dataloader: validation or test DataLoader
            spectral_extractor: SpectralFeatureExtractor instance
            return_logits: if True, also return (logits, labels) arrays

        Returns:
            metrics dict, or (metrics, logits, labels) if return_logits=True
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_logits = []
        all_preds_gnn: list[np.ndarray] = []   # GPU argmax → CPU numpy per batch
        all_logits_gnn: list[np.ndarray] = []  # only filled when return_logits
        amp_enabled = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if (
            amp_enabled and torch.cuda.is_bf16_supported()
        ) else torch.float16

        for batch in dataloader:
            signals = batch["signal"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True) if "mask" in batch else None

            if "spectral" in batch:
                spectral = batch["spectral"].to(self.device, non_blocking=True)
            else:
                B, L, C, T = signals.shape
                spectral = self._extract_spectral_batch(signals, spectral_extractor)

            with torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(signals, spectral, mask)
                logits = outputs["stage"]
                logits_gnn = outputs.get("stage_gnn")
            # argmax runs on GPU; only the (B,) int tensor crosses PCIe.
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if logits_gnn is not None:
                all_preds_gnn.append(logits_gnn.argmax(dim=-1).cpu().numpy())
            if return_logits:
                all_logits.append(logits.float().cpu().numpy())
                if logits_gnn is not None:
                    all_logits_gnn.append(logits_gnn.float().cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = self.metrics.compute_all(all_labels, all_preds)
        # When λ-fusion is active, surface a pure-GNN MF1 so the caller
        # can compare against the fused metric (diagnostic only; the
        # primary metric remains the fused one).
        if all_preds_gnn:
            gnn_preds = np.concatenate(all_preds_gnn)
            gnn_metrics = self.metrics.compute_all(all_labels, gnn_preds)
            metrics["macro_f1_gnn"] = gnn_metrics["macro_f1"]
            metrics["accuracy_gnn"] = gnn_metrics["accuracy"]

        if return_logits:
            all_logits = np.concatenate(all_logits)
            return metrics, all_logits, all_labels

        return metrics

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
