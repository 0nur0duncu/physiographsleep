"""Trainer — single-phase joint training for PhysioGraphSleep.

Earlier curriculum experiments (April 2026) showed that:
  - Decoder-only fine-tune with frozen encoder added zero F1_N1 over
    16 epochs (overfitting in the decoder, no signal from encoder).
  - End-to-end fine-tune with adaptive loss reweighting regressed
    val MF1 from 0.7599 to 0.7458.

So we keep one phase: joint encoder + decoder + heads with N1-boosted
weighted sampling. This matches TinySleepNet / AttnSleep / SleepTransformer
which are all single-phase end-to-end trained.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..configs.train_config import TrainConfig
from ..data.sampler import build_weighted_sampler
from ..data.spectral import SpectralFeatureExtractor
from ..data.n1_mixup import apply_n1_mixup
from ..models.losses import MultiTaskLoss
from ..models.physiographsleep import PhysioGraphSleep
from .callbacks import EarlyStopping, ModelCheckpoint
from .ema import ModelEMA
from .evaluator import Evaluator
from .scheduler import build_scheduler
from ..utils.io_utils import load_checkpoint

logger = logging.getLogger("physiographsleep.trainer")
logger.propagate = False  # avoid duplicate logs in jupyter/colab


class Trainer:
    """Single-phase joint trainer.

    - Trainable: waveform_stem + spectral_encoder + graph_encoder +
      sequence_decoder + main stage head + N1 auxiliary head.
    - Frozen: boundary / prev_stage / next_stage auxiliary heads (their
      losses still backprop into the encoder via the auxiliary losses if
      enabled, but the heads themselves do not need gradient updates
      beyond what the multi-task loss provides).
    - Sampling: WeightedRandomSampler with N1 boost (default 2.0x).
    """

    def __init__(
        self,
        model: PhysioGraphSleep,
        loss_fn: MultiTaskLoss,
        train_dataset: Dataset,
        train_labels: np.ndarray,
        val_loader: DataLoader,
        config: TrainConfig,
        data_config: Any,
        device: torch.device,
        spectral_extractor: SpectralFeatureExtractor | None = None,
        callback=None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.val_loader = val_loader
        self.config = config
        self.data_config = data_config
        self.device = device
        self.spectral_extractor = spectral_extractor
        self.evaluator = Evaluator(device)
        self.best_metrics: dict[str, float] = {}
        self.scaler = GradScaler("cuda", enabled=device.type == "cuda")
        self.amp_enabled = device.type == "cuda"
        self.callback = callback
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.amp_dtype = torch.bfloat16 if (
            device.type == "cuda" and torch.cuda.is_bf16_supported()
        ) else torch.float16
        self.use_scaler = self.amp_enabled and self.amp_dtype == torch.float16

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self) -> dict[str, float]:
        """Run joint single-phase training. Returns best val metrics."""
        # NOTE: All heads trainable. Freezing random-init aux heads while
        # still including their CE/BCE in the total loss pushes the encoder
        # to match random linear readouts — pure gradient noise that
        # inflates val_loss and steals capacity from the main stage head.
        loader = self._build_train_loader(n1_boost=self.config.n1_boost)
        optimizer = self._build_optimizer(self.config.lr)
        scheduler = build_scheduler(optimizer, self.config.scheduler)
        stopper = EarlyStopping(patience=self.config.patience)
        checkpoint = ModelCheckpoint(self.config.checkpoint_dir, mode="max")

        # EMA: track decayed copy of live weights. Evaluated instead of
        # the raw model for stable val metrics + small accuracy gain.
        ema = ModelEMA(self.model, decay=self.config.ema_decay)

        for epoch in range(self.config.epochs):
            train_loss, tr_preds, tr_labels = self._train_one_epoch(
                optimizer, loader, ema=ema,
            )
            train_metrics = self._train_metrics(tr_labels, tr_preds)
            # Evaluate with EMA weights swapped into the live model.
            with ema.swap_into(self.model):
                val_loss, val_metrics = self._evaluate_with_loss()

            self._log_epoch(epoch, train_loss, val_loss, val_metrics, train_metrics)
            scheduler.step()

            # Save EMA state (not raw) as the best checkpoint — that is the
            # inference model for downstream evaluation / post-processing.
            self._save_if_best(val_metrics, checkpoint, state_dict=ema.ema.state_dict())
            if stopper.step(val_metrics["macro_f1"]):
                logger.info(f"Early stop at epoch {epoch}")
                break

        # Reload best weights so caller / post-processing uses the best model.
        self._load_best()
        return self.best_metrics

    def _load_best(self) -> bool:
        path = Path(self.config.checkpoint_dir) / self.config.checkpoint_name
        if path.exists():
            ckpt = load_checkpoint(path, self.device)
            self.model.load_state_dict(ckpt["model"])
            self.best_metrics = ckpt.get("metrics", {})
            logger.info(
                f"Loaded best checkpoint {path.name} "
                f"(MF1={self.best_metrics.get('macro_f1', 0):.4f})"
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Loaders / optimizer / metrics helpers
    # ------------------------------------------------------------------
    def _build_train_loader(self, n1_boost: float | None = None) -> DataLoader:
        common = dict(
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.data_config.num_workers > 0,
            prefetch_factor=4 if self.data_config.num_workers > 0 else None,
        )
        if n1_boost is not None and n1_boost > 1.0:
            sampler = build_weighted_sampler(self.train_labels, n1_boost=n1_boost)
            return DataLoader(self.train_dataset, sampler=sampler, **common)
        return DataLoader(self.train_dataset, shuffle=True, **common)

    def _build_optimizer(self, lr: float) -> AdamW:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(
            trainable,
            lr=lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=self.config.optimizer.betas,
        )

    def _save_if_best(
        self,
        metrics: dict[str, float],
        checkpoint: ModelCheckpoint,
        state_dict: dict | None = None,
    ) -> None:
        state = {
            "model": state_dict if state_dict is not None else self.model.state_dict(),
            "metrics": metrics,
        }
        saved = checkpoint.step(
            metrics["macro_f1"], state, filename=self.config.checkpoint_name,
        )
        if saved:
            self.best_metrics = metrics
            logger.info(
                f"  New best MF1: {metrics['macro_f1']:.4f} -> "
                f"{self.config.checkpoint_name}"
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        ema: ModelEMA | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc="Train", leave=False)
        gpu_preds: list[torch.Tensor] = []
        gpu_labels: list[torch.Tensor] = []
        n1_mixup_cfg = self.config.n1_mixup
        for batch in pbar:
            signals = batch["signal"].to(self.device, non_blocking=True)
            spectral = (
                batch["spectral"].to(self.device, non_blocking=True)
                if "spectral" in batch
                else self.evaluator._extract_spectral_batch(
                    signals, self.spectral_extractor,
                )
            )
            targets = {
                "label":      batch["label"].to(self.device, non_blocking=True),
                "boundary":   batch["boundary"].to(self.device, non_blocking=True),
                "prev_label": batch["prev_label"].to(self.device, non_blocking=True),
                "next_label": batch["next_label"].to(self.device, non_blocking=True),
                "n1_label":   batch["n1_label"].to(self.device, non_blocking=True),
            }
            mask = batch["mask"].to(self.device, non_blocking=True) if "mask" in batch else None

            # N1-targeted Mixup (no-op if config is None or no N1 in batch)
            if n1_mixup_cfg is not None:
                mix_batch = {"signal": signals, "spectral": spectral, "label": targets["label"]}
                mix_batch, mix_info = apply_n1_mixup(mix_batch, n1_mixup_cfg)
                if mix_info is not None:
                    signals = mix_batch["signal"]
                    spectral = mix_batch["spectral"]
                    targets["label_soft"] = mix_info["soft_label"]

            with autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                predictions = self.model(signals, spectral, mask)
                losses = self.loss_fn(predictions, targets)

            optimizer.zero_grad(set_to_none=True)
            if self.use_scaler:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.grad_clip,
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                if ema is not None:
                    ema.update(self.model)
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.grad_clip,
                )
                optimizer.step()
                if ema is not None:
                    ema.update(self.model)

            total_loss += losses["total"].item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

            with torch.no_grad():
                gpu_preds.append(predictions["stage"].argmax(dim=1).detach())
                gpu_labels.append(targets["label"].detach())

        mean_loss = total_loss / max(num_batches, 1)
        all_preds = torch.cat(gpu_preds).cpu().numpy()
        all_labels = torch.cat(gpu_labels).cpu().numpy()
        return mean_loss, all_preds, all_labels

    def _train_metrics(
        self, labels: np.ndarray, preds: np.ndarray,
    ) -> dict[str, float]:
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        acc = float((labels == preds).mean()) if labels.size else 0.0
        f1 = self._compute_per_class_f1(labels, preds)
        return {
            "accuracy": acc,
            "macro_f1": float(f1.mean()),
            "per_class_f1": f1.tolist(),
        }

    @staticmethod
    def _compute_per_class_f1(
        labels: np.ndarray, preds: np.ndarray, num_classes: int = 5,
    ) -> np.ndarray:
        f1 = np.zeros(num_classes)
        for c in range(num_classes):
            tp = np.sum((preds == c) & (labels == c))
            fp = np.sum((preds == c) & (labels != c))
            fn = np.sum((preds != c) & (labels == c))
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1[c] = 2 * precision * recall / (precision + recall + 1e-8)
        return f1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_with_loss(self) -> tuple[float, dict[str, float]]:
        """Single val-pass: mean loss + full metric dictionary."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        gpu_preds: list[torch.Tensor] = []
        gpu_labels: list[torch.Tensor] = []

        for batch in self.val_loader:
            signals = batch["signal"].to(self.device, non_blocking=True)
            spectral = (
                batch["spectral"].to(self.device, non_blocking=True)
                if "spectral" in batch
                else self.evaluator._extract_spectral_batch(
                    signals, self.spectral_extractor,
                )
            )
            targets = {
                "label":      batch["label"].to(self.device, non_blocking=True),
                "boundary":   batch["boundary"].to(self.device, non_blocking=True),
                "prev_label": batch["prev_label"].to(self.device, non_blocking=True),
                "next_label": batch["next_label"].to(self.device, non_blocking=True),
                "n1_label":   batch["n1_label"].to(self.device, non_blocking=True),
            }
            mask = batch["mask"].to(self.device, non_blocking=True) if "mask" in batch else None

            with autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                preds = self.model(signals, spectral, mask)
                losses = self.loss_fn(preds, targets)

            total_loss += losses["total"].item()
            n_batches += 1
            gpu_preds.append(preds["stage"].argmax(dim=1).detach())
            gpu_labels.append(targets["label"].detach())

        mean_loss = total_loss / max(n_batches, 1)
        all_preds = torch.cat(gpu_preds).cpu().numpy()
        all_labels = torch.cat(gpu_labels).cpu().numpy()
        metrics = self.evaluator.metrics.compute_all(all_labels, all_preds)
        return mean_loss, metrics

    # ------------------------------------------------------------------
    # Logging + freeze helpers
    # ------------------------------------------------------------------
    def _log_epoch(
        self, epoch: int, loss: float, val_loss: float,
        metrics: dict[str, float],
        train_metrics: dict[str, float] | None = None,
    ) -> None:
        train_extra = ""
        if train_metrics is not None:
            train_extra = (
                f"TrainACC={train_metrics.get('accuracy', 0):.4f} | "
                f"TrainMF1={train_metrics.get('macro_f1', 0):.4f} | "
            )
        logger.info(
            f"Epoch {epoch:02d} | "
            f"TrLoss={loss:.4f} | VlLoss={val_loss:.4f} | "
            f"{train_extra}"
            f"ValACC={metrics['accuracy']:.4f} | "
            f"ValMF1={metrics['macro_f1']:.4f} | "
            f"\u03ba={metrics['kappa']:.4f}"
        )
        if self.callback is not None:
            try:
                self.callback(
                    epoch=epoch,
                    train_loss=loss, val_loss=val_loss,
                    train_metrics=train_metrics or {}, val_metrics=metrics,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Callback failed: {exc}")

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False
