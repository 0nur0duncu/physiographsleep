"""Trainer — 3-stage curriculum training for PhysioGraphSleep.

Stage-specific data strategies based on:
  - Zhang et al. (Frontiers Neuroscience 2023): two-branch trade-off,
    universal features first, rebalancing later.
  - Lee et al. (Frontiers Physiology 2023, SeriesSleepNet): adaptive
    F1-based loss weights, lower lr for sequence decoder.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..configs.train_config import TrainConfig
from ..data.sampler import build_weighted_sampler
from ..data.spectral import SpectralFeatureExtractor
from ..models.losses import MultiTaskLoss
from ..models.physiographsleep import PhysioGraphSleep
from .callbacks import EarlyStopping, ModelCheckpoint
from .evaluator import Evaluator
from .scheduler import build_scheduler
from ..utils.io_utils import load_checkpoint

logger = logging.getLogger("physiographsleep.trainer")


class Trainer:
    """Curriculum trainer with stage-specific data strategies.

    Stage A — Encoder pretrain:  N1 boost 2.0x sampler, standard focal loss
    Stage B — Decoder train:     Natural distribution (no boost), lower lr
    Stage C — End-to-end:        N1 boost 1.5x, adaptive F1-based loss weights
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

    def _build_train_loader(self, n1_boost: float | None = None) -> DataLoader:
        """Build a training DataLoader with optional N1-boosted sampling."""
        if n1_boost is not None and n1_boost > 1.0:
            sampler = build_weighted_sampler(self.train_labels, n1_boost=n1_boost)
            return DataLoader(
                self.train_dataset,
                batch_size=self.data_config.batch_size,
                sampler=sampler,
                num_workers=self.data_config.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def _load_stage_checkpoint(self, stage: str) -> bool:
        """Load best checkpoint for given stage. Returns True if loaded."""
        path = Path(self.config.checkpoint_dir) / f"stage-{stage.lower()}.pt"
        if path.exists():
            ckpt = load_checkpoint(path, self.device)
            self.model.load_state_dict(ckpt["model"])
            self.best_metrics = ckpt.get("metrics", {})
            logger.info(
                f"  Loaded {path.name} "
                f"(MF1={self.best_metrics.get('macro_f1', 0):.4f})"
            )
            return True
        return False

    def train(self, start_stage: str = "A") -> dict[str, float]:
        """Run curriculum training from a given stage."""
        stages = ["A", "B", "C"]
        start_idx = stages.index(start_stage.upper())

        if start_idx <= 0:
            logger.info("=== Stage A: Epoch encoder pretraining ===")
            self._run_stage_a()

        if start_idx <= 1:
            self._load_stage_checkpoint("A")
            logger.info("=== Stage B: Sequence decoder training ===")
            self._run_stage_b()

        if start_idx <= 2:
            self._load_stage_checkpoint("B")
            logger.info("=== Stage C: End-to-end fine-tuning ===")
            self._run_stage_c()

        self._load_stage_checkpoint("C")
        return self.best_metrics

    # ------------------------------------------------------------------
    # Stage A: Epoch encoder pretraining (N1 boost 2.0x)
    # ------------------------------------------------------------------
    def _run_stage_a(self) -> None:
        self._freeze_module(self.model.sequence_decoder)
        self._freeze_module(self.model.heads.boundary_head)
        self._freeze_module(self.model.heads.prev_head)
        self._freeze_module(self.model.heads.next_head)

        loader = self._build_train_loader(n1_boost=2.0)
        optimizer = self._build_optimizer(self.config.curriculum.stage_a_lr)
        scheduler = build_scheduler(optimizer, self.config.scheduler)
        stopper = EarlyStopping(patience=self.config.patience)
        checkpoint = ModelCheckpoint(self.config.checkpoint_dir, mode="max")

        for epoch in range(self.config.curriculum.stage_a_epochs):
            train_loss = self._train_one_epoch(optimizer, loader, stage="A")
            val_metrics = self.evaluator.evaluate(
                self.model, self.val_loader, self.spectral_extractor,
            )

            self._log_epoch("A", epoch, train_loss, val_metrics)
            scheduler.step()

            self._save_if_best(val_metrics, checkpoint, "A")
            if stopper.step(val_metrics["macro_f1"]):
                logger.info(f"Stage A early stop at epoch {epoch}")
                break

        self._unfreeze_all()

    # ------------------------------------------------------------------
    # Stage B: Sequence decoder — natural distribution, low lr
    # ------------------------------------------------------------------
    def _run_stage_b(self) -> None:
        self._freeze_module(self.model.waveform_stem)
        self._freeze_module(self.model.spectral_encoder)
        self._freeze_module(self.model.graph_encoder)

        loader = self._build_train_loader(n1_boost=None)
        optimizer = self._build_optimizer(self.config.curriculum.stage_b_lr)
        scheduler = build_scheduler(optimizer, self.config.scheduler)
        stopper = EarlyStopping(patience=self.config.patience)
        checkpoint = ModelCheckpoint(self.config.checkpoint_dir, mode="max")

        for epoch in range(self.config.curriculum.stage_b_epochs):
            train_loss = self._train_one_epoch(optimizer, loader, stage="B")
            val_metrics = self.evaluator.evaluate(
                self.model, self.val_loader, self.spectral_extractor,
            )

            self._log_epoch("B", epoch, train_loss, val_metrics)
            scheduler.step()

            self._save_if_best(val_metrics, checkpoint, "B")
            if stopper.step(val_metrics["macro_f1"]):
                logger.info(f"Stage B early stop at epoch {epoch}")
                break

        self._unfreeze_all()

    # ------------------------------------------------------------------
    # Stage C: End-to-end — mild N1 boost, adaptive F1-based loss
    # ------------------------------------------------------------------
    def _run_stage_c(self) -> None:
        loader = self._build_train_loader(n1_boost=1.5)
        optimizer = self._build_optimizer(self.config.curriculum.stage_c_lr)
        scheduler = build_scheduler(optimizer, self.config.scheduler)
        stopper = EarlyStopping(patience=self.config.patience)
        checkpoint = ModelCheckpoint(self.config.checkpoint_dir, mode="max")
        warmup = self.config.adaptive_loss.warmup_epochs

        for epoch in range(self.config.curriculum.stage_c_epochs):
            use_adaptive = epoch >= warmup
            result = self._train_one_epoch(
                optimizer, loader, stage="C", collect_preds=use_adaptive,
            )

            if isinstance(result, tuple):
                train_loss, train_preds, train_targets = result
                class_f1 = self._compute_per_class_f1(train_targets, train_preds)
                self.loss_fn.update_adaptive_weights(
                    class_f1,
                    K=self.config.adaptive_loss.K,
                    gamma=self.config.adaptive_loss.gamma,
                )
                logger.info(
                    f"  Adaptive F1: {[f'{f:.3f}' for f in class_f1]} | "
                    f"Weights: {[f'{w:.2f}' for w in self.loss_fn.focal.weight.tolist()]}"
                )
            else:
                train_loss = result

            val_metrics = self.evaluator.evaluate(
                self.model, self.val_loader, self.spectral_extractor,
            )

            self._log_epoch("C", epoch, train_loss, val_metrics)
            scheduler.step()

            self._save_if_best(val_metrics, checkpoint, "C")
            if stopper.step(val_metrics["macro_f1"]):
                logger.info(f"Stage C early stop at epoch {epoch}")
                break

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------
    def _train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        stage: str = "",
        collect_preds: bool = False,
    ) -> float | tuple[float, np.ndarray, np.ndarray]:
        """Train for one epoch.

        Returns mean loss, or (loss, preds, labels) when collect_preds=True.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_preds: list[np.ndarray] | None = [] if collect_preds else None
        all_labels: list[np.ndarray] | None = [] if collect_preds else None

        pbar = tqdm(loader, desc=f"Train {stage}", leave=False)
        for batch in pbar:
            signals = batch["signal"].to(self.device)

            if "spectral" in batch:
                spectral = batch["spectral"].to(self.device)
            else:
                spectral = self.evaluator._extract_spectral_batch(
                    signals, self.spectral_extractor,
                )

            targets = {
                "label": batch["label"].to(self.device),
                "boundary": batch["boundary"].to(self.device),
                "prev_label": batch["prev_label"].to(self.device),
                "next_label": batch["next_label"].to(self.device),
                "n1_label": batch["n1_label"].to(self.device),
            }

            mask = batch["mask"].to(self.device) if "mask" in batch else None
            predictions = self.model(signals, spectral, mask)
            losses = self.loss_fn(predictions, targets)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.optimizer.grad_clip,
            )
            optimizer.step()

            total_loss += losses["total"].item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

            if collect_preds:
                with torch.no_grad():
                    all_preds.append(predictions["stage"].argmax(dim=1).cpu().numpy())
                    all_labels.append(targets["label"].cpu().numpy())

        mean_loss = total_loss / max(num_batches, 1)
        if collect_preds:
            return mean_loss, np.concatenate(all_preds), np.concatenate(all_labels)
        return mean_loss

    @staticmethod
    def _compute_per_class_f1(
        labels: np.ndarray, preds: np.ndarray, num_classes: int = 5,
    ) -> np.ndarray:
        """Compute per-class F1 scores from flat arrays."""
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
    # Helpers
    # ------------------------------------------------------------------
    def _build_optimizer(self, lr: float) -> AdamW:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(
            trainable,
            lr=lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=self.config.optimizer.betas,
        )

    def _save_if_best(
        self, metrics: dict[str, float], checkpoint: ModelCheckpoint, stage: str,
    ) -> None:
        state = {
            "model": self.model.state_dict(),
            "metrics": metrics,
        }
        filename = f"stage-{stage.lower()}.pt"
        saved = checkpoint.step(metrics["macro_f1"], state, filename=filename)
        if saved:
            self.best_metrics = metrics
            logger.info(f"  New best MF1: {metrics['macro_f1']:.4f} -> {filename}")

    def _log_epoch(
        self, stage: str, epoch: int, loss: float, metrics: dict[str, float],
    ) -> None:
        logger.info(
            f"[{stage}] Epoch {epoch:02d} | "
            f"Loss={loss:.4f} | "
            f"ACC={metrics['accuracy']:.4f} | "
            f"MF1={metrics['macro_f1']:.4f} | "
            f"κ={metrics['kappa']:.4f}"
        )

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    def _unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
