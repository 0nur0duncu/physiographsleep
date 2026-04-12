"""Trainer — 3-stage curriculum training for PhysioGraphSleep."""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configs.train_config import TrainConfig
from ..data.spectral import SpectralFeatureExtractor
from ..models.losses import MultiTaskLoss
from ..models.physiographsleep import PhysioGraphSleep
from .callbacks import EarlyStopping, ModelCheckpoint
from .evaluator import Evaluator
from .scheduler import build_scheduler

logger = logging.getLogger("physiographsleep.trainer")


class Trainer:
    """Curriculum trainer with 3 training stages."""

    def __init__(
        self,
        model: PhysioGraphSleep,
        loss_fn: MultiTaskLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
        spectral_extractor: SpectralFeatureExtractor | None = None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.spectral_extractor = spectral_extractor
        self.evaluator = Evaluator(device)

        self.checkpoint = ModelCheckpoint(config.checkpoint_dir, mode="max")
        self.best_metrics: dict[str, float] = {}

    def train(self) -> dict[str, float]:
        """Run full 3-stage curriculum training."""
        logger.info("=== Stage A: Epoch encoder pretraining ===")
        self._run_stage_a()

        logger.info("=== Stage B: Sequence decoder training ===")
        self._run_stage_b()

        logger.info("=== Stage C: End-to-end fine-tuning ===")
        self._run_stage_c()

        return self.best_metrics

    # ------------------------------------------------------------------
    # Stage A: Epoch encoder pretraining
    # ------------------------------------------------------------------
    def _run_stage_a(self) -> None:
        """Train epoch encoder (stem + spectral + graph) with stage + N1 heads."""
        self._freeze_module(self.model.sequence_decoder)
        self._freeze_module(self.model.heads.boundary_head)
        self._freeze_module(self.model.heads.prev_head)
        self._freeze_module(self.model.heads.next_head)

        optimizer = self._build_optimizer(self.config.curriculum.stage_a_lr)
        scheduler = build_scheduler(optimizer, self.config.scheduler)
        stopper = EarlyStopping(patience=self.config.patience)

        for epoch in range(self.config.curriculum.stage_a_epochs):
            train_loss = self._train_one_epoch(optimizer, stage="A")
            val_metrics = self.evaluator.evaluate(
                self.model, self.val_loader, self.spectral_extractor,
            )

            self._log_epoch("A", epoch, train_loss, val_metrics)
            scheduler.step()

            self._save_if_best(val_metrics)
            if stopper.step(val_metrics["macro_f1"]):
                logger.info(f"Stage A early stop at epoch {epoch}")
                break

        self._unfreeze_all()

    # ------------------------------------------------------------------
    # Stage B: Sequence decoder (encoder frozen)
    # ------------------------------------------------------------------
    def _run_stage_b(self) -> None:
        """Train sequence decoder with encoder frozen."""
        self._freeze_module(self.model.waveform_stem)
        self._freeze_module(self.model.spectral_encoder)
        self._freeze_module(self.model.graph_encoder)

        optimizer = self._build_optimizer(self.config.curriculum.stage_b_lr)
        scheduler = build_scheduler(optimizer, self.config.scheduler)
        stopper = EarlyStopping(patience=self.config.patience)

        for epoch in range(self.config.curriculum.stage_b_epochs):
            train_loss = self._train_one_epoch(optimizer, stage="B")
            val_metrics = self.evaluator.evaluate(
                self.model, self.val_loader, self.spectral_extractor,
            )

            self._log_epoch("B", epoch, train_loss, val_metrics)
            scheduler.step()

            self._save_if_best(val_metrics)
            if stopper.step(val_metrics["macro_f1"]):
                logger.info(f"Stage B early stop at epoch {epoch}")
                break

        self._unfreeze_all()

    # ------------------------------------------------------------------
    # Stage C: End-to-end fine-tuning
    # ------------------------------------------------------------------
    def _run_stage_c(self) -> None:
        """Fine-tune entire model end-to-end."""
        optimizer = self._build_optimizer(self.config.curriculum.stage_c_lr)
        scheduler = build_scheduler(optimizer, self.config.scheduler)
        stopper = EarlyStopping(patience=self.config.patience)

        for epoch in range(self.config.curriculum.stage_c_epochs):
            train_loss = self._train_one_epoch(optimizer, stage="C")
            val_metrics = self.evaluator.evaluate(
                self.model, self.val_loader, self.spectral_extractor,
            )

            self._log_epoch("C", epoch, train_loss, val_metrics)
            scheduler.step()

            self._save_if_best(val_metrics)
            if stopper.step(val_metrics["macro_f1"]):
                logger.info(f"Stage C early stop at epoch {epoch}")
                break

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------
    def _train_one_epoch(self, optimizer: torch.optim.Optimizer, stage: str = "") -> float:
        """Train for one epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train {stage}", leave=False)
        for batch in pbar:
            signals = batch["signal"].to(self.device)

            # Use pre-computed spectral features if available, else extract
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

        return total_loss / max(num_batches, 1)

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

    def _save_if_best(self, metrics: dict[str, float]) -> None:
        state = {
            "model": self.model.state_dict(),
            "metrics": metrics,
        }
        saved = self.checkpoint.step(metrics["macro_f1"], state)
        if saved:
            self.best_metrics = metrics
            logger.info(f"  New best MF1: {metrics['macro_f1']:.4f}")

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
