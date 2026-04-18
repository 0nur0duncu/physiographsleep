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
from torch.amp import GradScaler, autocast
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
logger.propagate = False  # avoid duplicate logs in jupyter/colab


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
        # Perf: pick fastest cuDNN kernels for our fixed input shape
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        # bf16 on Ada/Hopper/Blackwell skips GradScaler entirely
        self.amp_dtype = torch.bfloat16 if (
            device.type == "cuda" and torch.cuda.is_bf16_supported()
        ) else torch.float16
        self.use_scaler = self.amp_enabled and self.amp_dtype == torch.float16

    def _build_train_loader(self, n1_boost: float | None = None) -> DataLoader:
        """Build a training DataLoader with optional N1-boosted sampling."""
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
        """Run curriculum training from a given stage.

        Stages B and C are optional and OFF by default. The final loaded
        model is the best across all enabled stages (by val macro-F1).
        """
        stages = ["A", "B", "C"]
        start_idx = stages.index(start_stage.upper())
        enable_b = getattr(self.config.curriculum, "enable_stage_b", False)
        enable_c = getattr(self.config.curriculum, "enable_stage_c", False)

        ran_stages: list[str] = []

        if start_idx <= 0:
            logger.info("=== Stage A: Joint encoder+decoder pretraining ===")
            self._run_stage_a()
            ran_stages.append("A")

        if start_idx <= 1 and enable_b:
            self._load_stage_checkpoint("A")
            logger.info("=== Stage B: Decoder fine-tune (encoder frozen) ===")
            self._run_stage_b()
            ran_stages.append("B")

        if start_idx <= 2 and enable_c:
            loaded_b = enable_b and self._load_stage_checkpoint("B")
            if not loaded_b:
                self._load_stage_checkpoint("A")
            logger.info("=== Stage C: End-to-end fine-tune (adaptive loss) ===")
            self._run_stage_c()
            ran_stages.append("C")

        # Pick the best stage by reading saved checkpoints (source of truth).
        best_stage, best_mf1 = None, -1.0
        for s in ran_stages:
            ckpt_path = self.config.checkpoint_dir + f"/stage-{s.lower()}.pt"
            try:
                from physiographsleep.utils.io_utils import load_checkpoint
                ck = load_checkpoint(ckpt_path, self.device)
                m = float(ck.get("metrics", {}).get("macro_f1", 0.0))
                logger.info(f"  Stage {s} best val MF1 = {m:.4f}")
                if m > best_mf1:
                    best_mf1, best_stage = m, s
            except FileNotFoundError:
                continue

        if best_stage is not None:
            logger.info(f"=== Best across stages: {best_stage} (MF1={best_mf1:.4f}) ===")
            self._load_stage_checkpoint(best_stage)
        return self.best_metrics

    # ------------------------------------------------------------------
    # Stage A: Joint encoder + decoder pretraining (N1 boost 2.0x)
    # ------------------------------------------------------------------
    def _run_stage_a(self) -> None:
        # Only freeze auxiliary heads that are not used to drive Stage A objective.
        # Decoder + main + aux N1 heads remain trainable so the entire pipeline
        # learns end-to-end with N1 boost from epoch 0.
        self._freeze_module(self.model.heads.boundary_head)
        self._freeze_module(self.model.heads.prev_head)
        self._freeze_module(self.model.heads.next_head)

        loader = self._build_train_loader(n1_boost=2.0)
        optimizer = self._build_optimizer(self.config.curriculum.stage_a_lr)
        scheduler = build_scheduler(optimizer, self.config.scheduler)
        stopper = EarlyStopping(patience=self.config.patience)
        checkpoint = ModelCheckpoint(self.config.checkpoint_dir, mode="max")

        for epoch in range(self.config.curriculum.stage_a_epochs):
            train_loss, tr_preds, tr_labels = self._train_one_epoch(
                optimizer, loader, stage="A",
            )
            train_metrics = self._train_metrics(tr_labels, tr_preds)
            val_loss, val_metrics = self._evaluate_with_loss()

            self._log_epoch("A", epoch, train_loss, val_loss, val_metrics, train_metrics)
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
            train_loss, tr_preds, tr_labels = self._train_one_epoch(
                optimizer, loader, stage="B",
            )
            train_metrics = self._train_metrics(tr_labels, tr_preds)
            val_loss, val_metrics = self._evaluate_with_loss()

            self._log_epoch("B", epoch, train_loss, val_loss, val_metrics, train_metrics)
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
            train_loss, tr_preds, tr_labels = self._train_one_epoch(
                optimizer, loader, stage="C",
            )
            train_metrics = self._train_metrics(tr_labels, tr_preds)

            if use_adaptive:
                class_f1 = self._compute_per_class_f1(tr_labels, tr_preds)
                self.loss_fn.update_adaptive_weights(
                    class_f1,
                    K=self.config.adaptive_loss.K,
                    gamma=self.config.adaptive_loss.gamma,
                )
                logger.info(
                    f"  Adaptive F1: {[f'{f:.3f}' for f in class_f1]} | "
                    f"Weights: {[f'{w:.2f}' for w in self.loss_fn.focal.weight.tolist()]}"
                )

            val_loss, val_metrics = self._evaluate_with_loss()

            self._log_epoch("C", epoch, train_loss, val_loss, val_metrics, train_metrics)
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
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Train for one epoch. Returns (mean_loss, preds, labels)."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Train {stage}", leave=False)
        gpu_preds: list[torch.Tensor] = []
        gpu_labels: list[torch.Tensor] = []
        for batch in pbar:
            signals = batch["signal"].to(self.device, non_blocking=True)

            if "spectral" in batch:
                spectral = batch["spectral"].to(self.device, non_blocking=True)
            else:
                spectral = self.evaluator._extract_spectral_batch(
                    signals, self.spectral_extractor,
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
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.grad_clip,
                )
                optimizer.step()

            total_loss += losses["total"].item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

            # Keep on GPU; transfer once after epoch (saves ~1k device syncs)
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
        """Compute lightweight train metrics for logging."""
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
        self, stage: str, epoch: int, loss: float,
        val_loss: float, metrics: dict[str, float],
        train_metrics: dict[str, float] | None = None,
    ) -> None:
        train_extra = ""
        if train_metrics is not None:
            train_extra = (
                f"TrainACC={train_metrics.get('accuracy', 0):.4f} | "
                f"TrainMF1={train_metrics.get('macro_f1', 0):.4f} | "
            )
        logger.info(
            f"[{stage}] Epoch {epoch:02d} | "
            f"TrLoss={loss:.4f} | VlLoss={val_loss:.4f} | "
            f"{train_extra}"
            f"ValACC={metrics['accuracy']:.4f} | "
            f"ValMF1={metrics['macro_f1']:.4f} | "
            f"\u03ba={metrics['kappa']:.4f}"
        )
        if self.callback is not None:
            try:
                self.callback(
                    stage=stage, epoch=epoch,
                    train_loss=loss, val_loss=val_loss,
                    train_metrics=train_metrics or {}, val_metrics=metrics,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Callback failed: {exc}")

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    def _unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def _evaluate_with_loss(self) -> tuple[float, dict[str, float]]:
        """Run evaluator AND compute mean validation loss in one pass."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        gpu_preds: list[torch.Tensor] = []
        gpu_labels: list[torch.Tensor] = []

        for batch in self.val_loader:
            signals = batch["signal"].to(self.device, non_blocking=True)
            spectral = batch["spectral"].to(self.device, non_blocking=True) if "spectral" in batch else \
                self.evaluator._extract_spectral_batch(signals, self.spectral_extractor)
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
