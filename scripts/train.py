"""Train PhysioGraphSleep model."""

import argparse
import logging
import sys
import warnings
import os

# Suppress noisy warnings before any imports
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Channels contain different.*")
warnings.filterwarnings("ignore", message=".*Highpass cutoff frequency.*")
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
os.environ["MIOPEN_LOG_LEVEL"] = "0"

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from physiographsleep.configs import ExperimentConfig
from physiographsleep.data.dataset import SleepEDFDataset
from physiographsleep.data.loader import load_sleep_edf
from physiographsleep.data.sampler import build_weighted_sampler
from physiographsleep.data.spectral import SpectralFeatureExtractor
from physiographsleep.data.transforms import SleepTransforms
from physiographsleep.models.losses import MultiTaskLoss
from physiographsleep.models.physiographsleep import PhysioGraphSleep
from physiographsleep.training.trainer import Trainer
from physiographsleep.utils.logging_utils import setup_logger
from physiographsleep.utils.reproducibility import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhysioGraphSleep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="physiographsleep/dataset/sleep-edfx")
    args = parser.parse_args()

    # Config
    config = ExperimentConfig()
    config.seed = args.seed
    config.device = args.device
    config.debug = args.debug
    config.data.batch_size = args.batch_size
    config.data.data_dir = args.data_dir

    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger(log_dir=config.train.log_dir)
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading Sleep-EDF data...")
    data = load_sleep_edf(config.data)

    # Transforms
    train_transform = SleepTransforms(config.data)

    # Datasets
    train_ds = SleepEDFDataset(
        data["train"]["epochs"], data["train"]["labels"],
        config=config.data, transform=train_transform,
        spectral=data["train"].get("spectral"),
    )
    val_ds = SleepEDFDataset(
        data["val"]["epochs"], data["val"]["labels"],
        config=config.data,
        spectral=data["val"].get("spectral"),
    )

    # Sampler
    sampler = build_weighted_sampler(data["train"]["labels"])

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=config.data.batch_size,
        sampler=sampler, num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.data.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # Model
    model = PhysioGraphSleep(config.model)
    param_counts = model.count_parameters()
    logger.info("Parameter counts:")
    for name, count in param_counts.items():
        logger.info(f"  {name}: {count:,}")

    # Loss
    class_counts = np.bincount(data["train"]["labels"], minlength=5)
    class_weights = 1.0 / (class_counts.astype(np.float32) + 1e-6)
    class_weights = class_weights / class_weights.sum() * 5

    # v2: Extra N1 (class 1) weight boost
    class_weights[1] *= 2.5

    class_weights_tensor = torch.from_numpy(class_weights).float()

    # v2: Per-class gamma — higher gamma for N1
    per_class_gamma = {1: 4.0}

    loss_fn = MultiTaskLoss(
        config.train.loss,
        class_weights=class_weights_tensor,
        per_class_gamma=per_class_gamma,
    )

    # Spectral extractor
    spectral = SpectralFeatureExtractor(config.data)

    # Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.train,
        device=device,
        spectral_extractor=spectral,
    )

    # Train
    logger.info("Starting training...")
    best_metrics = trainer.train()

    logger.info("Training complete!")
    from physiographsleep.evaluation.metrics import MetricsCalculator
    logger.info(MetricsCalculator.format_report(best_metrics))


if __name__ == "__main__":
    main()
