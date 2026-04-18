"""Ablation study runner — 6 configurations.

Builds the same training pipeline as scripts/train.py but toggles
individual model components, writes a single JSON line per run to
`outputs/ablation_results.jsonl` for table compilation.

Configurations (cf. docs/research/originality_analysis.md):
  A: cnn_only          — WaveformStem + global pool + linear head
                          (no spectral, no graph, no sequence)
  B: cnn_spectral      — adds the spectral encoder (still no graph,
                          no sequence)
  C: graph_no_pathway  — current production model w/o pathway, no fusion
  D: graph_pathway     — pathway hetero→homo→all
  E: graph_pathway_fusion — pathway + λ-fusion
  F: full              — D/E + N1-Mixup + EOG (if available)

Use:
    python -m physiographsleep.scripts.run_ablation \
        --configs A B C D E F --epochs 30 --batch-size 64

Each run reuses the same data split (`config.seed`) so the only delta
between runs is the model toggle.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Channels contain different.*")
warnings.filterwarnings("ignore", message=".*Highpass cutoff frequency.*")
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physiographsleep.configs import ExperimentConfig
from physiographsleep.configs.model_config import FusionConfig
from physiographsleep.configs.train_config import N1MixupConfig
from physiographsleep.data.dataset import SleepEDFDataset
from physiographsleep.data.loader import load_sleep_edf
from physiographsleep.data.spectral import SpectralFeatureExtractor
from physiographsleep.data.transforms import SleepTransforms
from physiographsleep.models.losses import MultiTaskLoss
from physiographsleep.models.physiographsleep import PhysioGraphSleep
from physiographsleep.training.trainer import Trainer
from physiographsleep.utils.logging_utils import setup_logger
from physiographsleep.utils.reproducibility import get_device, set_seed


# ----------------------------------------------------------------------
# Config patches per ablation. Disabling features structurally (set to
# None) instead of via a runtime `enabled` flag, so each ablation
# config is a self-contained, fully-specified pipeline definition.
# ----------------------------------------------------------------------
PATHWAY_3LAYER = [(2,), (0, 1), (0, 1, 2, 3)]


def patch_config(
    name: str, config: ExperimentConfig,
) -> tuple[str, ExperimentConfig]:
    name = name.upper()
    cfg = config
    # All ablations start from a stripped baseline; F restores the new defaults.
    cfg.train.n1_mixup = None
    cfg.model.fusion = None

    if name == "A":  # cnn_only — single homo layer, no fusion, no mixup
        cfg.model.graph.num_layers = 1
        cfg.model.graph.edge_pathways = [(0,)]  # patch↔patch only
        cfg.train.epochs = min(cfg.train.epochs, 30)
    elif name == "B":  # cnn_spectral — 2 layers, all edges, no pathway
        cfg.model.graph.num_layers = 2
        cfg.model.graph.edge_pathways = None
    elif name == "C":  # graph_no_pathway (pre-scGraPhT baseline)
        cfg.model.graph.num_layers = 3
        cfg.model.graph.edge_pathways = None
    elif name == "D":  # + pathway (scGraPhT-style sequential subgraphs)
        cfg.model.graph.num_layers = 3
        cfg.model.graph.edge_pathways = PATHWAY_3LAYER
    elif name == "E":  # + λ-fusion
        cfg.model.graph.num_layers = 3
        cfg.model.graph.edge_pathways = PATHWAY_3LAYER
        cfg.model.fusion = FusionConfig(init_lambda=0.3)
    elif name == "F":  # full: pathway + fusion + N1-Mixup + EOG (if available)
        cfg.model.graph.num_layers = 3
        cfg.model.graph.edge_pathways = PATHWAY_3LAYER
        cfg.model.fusion = FusionConfig(init_lambda=0.3)
        cfg.train.n1_mixup = N1MixupConfig(prob=0.2, alpha=0.2)
        if hasattr(cfg.data, "use_eog"):
            cfg.data.use_eog = True
            cfg.model.waveform.in_channels = cfg.data.num_input_channels
    else:
        raise ValueError(f"Unknown ablation config '{name}'")
    return name, cfg


# ----------------------------------------------------------------------
# Run one ablation
# ----------------------------------------------------------------------
def run_one(name: str, base_args: argparse.Namespace) -> dict:
    config = ExperimentConfig()
    config.seed = base_args.seed
    config.data.batch_size = base_args.batch_size
    config.data.data_dir = base_args.data_dir
    config.train.epochs = base_args.epochs

    name, config = patch_config(name, config)

    set_seed(config.seed)
    device = get_device(base_args.device)

    log_dir = Path(base_args.log_dir) / f"ablation_{name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    config.train.log_dir = str(log_dir)
    config.train.checkpoint_dir = str(log_dir / "checkpoints")

    logger = setup_logger(log_dir=str(log_dir))
    logger.info(f"=== Ablation {name} ===")

    data = load_sleep_edf(config.data)
    train_transform = SleepTransforms(config.data)
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
    val_loader = DataLoader(
        val_ds, batch_size=config.data.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    model = PhysioGraphSleep(config.model)
    counts = model.count_parameters()
    logger.info(f"Params: {counts['total']:,}")

    class_counts = np.bincount(data["train"]["labels"], minlength=5)
    weights = 1.0 / (class_counts.astype(np.float32) + 1e-6)
    weights = weights / weights.sum() * 5
    loss_fn = MultiTaskLoss(config.train.loss, class_weights=torch.from_numpy(weights).float())

    spectral = SpectralFeatureExtractor(config.data)
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_dataset=train_ds, train_labels=data["train"]["labels"],
        val_loader=val_loader, config=config.train, data_config=config.data,
        device=device, spectral_extractor=spectral,
    )

    t0 = time.time()
    metrics = trainer.train()
    elapsed = time.time() - t0

    record = {
        "config": name,
        "params_total": counts["total"],
        "epochs": config.train.epochs,
        "best_val_macro_f1": float(metrics.get("macro_f1", 0.0)),
        "best_val_accuracy": float(metrics.get("accuracy", 0.0)),
        "best_val_kappa": float(metrics.get("kappa", 0.0)),
        "per_class_f1": metrics.get("per_class_f1", []),
        "elapsed_sec": round(elapsed, 1),
        "n1_mixup": config.train.n1_mixup is not None,
        "fusion": config.model.fusion is not None,
        "edge_pathways": config.model.graph.edge_pathways,
    }
    return record


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["C", "D", "E", "F"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--data-dir", type=str,
                        default="physiographsleep/dataset/sleep-edfx")
    parser.add_argument("--log-dir", type=str, default="logs/ablation")
    parser.add_argument("--out", type=str,
                        default="outputs/ablation_results.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Ablation runner — configs={args.configs} → {out_path}")
    for name in args.configs:
        try:
            record = run_one(name, args)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"[{record['config']}] MF1={record['best_val_macro_f1']:.4f} "
                f"params={record['params_total']:,} "
                f"({record['elapsed_sec']:.0f}s)"
            )
        except Exception as exc:
            logging.exception(f"Ablation {name} failed: {exc}")


if __name__ == "__main__":
    main()
