"""K-fold subject-wise cross-validation runner.

Implements the MUST-have evaluation protocol that all 3 strategy docs
(`research/improvement_strategies.md` S2, `originality_analysis.md`,
`fair_comparison_analysis.md`) flag for fair SOTA comparison and reviewer
expectations.

For Sleep-EDF-20 the convention is 20-fold leave-one-subject-out (LOSO)
or k-fold subject-wise. We support both via `--folds N`:
  --folds 20 → LOSO (test=1 subject, val=2 subjects, train=17)
  --folds 10 → 10-fold (test=2 subjects each)

Per-fold metrics are appended as JSONL to `outputs/cv_results.jsonl`.
A final "summary" line aggregates mean ± std across folds.

Usage:
    python -m physiographsleep.scripts.run_cv --folds 20 --epochs 30
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
from physiographsleep.data.dataset import SleepEDFDataset, get_subject_ids
from physiographsleep.data.loader import load_sleep_edf_per_subject
from physiographsleep.data.spectral import SpectralFeatureExtractor
from physiographsleep.data.transforms import SleepTransforms
from physiographsleep.models.losses import MultiTaskLoss
from physiographsleep.models.physiographsleep import PhysioGraphSleep
from physiographsleep.training.trainer import Trainer
from physiographsleep.utils.logging_utils import setup_logger
from physiographsleep.utils.reproducibility import get_device, set_seed


def _build_fold_splits(
    subject_ids: list[str], n_folds: int, seed: int = 42,
) -> list[dict[str, list[str]]]:
    """Return list of {train, val, test} subject-id splits.

    Test fold rotates through `n_folds` chunks of subjects. Validation
    fold takes the next 2 subjects (cyclically) so train > val > test
    discipline is preserved across folds.
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(subject_ids).tolist()
    n = len(shuffled)

    if n_folds > n:
        raise ValueError(f"n_folds={n_folds} > num_subjects={n}")

    chunk = max(1, n // n_folds)
    splits: list[dict[str, list[str]]] = []
    for k in range(n_folds):
        test_start = k * chunk
        test_end = (k + 1) * chunk if k < n_folds - 1 else n
        test = shuffled[test_start:test_end]
        # 2 validation subjects taken cyclically just after the test block
        rest = [s for s in shuffled if s not in test]
        val_n = max(1, min(2, len(rest) // 8))
        val = rest[:val_n]
        train = rest[val_n:]
        splits.append({"train": sorted(train), "val": sorted(val), "test": sorted(test)})
    return splits


def _concat_split(
    per_subj: dict[str, dict[str, np.ndarray]],
    subject_ids: list[str],
) -> dict[str, np.ndarray]:
    epochs, labels, spectral = [], [], []
    for sid in subject_ids:
        if sid not in per_subj:
            continue
        epochs.append(per_subj[sid]["epochs"])
        labels.append(per_subj[sid]["labels"])
        spectral.append(per_subj[sid]["spectral"])
    return {
        "epochs": np.concatenate(epochs, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "spectral": np.concatenate(spectral, axis=0),
    }


def run_one_fold(
    fold_idx: int,
    splits: dict[str, list[str]],
    per_subj: dict[str, dict[str, np.ndarray]],
    base_args: argparse.Namespace,
) -> dict:
    config = ExperimentConfig()
    config.seed = base_args.seed + fold_idx  # different init per fold
    config.data.batch_size = base_args.batch_size
    config.train.epochs = base_args.epochs

    set_seed(config.seed)
    device = get_device(base_args.device)

    log_dir = Path(base_args.log_dir) / f"fold_{fold_idx:02d}"
    log_dir.mkdir(parents=True, exist_ok=True)
    config.train.log_dir = str(log_dir)
    config.train.checkpoint_dir = str(log_dir / "checkpoints")

    logger = setup_logger(log_dir=str(log_dir))
    logger.info(
        f"=== Fold {fold_idx} | train={splits['train']} | "
        f"val={splits['val']} | test={splits['test']} ==="
    )

    train_data = _concat_split(per_subj, splits["train"])
    val_data = _concat_split(per_subj, splits["val"])
    test_data = _concat_split(per_subj, splits["test"])

    train_transform = SleepTransforms(config.data)
    train_ds = SleepEDFDataset(
        train_data["epochs"], train_data["labels"],
        config=config.data, transform=train_transform,
        spectral=train_data["spectral"],
    )
    val_ds = SleepEDFDataset(
        val_data["epochs"], val_data["labels"],
        config=config.data, spectral=val_data["spectral"],
    )
    test_ds = SleepEDFDataset(
        test_data["epochs"], test_data["labels"],
        config=config.data, spectral=test_data["spectral"],
    )

    val_loader = DataLoader(
        val_ds, batch_size=config.data.batch_size, shuffle=False,
        num_workers=config.data.num_workers, pin_memory=config.data.pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.data.batch_size, shuffle=False,
        num_workers=config.data.num_workers, pin_memory=config.data.pin_memory,
    )

    model = PhysioGraphSleep(config.model)

    class_counts = np.bincount(train_data["labels"], minlength=5)
    weights = 1.0 / (class_counts.astype(np.float32) + 1e-6)
    weights = weights / weights.sum() * 5
    loss_fn = MultiTaskLoss(
        config.train.loss, class_weights=torch.from_numpy(weights).float(),
    )

    spectral = SpectralFeatureExtractor(config.data)
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_dataset=train_ds, train_labels=train_data["labels"],
        val_loader=val_loader, config=config.train, data_config=config.data,
        device=device, spectral_extractor=spectral,
    )

    t0 = time.time()
    val_metrics = trainer.train()
    elapsed = time.time() - t0

    # Final evaluation on the held-out *test* subjects
    test_loss, test_metrics = trainer._evaluate_with_loss_external(test_loader) \
        if hasattr(trainer, "_evaluate_with_loss_external") else (None, {})

    # Fall back to internal evaluator if no external helper exists
    if not test_metrics:
        from physiographsleep.evaluation.evaluator import Evaluator
        ev = Evaluator(device)
        # Switch trainer.val_loader so we can reuse _evaluate_with_loss
        old_loader = trainer.val_loader
        trainer.val_loader = test_loader
        try:
            _, test_metrics = trainer._evaluate_with_loss()
        finally:
            trainer.val_loader = old_loader

    record = {
        "fold": fold_idx,
        "test_subjects": splits["test"],
        "val_subjects": splits["val"],
        "n_train_subjects": len(splits["train"]),
        "best_val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
        "best_val_accuracy": float(val_metrics.get("accuracy", 0.0)),
        "test_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
        "test_accuracy": float(test_metrics.get("accuracy", 0.0)),
        "test_kappa": float(test_metrics.get("kappa", 0.0)),
        "test_per_class_f1": test_metrics.get("per_class_f1", []),
        "elapsed_sec": round(elapsed, 1),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--log-dir", type=str, default="logs/cv")
    parser.add_argument("--out", type=str, default="outputs/cv_results.jsonl")
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=None,
                        help="exclusive upper bound; defaults to --folds")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig()
    print("Loading per-subject data (one-time cache build)...")
    per_subj = load_sleep_edf_per_subject(config.data)
    subject_ids = sorted(per_subj.keys())
    print(f"Loaded {len(subject_ids)} subjects: {subject_ids}")

    splits = _build_fold_splits(subject_ids, args.folds, seed=args.seed)
    end = args.end_fold if args.end_fold is not None else len(splits)

    fold_records: list[dict] = []
    for k in range(args.start_fold, end):
        try:
            rec = run_one_fold(k, splits[k], per_subj, args)
            fold_records.append(rec)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"[fold {k:02d}] test_MF1={rec['test_macro_f1']:.4f} "
                f"acc={rec['test_accuracy']:.4f} "
                f"({rec['elapsed_sec']:.0f}s)"
            )
        except Exception as exc:
            logging.exception(f"Fold {k} failed: {exc}")

    if fold_records:
        mf1 = np.array([r["test_macro_f1"] for r in fold_records])
        acc = np.array([r["test_accuracy"] for r in fold_records])
        summary = {
            "summary": True,
            "n_folds": len(fold_records),
            "test_macro_f1_mean": float(mf1.mean()),
            "test_macro_f1_std": float(mf1.std(ddof=1) if mf1.size > 1 else 0.0),
            "test_accuracy_mean": float(acc.mean()),
            "test_accuracy_std": float(acc.std(ddof=1) if acc.size > 1 else 0.0),
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        print(
            f"\nCV summary: MF1={summary['test_macro_f1_mean']:.4f} "
            f"± {summary['test_macro_f1_std']:.4f}"
        )


if __name__ == "__main__":
    main()
