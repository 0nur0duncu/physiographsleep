"""Evaluate a trained PhysioGraphSleep model."""

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from physiographsleep.configs import ExperimentConfig
from physiographsleep.data.dataset import SleepEDFDataset
from physiographsleep.data.loader import load_sleep_edf
from physiographsleep.data.spectral import SpectralFeatureExtractor
from physiographsleep.evaluation.metrics import MetricsCalculator
from physiographsleep.evaluation.postprocessing import HMMPostProcessor
from physiographsleep.evaluation.visualization import plot_confusion_matrix
from physiographsleep.models.physiographsleep import PhysioGraphSleep
from physiographsleep.training.evaluator import Evaluator
from physiographsleep.utils.io_utils import load_checkpoint
from physiographsleep.utils.logging_utils import setup_logger
from physiographsleep.utils.reproducibility import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PhysioGraphSleep")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, default="physiographsleep/dataset/sleep-edfx")
    parser.add_argument("--use-hmm", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    config = ExperimentConfig()
    config.data.data_dir = args.data_dir

    set_seed(config.seed)
    device = get_device(args.device)
    logger = setup_logger()

    # Load data
    data = load_sleep_edf(config.data)
    test_ds = SleepEDFDataset(
        data["test"]["epochs"], data["test"]["labels"], config=config.data,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.data.batch_size, shuffle=False,
        num_workers=config.data.num_workers,
    )

    # Load model
    model = PhysioGraphSleep(config.model)
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # Evaluate
    spectral = SpectralFeatureExtractor(config.data)
    evaluator = Evaluator(device)
    metrics = evaluator.evaluate(model, test_loader, spectral)

    calc = MetricsCalculator()
    logger.info("=== Raw Results ===")
    logger.info(calc.format_report(metrics))

    # HMM post-processing
    if args.use_hmm:
        logger.info("=== HMM Post-processed Results ===")
        hmm = HMMPostProcessor()
        hmm.fit(data["train"]["labels"])

        # Re-run to get raw predictions
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                signals = batch["signal"].to(device)
                spec = evaluator._extract_spectral_batch(signals, spectral)
                outputs = model(signals, spec)
                preds = outputs["stage"].argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch["label"].numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        smoothed = hmm.decode(all_preds)
        hmm_metrics = calc.compute_all(all_labels, smoothed)
        logger.info(calc.format_report(hmm_metrics))

        # Confusion matrix
        from pathlib import Path
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cm = calc.confusion_matrix(all_labels, smoothed)
        plot_confusion_matrix(cm, save_path=out_dir / "confusion_matrix_hmm.png")
        logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
