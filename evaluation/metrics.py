"""Metrics calculator — accuracy, macro-F1, kappa, per-class F1, confusion matrix."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


class MetricsCalculator:
    """Compute all sleep staging evaluation metrics."""

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute full metric suite.

        Args:
            y_true: (N,) ground truth labels
            y_pred: (N,) predicted labels

        Returns:
            dict with all metrics
        """
        result = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true, y_pred, average="macro",
                zero_division=0, labels=range(5),
            ),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }

        # Per-class F1
        per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(5))
        for i, name in enumerate(STAGE_NAMES):
            result[f"f1_{name}"] = per_class[i] if i < len(per_class) else 0.0

        return result

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(y_true, y_pred, labels=range(5))

    @staticmethod
    def format_report(metrics: dict[str, float]) -> str:
        """Format metrics as readable string."""
        lines = [
            f"ACC:  {metrics['accuracy']:.4f}",
            f"MF1:  {metrics['macro_f1']:.4f}",
            f"κ:    {metrics['kappa']:.4f}",
            f"MCC:  {metrics['mcc']:.4f}",
            "Per-class F1:",
        ]
        for name in STAGE_NAMES:
            key = f"f1_{name}"
            if key in metrics:
                lines.append(f"  {name}: {metrics[key]:.4f}")
        return "\n".join(lines)
