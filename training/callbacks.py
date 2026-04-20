"""Early stopping and model checkpoint callbacks."""

from pathlib import Path
from typing import Any

import torch

from ..utils.io_utils import save_checkpoint


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-3):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Update state and return True if should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """Save best model checkpoint based on monitored metric."""

    def __init__(self, save_dir: str, mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.mode = mode
        self.best_score: float | None = None

    def step(self, score: float, state: dict[str, Any], filename: str = "best.pt") -> bool:
        """Save if score improved. Returns True if saved."""
        if self.best_score is None:
            self.best_score = score
            save_checkpoint(state, self.save_dir / filename)
            return True

        improved = (
            score > self.best_score if self.mode == "max"
            else score < self.best_score
        )

        if improved:
            self.best_score = score
            save_checkpoint(state, self.save_dir / filename)
            return True

        return False
