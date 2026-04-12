"""Checkpoint and I/O utilities."""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    state: dict[str, Any],
    filepath: str | Path,
) -> None:
    """Save model checkpoint to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load model checkpoint from disk."""
    return torch.load(filepath, map_location=device, weights_only=False)
