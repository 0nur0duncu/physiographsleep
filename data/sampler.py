"""N1-aware weighted sampler for class-imbalanced sleep staging."""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that oversamples minority classes.

    Args:
        labels: (N,) integer labels 0–4

    Returns:
        WeightedRandomSampler instance
    """
    class_counts = np.bincount(labels, minlength=5).astype(np.float64)
    class_weights = 1.0 / (class_counts + 1e-6)

    # Extra boost for N1 (class 1) — aggressive oversampling for extreme minority
    class_weights[1] *= 4.0

    # Normalize
    class_weights /= class_weights.sum()

    sample_weights = class_weights[labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
