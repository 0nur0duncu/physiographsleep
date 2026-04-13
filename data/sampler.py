"""N1-aware weighted sampler for class-imbalanced sleep staging."""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(
    labels: np.ndarray,
    n1_boost: float = 2.0,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that oversamples minority classes.

    Args:
        labels: (N,) integer labels 0–4
        n1_boost: extra multiplier for N1 (class 1) sampling weight

    Returns:
        WeightedRandomSampler instance
    """
    class_counts = np.bincount(labels, minlength=5).astype(np.float64)
    class_weights = 1.0 / (class_counts + 1e-6)

    if n1_boost > 1.0:
        class_weights[1] *= n1_boost

    class_weights /= class_weights.sum()

    sample_weights = class_weights[labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
