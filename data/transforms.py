"""Data augmentations for EEG signals."""

import numpy as np

from ..configs.data_config import DataConfig


class SleepTransforms:
    """Composable EEG augmentations for training.

    Applied per-sequence (a tensor of L epochs) via `transform_sequence`.
    Augmentations: Gaussian noise, time shift, amplitude scaling, time masking.
    """

    def __init__(self, config: DataConfig):
        self.noise_std = config.gaussian_noise_std
        self.shift_max = config.time_shift_max
        self.scale_range = config.amplitude_scale_range
        self.dc_shift_std = config.dc_shift_std
        self.enabled = config.use_augmentation

    def transform_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a sequence of epochs (vectorized).

        Args:
            seq: (L, C, T) — sequence of epochs

        Returns:
            Augmented sequence: (L, C, T)
        """
        if not self.enabled:
            return seq

        seq = seq.copy()
        L = seq.shape[0]
        C = seq.shape[1]
        T = seq.shape[-1]

        # Subject-level DC shift — one constant per (sequence, channel),
        # same across all L epochs. Simulates amplifier baseline drift /
        # electrode impedance offset that differ between subjects.
        # Applied with p=0.8 because subject-domain bias is the dominant
        # overfit source at Sleep-EDF-20 scale (20 subjects → any DC cue
        # can leak subject identity). 2ch-aware: each channel gets its
        # own independent shift.
        if self.dc_shift_std > 0 and np.random.random() < 0.8:
            dc = np.random.normal(
                0.0, self.dc_shift_std, size=(1, C, 1),
            ).astype(seq.dtype)
            seq += dc

        # Gaussian noise — independent per epoch
        noise_mask = np.random.random(L) < 0.5
        if noise_mask.any():
            noise = np.random.normal(0, self.noise_std, seq[noise_mask].shape)
            seq[noise_mask] += noise.astype(seq.dtype)

        # Amplitude scale — independent per epoch
        scale_mask = np.random.random(L) < 0.5
        n_scale = scale_mask.sum()
        if n_scale > 0:
            low, high = self.scale_range
            scales = np.random.uniform(low, high, size=(n_scale, 1, 1)).astype(seq.dtype)
            seq[scale_mask] *= scales

        # Time shift — independent per epoch (needs loop, but cheap)
        shift_mask = np.random.random(L) < 0.3
        for i in np.where(shift_mask)[0]:
            shift = np.random.randint(-self.shift_max, self.shift_max + 1)
            seq[i] = np.roll(seq[i], shift, axis=-1)

        # Time mask — independent per epoch (needs loop for variable mask length)
        tmask_mask = np.random.random(L) < 0.3
        for i in np.where(tmask_mask)[0]:
            mask_len = np.random.randint(T // 20, T // 7 + 1)
            start = np.random.randint(0, T - mask_len)
            seq[i, ..., start : start + mask_len] = 0.0

        return seq
