"""Data augmentations for EEG signals."""

import numpy as np

from ..configs.data_config import DataConfig


class SleepTransforms:
    """Composable EEG augmentations for training.

    Applied per-epoch or per-sequence (batch of epochs).
    Augmentations: Gaussian noise, time shift, amplitude scaling, time masking.
    """

    def __init__(self, config: DataConfig):
        self.noise_std = config.gaussian_noise_std
        self.shift_max = config.time_shift_max
        self.scale_range = config.amplitude_scale_range
        self.enabled = config.use_augmentation

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        """Apply random augmentations to one epoch.

        Args:
            epoch: (C, T) — single epoch signal

        Returns:
            Augmented epoch: (C, T)
        """
        if not self.enabled:
            return epoch

        epoch = epoch.copy()

        if np.random.random() < 0.5:
            epoch = self._add_noise(epoch)

        if np.random.random() < 0.3:
            epoch = self._time_shift(epoch)

        if np.random.random() < 0.5:
            epoch = self._amplitude_scale(epoch)

        if np.random.random() < 0.3:
            epoch = self._time_mask(epoch)

        return epoch

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
        T = seq.shape[-1]

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

    def _add_noise(self, epoch: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.noise_std, epoch.shape)
        return epoch + noise.astype(epoch.dtype)

    def _time_shift(self, epoch: np.ndarray) -> np.ndarray:
        """Random circular time shift."""
        shift = np.random.randint(-self.shift_max, self.shift_max + 1)
        return np.roll(epoch, shift, axis=-1)

    def _amplitude_scale(self, epoch: np.ndarray) -> np.ndarray:
        """Random amplitude scaling."""
        low, high = self.scale_range
        scale = np.random.uniform(low, high)
        return epoch * scale

    def _time_mask(self, epoch: np.ndarray) -> np.ndarray:
        """Zero out a random contiguous segment (SpecAugment-style time masking).

        Masks 5-15% of the epoch duration, forcing the model to use
        broader temporal context rather than relying on any single segment.
        """
        T = epoch.shape[-1]
        mask_len = np.random.randint(T // 20, T // 7 + 1)  # 5-14% of T
        start = np.random.randint(0, T - mask_len)
        epoch[..., start : start + mask_len] = 0.0
        return epoch
