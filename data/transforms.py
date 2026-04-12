"""Data augmentations for EEG signals."""

import numpy as np

from ..configs.data_config import DataConfig


class SleepTransforms:
    """Composable EEG augmentations for training.

    Applied per-epoch (single epoch signal).
    Augmentations: Gaussian noise, time shift, amplitude scaling.
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

        return epoch

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
