"""Spectral feature extraction — per-patch, per-band features for spectral tokens."""

import numpy as np
from scipy.signal import welch

from ..configs.data_config import DataConfig


class SpectralFeatureExtractor:
    """Extract spectral features from raw EEG epochs.

    For each epoch (3000 samples @ 100 Hz):
    - Split into 6 patches of 500 samples (5 seconds each)
    - For each patch, compute 7 features per 5 frequency bands
    - Output: (num_bands, features_per_band) = (5, 42)
      where features_per_band = 6 patches × 7 features

    The 5 bands: delta (0.5–4), theta (4–8), alpha (8–11), sigma (11–16), beta (16–30)
    The 7 features per patch per band:
        abs_power, rel_power, log_power, spectral_entropy,
        delta_theta_ratio, sigma_beta_ratio, theta_alpha_ratio
    """

    def __init__(self, config: DataConfig):
        self.fs = config.sampling_rate
        self.num_patches = config.num_patches
        self.patch_samples = config.patch_samples
        self.bands = config.band_ranges
        self.band_names = list(self.bands.keys())
        self.nperseg = min(256, self.patch_samples)

    def extract_epoch(self, signal: np.ndarray) -> np.ndarray:
        """Extract spectral features for one epoch.

        Args:
            signal: (T,) raw signal, T=3000

        Returns:
            features: (5, 42) — per-band feature matrix
        """
        patches = self._split_patches(signal)
        all_patch_features = []

        for patch in patches:
            patch_features = self._compute_patch_features(patch)
            all_patch_features.append(patch_features)

        # Stack: (6, 5, 7) → reshape to (5, 42)
        stacked = np.stack(all_patch_features, axis=0)  # (6, 5, 7)
        features = stacked.transpose(1, 0, 2).reshape(len(self.band_names), -1)
        return features  # (5, 42)

    def extract_batch(self, signals: np.ndarray) -> np.ndarray:
        """Extract spectral features for a batch.

        Args:
            signals: (B, T) — single-channel
                     (B, 1, T) — single-channel with explicit channel axis
                     (B, C, T) — multi-channel (C >= 2); features are
                                 concatenated along the feature axis so
                                 every channel contributes independent
                                 spectral descriptors (literature-standard
                                 early fusion for multi-modal PSG).

        Returns:
            features: (B, 5, 42 * C)  where C is the number of channels
                      present in `signals`. For C=1 this is (B, 5, 42).
        """
        from tqdm import tqdm
        if signals.ndim == 2:
            # (B, T) → treat as single-channel
            return np.stack(
                [self.extract_epoch(s) for s in tqdm(
                    signals, desc="Spectral features", leave=False,
                )],
                axis=0,
            )
        if signals.ndim != 3:
            raise ValueError(
                f"Expected signals of shape (B, T) or (B, C, T), got {signals.shape}"
            )

        # (B, C, T): extract per-channel, concatenate features along axis 2.
        B, C, _ = signals.shape
        per_channel = []
        for c in range(C):
            per_channel.append(
                np.stack(
                    [self.extract_epoch(s) for s in tqdm(
                        signals[:, c, :],
                        desc=f"Spectral ch{c}",
                        leave=False,
                    )],
                    axis=0,
                )
            )
        # Each element is (B, 5, 42); concat along feature dim → (B, 5, 42*C)
        return np.concatenate(per_channel, axis=2)

    def _split_patches(self, signal: np.ndarray) -> list[np.ndarray]:
        """Split epoch into 6 non-overlapping 5-second patches."""
        patches = []
        for i in range(self.num_patches):
            start = i * self.patch_samples
            end = start + self.patch_samples
            patches.append(signal[start:end])
        return patches

    def _compute_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """Compute 7 spectral features per band for a single patch.

        Returns:
            features: (5, 7) — one row per band
        """
        freqs, psd = welch(patch, fs=self.fs, nperseg=self.nperseg)
        total_power = np.sum(psd) + 1e-12

        band_powers = {}
        for name, (low, high) in self.bands.items():
            idx = np.where((freqs >= low) & (freqs < high))[0]
            band_powers[name] = np.sum(psd[idx]) + 1e-12

        features = []
        for name in self.band_names:
            bp = band_powers[name]
            abs_power = bp
            rel_power = bp / total_power
            log_power = np.log(bp)
            entropy = self._spectral_entropy(psd, freqs, self.bands[name])

            # Cross-band ratios (clamped to avoid inf)
            delta_theta = np.log(band_powers["delta"] / band_powers["theta"])
            sigma_beta = np.log(band_powers["sigma"] / band_powers["beta"])
            theta_alpha = np.log(band_powers["theta"] / band_powers["alpha"])

            features.append([
                abs_power, rel_power, log_power, entropy,
                delta_theta, sigma_beta, theta_alpha,
            ])

        return np.array(features, dtype=np.float32)  # (5, 7)

    @staticmethod
    def _spectral_entropy(
        psd: np.ndarray,
        freqs: np.ndarray,
        band: tuple[float, float],
    ) -> float:
        """Compute Shannon spectral entropy within a frequency band."""
        idx = np.where((freqs >= band[0]) & (freqs < band[1]))[0]
        if len(idx) == 0:
            return 0.0
        band_psd = psd[idx]
        band_psd = band_psd / (np.sum(band_psd) + 1e-12)
        entropy = -np.sum(band_psd * np.log(band_psd + 1e-12))
        return float(entropy)
