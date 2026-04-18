"""Data pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""

    # Paths
    data_dir: str = "physiographsleep/dataset/sleep-edfx"
    cache_dir: str = "physiographsleep/dataset/cache"

    # Dataset
    num_subjects: int = 20
    channel: str = "EEG Fpz-Cz"
    use_eog: bool = False
    sampling_rate: int = 100
    epoch_duration: int = 30  # seconds
    epoch_samples: int = 3000  # sampling_rate × epoch_duration

    # Split
    train_subjects: int = 14
    val_subjects: int = 3
    seed: int = 42

    # Sequence
    seq_len: int = 25

    # Spectral
    band_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 11.0),
            "sigma": (11.0, 16.0),
            "beta": (16.0, 30.0),
        }
    )
    num_patches: int = 6
    patch_duration: int = 5  # seconds
    patch_samples: int = 500  # sampling_rate × patch_duration

    # Filtering
    bandpass_low: float = 0.3
    bandpass_high: float = 35.0

    # Wake trimming (literature standard: DeepSleepNet/TinySleepNet/AttnSleep/SleepTransformer)
    # Keep only N minutes of W before first sleep epoch and after last sleep epoch.
    # Set to 0 to disable. Standard value = 30 minutes.
    wake_trim_minutes: int = 30

    # Augmentation
    use_augmentation: bool = True
    gaussian_noise_std: float = 0.01
    time_shift_max: int = 50  # samples
    amplitude_scale_range: tuple[float, float] = (0.9, 1.1)

    # Loading
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    @property
    def num_bands(self) -> int:
        return len(self.band_ranges)

    @property
    def features_per_band(self) -> int:
        return self.num_patches * 7  # 7 spectral features per patch per band
