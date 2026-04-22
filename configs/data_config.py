"""Data pipeline configuration."""

import sys
from dataclasses import dataclass, field
from pathlib import Path


def _default_num_workers() -> int:
    """Auto-select num_workers based on platform.

    Linux/macOS (including Colab): 2 — DataLoader workers run in parallel,
    augmentation + tensor conversion overlap with GPU forward/backward.
    Windows: 0 — `spawn` start method has heavy overhead and often hangs
    with small datasets; main-thread loading is faster in practice.
    """
    return 0 if sys.platform.startswith("win") else 2


def _default_batch_size() -> int:
    """Default batch size. 64 saturates T4/V100 for this 584K-param model
    without exceeding ~30% VRAM at seq_len=25. Keep 32 on CPU-only runs
    where VRAM isn't the bottleneck but convergence stability is.
    """
    try:
        import torch  # local import to avoid hard dependency at config parse
        return 64 if torch.cuda.is_available() else 32
    except ImportError:
        return 32


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""

    # Paths
    data_dir: str = "physiographsleep/dataset/sleep-edfx"
    cache_dir: str = "physiographsleep/dataset/cache"

    # Dataset
    # num_subjects: 20 → Sleep-EDF-20 (SC400..SC419 ilk 20 subject)
    #                78 → Sleep-EDF-78 Expanded (tüm SC4** auto-detected,
    #                     eksik/hatalı dosyalar atlanır, ~78 geçerli subject)
    #                None → downloaded sleep-cassette klasöründeki tüm
    #                     geçerli subject'leri kullan (esnek mod).
    # Cache anahtarı num_subjects ile parametrize — EDF-20 ve EDF-78
    # önbellekleri birbirine çarpışmaz.
    num_subjects: int | None = 20
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

    # Augmentation (Strategy 2a — April 2026)
    # Baseline was gaussian_noise_std=0.01, amplitude_scale_range=(0.9, 1.1).
    # Too mild for 20-subject Sleep-EDF (val MF1 ceiled at 0.787 while
    # train climbed to 0.88). Subject-domain gap dominates; stronger
    # per-epoch perturbations should regularize without semantic damage.
    # Per-channel independent perturbation preserves 1ch/2ch parity.
    # Revert to 0.01 / (0.9, 1.1) if val MF1 drops > 0.5 pp.
    use_augmentation: bool = True
    gaussian_noise_std: float = 0.03
    time_shift_max: int = 50  # samples (unchanged — already ±50 / 3000 ≈ 1.6%)
    amplitude_scale_range: tuple[float, float] = (0.8, 1.2)
    # Subject-level DC shift (Strategy 2b — April 2026). One constant
    # offset per (sequence, channel) shared across all L epochs in the
    # sequence. Simulates amplifier baseline / electrode impedance drift
    # that varies between subjects. σ=0.1 relative to z-scored signal
    # (signal σ ≈ 1) is ~10% of dynamic range — strong enough to break
    # DC-based subject identification, small enough to preserve semantic
    # content. Set to 0.0 to disable.
    dc_shift_std: float = 0.1

    # Loading
    # batch_size: T4/V100 için 64 (model 584K param, seq_len=25 → ~10 MB/batch).
    # Linux/Colab'da workers paralel, Windows'ta spawn overhead nedeniyle 0.
    batch_size: int = field(default_factory=_default_batch_size)
    num_workers: int = field(default_factory=_default_num_workers)
    pin_memory: bool = True

    @property
    def num_bands(self) -> int:
        return len(self.band_ranges)

    @property
    def features_per_band(self) -> int:
        return self.num_patches * 7  # 7 spectral features per patch per band

    @property
    def num_input_channels(self) -> int:
        """Number of waveform input channels: 1 (EEG only) or 2 (EEG + EOG)."""
        return 2 if self.use_eog else 1
