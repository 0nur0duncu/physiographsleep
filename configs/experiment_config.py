"""Experiment-level configuration — combines model, data, training."""

from dataclasses import dataclass, field

from .model_config import ModelConfig
from .data_config import DataConfig
from .train_config import TrainConfig


def sync_channel_config(config: "ExperimentConfig") -> "ExperimentConfig":
    """Propagate channel-count changes across the model configuration.

    Call this after toggling `config.data.use_eog` so the WaveformStem
    and SpectralTokenEncoder pick up the correct input dimensions.
    Idempotent — safe to call multiple times.

    Must also be called before instantiating `PhysioGraphSleep`, otherwise
    the 2-channel path in the waveform stem and the wider spectral
    features (84 instead of 42) will not be configured.
    """
    C = config.data.num_input_channels
    config.model.waveform.in_channels = C
    # Per-band spectral feature count scales with the number of input
    # channels (each channel contributes 6 patches × 7 features = 42).
    config.model.spectral.features_per_band = config.data.features_per_band * C
    return config


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Experiment metadata
    name: str = "physiographsleep_v1"
    description: str = ""
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    debug: bool = False

    def __post_init__(self) -> None:
        # Ensure defaults are consistent (1ch EEG by default).
        sync_channel_config(self)
