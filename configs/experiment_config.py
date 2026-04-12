"""Experiment-level configuration — combines model, data, training."""

from dataclasses import dataclass, field

from .model_config import ModelConfig
from .data_config import DataConfig
from .train_config import TrainConfig


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
