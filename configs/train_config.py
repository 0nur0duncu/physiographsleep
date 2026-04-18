"""Training configuration."""

from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """AdamW optimizer settings."""

    lr: float = 1e-3
    weight_decay: float = 2e-2
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0


@dataclass
class SchedulerConfig:
    """Cosine annealing with warm restarts."""

    t_0: int = 30
    t_mult: int = 2
    eta_min: float = 1e-6


@dataclass
class LossConfig:
    """Multi-task loss weights and focal loss settings."""

    stage_weight: float = 1.0
    boundary_weight: float = 0.35
    prev_stage_weight: float = 0.20
    next_stage_weight: float = 0.20
    n1_aux_weight: float = 0.30
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1


@dataclass
class TrainConfig:
    """Single-phase joint training configuration.

    Empirically validated curriculum simplification (April 2026):
      - Two-stage encoder-then-decoder training (formerly Stage B) gave
        zero F1_N1 gain over 16 epochs while train loss kept dropping.
      - End-to-end fine-tune with adaptive loss reweighting (formerly
        Stage C) regressed val MF1 from 0.7599 to 0.7458.
      - Joint encoder + decoder + heads training with N1-boost sampling
        matches single-pass training in TinySleepNet / AttnSleep /
        SleepTransformer and is sufficient.
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    # Single training schedule
    epochs: int = 60
    lr: float = 1e-3
    n1_boost: float = 2.0
    patience: int = 12

    # Logging
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best.pt"
