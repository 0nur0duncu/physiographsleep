"""Training configuration."""

from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """Optimizer settings."""

    lr: float = 1e-3
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0


@dataclass
class SchedulerConfig:
    """Learning rate scheduler settings."""

    t_0: int = 30
    t_mult: int = 2
    eta_min: float = 1e-6


@dataclass
class LossConfig:
    """Multi-task loss weights and settings."""

    stage_weight: float = 1.0
    boundary_weight: float = 0.35
    prev_stage_weight: float = 0.20
    next_stage_weight: float = 0.20
    n1_aux_weight: float = 0.30
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05


@dataclass
class AdaptiveLossConfig:
    """Adaptive F1-based loss weight settings (SeriesSleepNet-inspired)."""

    warmup_epochs: int = 5
    K: float = 10.0
    gamma: float = 2.0


@dataclass
class CurriculumConfig:
    """3-stage curriculum training settings."""

    # Stage A: epoch encoder pretrain
    stage_a_epochs: int = 30
    stage_a_lr: float = 1e-3

    # Stage B: sequence decoder (encoder frozen)
    stage_b_epochs: int = 25
    stage_b_lr: float = 1e-4

    # Stage C: end-to-end fine-tune
    stage_c_epochs: int = 25
    stage_c_lr: float = 1e-4


@dataclass
class TrainConfig:
    """Full training configuration."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    adaptive_loss: AdaptiveLossConfig = field(default_factory=AdaptiveLossConfig)

    # General
    max_epochs: int = 80  # sum of curriculum stages (30+25+25)
    patience: int = 15
    val_interval: int = 1

    # Logging
    use_wandb: bool = False
    project_name: str = "physiographsleep"
    experiment_name: str = "default"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
