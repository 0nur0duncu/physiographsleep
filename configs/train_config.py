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
    """Adaptive F1-based loss weight settings (SeriesSleepNet-inspired).

    Conservative defaults to prevent the loss-rebalancing in Stage C from
    degrading val MF1 (observed regression with K=10, gamma=2 in earlier runs).
    """

    warmup_epochs: int = 5
    K: float = 5.0
    gamma: float = 1.0


@dataclass
class CurriculumConfig:
    """3-stage curriculum training settings.

    NOTE: Stage B and Stage C are OPTIONAL. Empirically:
      - Stage B (encoder frozen, decoder fine-tune) yielded no F1_N1 gain.
      - Stage C (end-to-end + adaptive loss) regressed val MF1 in earlier
        runs because Stage A already trains decoder jointly. Enable only
        when you want to experiment with the SeriesSleepNet-style adaptive
        rebalancing.
    """

    # Stage A: joint encoder + decoder pretrain (with N1 boost)
    stage_a_epochs: int = 30
    stage_a_lr: float = 1e-3

    # Stage B (optional): freeze encoder, fine-tune decoder only
    enable_stage_b: bool = False
    stage_b_epochs: int = 15
    stage_b_lr: float = 1e-4

    # Stage C (optional): end-to-end fine-tune with adaptive loss
    enable_stage_c: bool = False
    stage_c_epochs: int = 20
    stage_c_lr: float = 5e-5


@dataclass
class TrainConfig:
    """Full training configuration."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    adaptive_loss: AdaptiveLossConfig = field(default_factory=AdaptiveLossConfig)

    # General
    patience: int = 10

    # Logging
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
