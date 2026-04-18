"""Training configuration."""

from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """AdamW optimizer settings."""

    lr: float = 1e-3
    # weight_decay bumped 2e-2 → 5e-2 after April 2026 1ch run showed
    # val_loss diverging (0.856 → 0.907 over 4 epochs) while train kept
    # falling. AttnSleep / SleepTransformer use 1e-2–1e-1 range.
    weight_decay: float = 5e-2
    betas: tuple[float, float] = (0.9, 0.999)
    # grad_clip=5.0 matches AttnSleep/SleepTransformer; 1.0 was too
    # aggressive under AdamW + bfloat16 AMP (suppresses legitimate
    # gradient signal especially early in training).
    grad_clip: float = 5.0


@dataclass
class SchedulerConfig:
    """Plain cosine annealing (no restarts).

    t_max should equal the total number of training epochs so that LR
    smoothly decays from `lr` to `eta_min` across the full run.
    """

    t_max: int = 60
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
class N1MixupConfig:
    """N1-targeted Mixup augmentation (literature: PMC 9726872, +N1 acc).

    Mixes ONLY samples whose center label is N1 (class id=1) with a
    random other-class sample in the *same batch*. Mixup is performed
    on the raw waveform sequence + spectral features so all downstream
    encoders see the interpolated signal, while the soft label is fed
    to the focal loss via standard mixup label-blending.

    All other auxiliary heads (boundary/prev/next/n1_aux) keep the
    *center* sample's hard label — the augmentation is targeted at
    N1 boundary disambiguation only.

    Defaults are deliberately conservative (prob=0.2, alpha=0.2) so the
    training signal stays mostly clean; aggressive mixup destabilised
    F1_N1 in early-stop runs (April 2026 sweep).
    """

    prob: float = 0.2         # per-batch probability of triggering mixup
    alpha: float = 0.2        # Beta(α, α) — small α keeps λ near {0, 1}
    n1_class_id: int = 1


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
    # Set `n1_mixup=None` to disable the augmentation (used by ablation).
    n1_mixup: N1MixupConfig | None = field(default_factory=N1MixupConfig)

    # Single training schedule
    epochs: int = 60
    lr: float = 1e-3
    # Sampler inverse-frequency already gives N1 ~7x boost vs N2.
    # Extra n1_boost multiplies on top; 2.0 -> effective 14x (too noisy,
    # caused F1_N1 oscillation). 1.3 is a mild reinforcement that keeps
    # N1 presence without drowning the batch in N1-like patterns.
    n1_boost: float = 1.3
    # Patience tightened from 12 to 6: April 2026 runs showed val_loss
    # bottoming at epoch 4-5 then climbing for 3+ epochs while train
    # loss kept dropping (textbook overfit). 6 catches the inflection.
    patience: int = 6
    # Exponential Moving Average of model weights. SleepTransformer /
    # XSleepNet standard: stabilises val metrics + adds ~0.01-0.02 MF1.
    ema_decay: float = 0.999

    # Logging
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best.pt"
