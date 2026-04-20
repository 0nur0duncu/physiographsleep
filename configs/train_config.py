"""Training configuration."""

from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """AdamW optimizer settings."""

    lr: float = 1e-3
    # weight_decay 4e-2: increased from 2e-2 (April 2026) after
    # p1_f1_m1_waf1 showed 17.4 pp train-val MF1 generalization gap
    # (train 0.9541 / val 0.7801) + fitted T=1.087 via temperature
    # scaling — calibration was fine, divergence was true overfit.
    # 5e-2 previously caused a 6-point regression (under-fit); 4e-2
    # is the midpoint. Revert to 2e-2 if val MF1 drops >0.5 pp.
    weight_decay: float = 4e-2
    betas: tuple[float, float] = (0.9, 0.999)
    # grad_clip=5.0 matches AttnSleep/SleepTransformer; 1.0 was too
    # aggressive under AdamW + bfloat16 AMP (suppresses legitimate
    # gradient signal especially early in training).
    grad_clip: float = 5.0


@dataclass
class SchedulerConfig:
    """Linear warmup + cosine annealing.

    t_max should equal the total number of training epochs.
    The first `warmup_epochs` epochs linearly ramp LR from ~0 to peak;
    the remaining epochs follow a cosine decay to `eta_min`.

    Warmup prevents over-confident early updates that cause val_loss to
    diverge from val_MF1 (FocalLoss + label_smoothing artifact). Standard
    in SleepTransformer / AttnSleep / ViT literature.
    """

    # t_max 30: 20-epoch Sleep-EDF-20 runs (April 2026) show val MF1
    # plateauing after epoch ~13 (ΔMF1 < 0.005 per epoch). 60 epoch
    # cosine tail was 2× wasted compute with no measurable MF1 gain.
    t_max: int = 30
    eta_min: float = 1e-6
    warmup_epochs: int = 3


@dataclass
class LossConfig:
    """Multi-task loss weights and focal loss settings."""

    stage_weight: float = 1.0
    boundary_weight: float = 0.35
    prev_stage_weight: float = 0.20
    next_stage_weight: float = 0.20
    n1_aux_weight: float = 0.30
    # GNN branch deep-supervision weight (only active when fusion is ON).
    # Without this, detach_gnn_for_lambda=True cuts all stage-loss gradient
    # to the GNN branch, causing it to collapse.
    gnn_stage_weight: float = 0.5
    focal_gamma: float = 2.0
    # label_smoothing 0.0: val_loss drift (epoch 2'den sonra yükseliş)
    # FocalLoss(γ=2)'nin inherent overconfidence davranışından kaynaklanır.
    # ls=0.05 kullanıldığında drift daha büyük (1.11→1.53), ls=0.0 ile
    # daha küçük (0.64→0.92) ama tamamen yok olmuyor. Çünkü FocalLoss
    # zor örneklere odaklanırken kolay örneklerde logit büyütmeye devam
    # eder → cross-entropy bileşeni bunu cezalandırır.
    # Best-checkpoint MF1 ile seçildiğinden (loss değil) pratik sorun yok.
    # ls=0.05 ileride bir ablation olarak test edilebilir.
    label_smoothing: float = 0.0

    # --- Class weight strategy ---
    # "none"         : No class weights on focal loss (current default).
    # "inverse_freq" : Static 1/count weights (original approach).
    # "class_balanced": Cui et al. 2019 effective-number weights.
    # "adaptive_f1"  : SeriesSleepNet-style per-epoch F1-based dynamic
    #                  weights. First `adaptive_warmup` epochs use
    #                  uniform weights, then W_i = 1 - log_K(CF_i)^γ_a.
    weight_strategy: str = "none"

    # Adaptive F1-based weight hyperparams (SeriesSleepNet, Frontiers 2023)
    adaptive_warmup: int = 5    # epochs with uniform weights before adapting
    adaptive_K: float = 10.0    # log base for weight formula
    adaptive_gamma: float = 3.0  # exponent for weight formula

    # Class-balanced effective number (Cui et al. CVPR 2019)
    cb_beta: float = 0.9999     # β ∈ {0.9, 0.99, 0.999, 0.9999}


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
    # epochs 30: matches scheduler.t_max. Empirically val MF1 plateaus
    # around epoch 13-17 on Sleep-EDF-20 (April 2026); 60 epochs gave
    # no MF1 gain over 30 in multiple runs. Increase for larger datasets.
    epochs: int = 30
    lr: float = 1e-3
    # Sampler inverse-frequency already gives N1 ~7x boost vs N2.
    # Extra n1_boost multiplies on top; 2.0 -> effective 14x (too noisy,
    # caused F1_N1 oscillation). 1.3 is a mild reinforcement that keeps
    # N1 presence without drowning the batch in N1-like patterns.
    n1_boost: float = 1.3
    # Patience 5: reduced from 10 (April 2026). p1_f1_m1_waf1 showed
    # val MF1 peaked at epoch 11 (0.7871) and stayed flat-to-down
    # through epoch 21 while train loss kept dropping (0.79 → 0.19).
    # Temperature scaling confirmed over-fit, not calibration drift.
    # patience=5 stops within ~2-3 epochs of plateau onset. Increase
    # to 10 if you see an early false-plateau cycle.
    patience: int = 5
    # Exponential Moving Average of model weights. SleepTransformer /
    # XSleepNet standard: stabilises val metrics + adds ~0.01-0.02 MF1.
    ema_decay: float = 0.999

    # Logging
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best.pt"
