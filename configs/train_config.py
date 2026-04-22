"""Training configuration."""

from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """AdamW optimizer settings."""

    # lr 3e-4 (April 22 2026 gradual-learning fix): down from peak 1e-3.
    # Live run showed Val MF1 0.76 @epoch2 → 0.73 @epoch3 exactly when
    # warmup hit 6e-4 (still ramping). Train MF1 kept climbing to 0.81
    # while val froze → classic "LR too high for capacity" collapse for
    # a 584K model. 3e-4 is the AdamW sweet spot for this scale (see
    # Karpathy nanoGPT / LLaMA recipe) and lets the 10-epoch warmup end
    # at a value the regulariser stack (dropout, prototype_noise,
    # wd=8e-2, focal+ls) can still shape.
    lr: float = 3e-4
    # weight_decay 4e-2 (April 22 2026): back down from 8e-2. With
    # lr=3e-4 + warmup=10 the effective step-wise regularisation is
    # already much softer than the previous 1e-3 peak. 8e-2 together
    # with the lower LR was over-regularising: train loss components
    # were still dropping while val loss floored (ls=0.05 + focal γ=2
    # floor). 4e-2 lets the encoder refine representations in the
    # cosine decay tail without the heavy wd pulling weights back.
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

    # t_max 40 (April 22 2026): with lr=3e-4 and warmup=10 the cosine
    # decay tail starts at epoch 10 and has 30 epochs to anneal to
    # eta_min. Live run showed the model was improving monotonically
    # when early-stop killed it at epoch 5 — we simply didn't give it
    # enough runway. 40 is cheap on T4 (~80s/epoch = ~55 min).
    t_max: int = 40
    eta_min: float = 1e-6
    # warmup 10 (April 22 2026): up from 5. Live run diagnostic:
    # epoch 3 @ lr=6.04e-4 → ValMF1 0.76→0.73 collapse. Warmup=5 was
    # too aggressive — val broke DURING warmup. 10 epochs (25% of
    # budget) is the LLaMA / ViT standard; gradient landscape has time
    # to stabilise before LR passes the 2e-4 danger zone where this
    # model started overfitting.
    warmup_epochs: int = 10


@dataclass
class LossConfig:
    """Multi-task loss weights and focal loss settings."""

    stage_weight: float = 1.0
    # April 22 2026 — REVERTED aux-head zero experiment. The hypothesis
    # that prev/next heads were dead (loss 0.70 for 18 epochs) was
    # wrong: 0.70 is well BELOW random 5-class CE (1.61), i.e. the
    # heads had converged and WERE providing regularisation on the
    # shared encoder. Setting their weight to 0 on run jzaveo41 let
    # their loss drift back to 1.79 (random init) AND dropped test MF1
    # from 0.8230 to 0.8030. They were not dead, they were saturated
    # useful signals. Restored to 0.05 each. Boundary head drives N1
    # boundary disambiguation directly; ablation C bumps 0.10 -> 0.15
    # (baseline owxw7cdg @ 0.10 gave Test MF1 biased=0.8243, N1=0.5781).
    boundary_weight: float = 0.15
    prev_stage_weight: float = 0.05
    next_stage_weight: float = 0.05
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
    # April 22 2026: ls 0.10 -> 0.05. Live run showed val loss
    # locked at 0.55 from epoch 2 onwards — mathematically this was
    # the ls=0.10 floor for a 5-class focal loss (~0.365) plus the
    # frozen aux components. Val loss ceased to be a training signal
    # and became a calibration floor. Reducing ls to 0.05 halves the
    # floor and restores monotonic val-loss visibility; focal γ=2
    # already prevents over-confident logits without the full 0.10
    # smoothing.
    label_smoothing: float = 0.05

    # --- Class weight strategy ---
    # "none"         : No class weights on focal loss (current default).
    # "inverse_freq" : Static 1/count weights (original approach).
    # "class_balanced": Cui et al. 2019 effective-number weights.
    # "adaptive_f1"  : SeriesSleepNet-style per-epoch F1-based dynamic
    #                  weights. First `adaptive_warmup` epochs use
    #                  uniform weights, then W_i = 1 - log_K(CF_i)^γ_a.
    weight_strategy: str = "none"

    # Adaptive F1-based weight hyperparams.
    # adaptive_warmup 2 (April 2026): down from 5. Previously the
    # reweighting only activated for the last 1-2 epochs before early
    # stop, which had no measurable effect on Val N1 F1. With the
    # val-source fix (trainer uses val per_class_f1, not train) the
    # weights are safe to engage early — warmup=2 gives the EMA a
    # first checkpoint, then lets the minority boost kick in.
    # adaptive_warmup 3 (April 22 2026 — fourth revision): with EMA
    # smoothing now applied to the weight vector (decay=0.7), the hard
    # step-transition that previously required postponing activation
    # is gone. 3 epochs of uniform weights is enough to let LR warmup
    # stabilise gradient statistics before reweighting starts to ramp
    # in slowly.
    adaptive_warmup: int = 3    # epochs with uniform weights before adapting
    # adaptive_ema_decay 0.7: new weights = 0.7 * old + 0.3 * target.
    # A 2.26x target weight is reached in ~5 epochs instead of 1:
    #   e=1: 1.0  -> 0.7*1.0  + 0.3*2.26 = 1.38
    #   e=2: 1.38 -> 0.7*1.38 + 0.3*2.26 = 1.64
    #   e=3: 1.64 -> 0.7*1.64 + 0.3*2.26 = 1.83
    #   e=4:                              = 1.96
    #   e=5:                              = 2.05
    # Optimiser sees a slowly moving loss landscape; no step shock.
    adaptive_ema_decay: float = 0.7
    adaptive_K: float = 10.0    # coefficient for weight formula
    adaptive_gamma: float = 3.0  # exponent for weight formula
    # adaptive_K 10.0 (April 22 2026 — second revision): lowered from
    # 20.0. The original K=20 choice was compensating for the mean
    # normalisation in compute_adaptive_f1_weights which divided every
    # class weight by the batch mean — a bug that silently halved the
    # gradient signal on every non-minority class and caused the
    # observed ValMF1 regression on run jzaveo41 (0.7604 @ epoch 2 ->
    # 0.6913 @ epoch 3). With mean-normalisation removed and weights
    # clamped to [1.0, 5.0], K=10 now yields a clean profile: N1 at
    # F1=0.43 gets w=3.7x, W/N2 at F1=0.79 get w=1.2x, N3 at F1=0.86
    # gets w=1.05x. Every class keeps at least a full gradient while
    # the minority receives a genuine ~4x boost — no silent damping.

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

    # epochs 50: matches scheduler.t_max=40 + 10 epoch safety margin.
    # Early stopping (patience=10) will cut it short if val truly
    # plateaus; otherwise the extra budget lets cosine annealing
    # actually reach eta_min=1e-6 where fine refinement happens.
    epochs: int = 50
    # lr 3e-4 (April 22 2026): mirror OptimizerConfig.lr. trainer reads
    # from TrainConfig.lr.
    lr: float = 3e-4
    # Sampler inverse-frequency already gives N1 ~7x boost vs N2.
    # Extra n1_boost multiplies on top; 2.0 -> effective 14x (too noisy,
    # caused F1_N1 oscillation). 1.3 is a mild reinforcement that keeps
    # N1 presence without drowning the batch in N1-like patterns.
    # April 2026: unchanged (adaptive_f1 reweighting is the main N1
    # signal booster after val-source fix).
    n1_boost: float = 1.3
    # Patience 10 (April 22 2026): up from 3. Live run ground truth:
    # epoch 2 was best (0.7629), epochs 3-5 were a WARMUP GLITCH at
    # LR=6-10e-4, NOT a real plateau. patience=3 killed the run during
    # a transient LR shock. With lr=3e-4 + warmup=10 we want the model
    # to survive 8-10 noisy epochs before declaring plateau. EMA-based
    # best-checkpoint saving means we never lose a good weight.
    patience: int = 10
    # Exponential Moving Average of model weights. SleepTransformer /
    # XSleepNet standard: stabilises val metrics + adds ~0.01-0.02 MF1.
    ema_decay: float = 0.999

    # Logging
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best.pt"
