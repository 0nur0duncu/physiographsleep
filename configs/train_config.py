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
    # weight_decay 4e-2: increased from 2e-2 (April 2026) after
    # p1_f1_m1_waf1 showed 17.4 pp train-val MF1 generalization gap
    # (train 0.9541 / val 0.7801) + fitted T=1.087 via temperature
    # scaling — calibration was fine, divergence was true overfit.
    # 5e-2 previously caused a 6-point regression (under-fit); 4e-2
    # is the midpoint. Revert to 2e-2 if val MF1 drops >0.5 pp.
    # April 2026 overfit rerun: raised 4e-2 -> 8e-2 together with
    # decoder dropout 0.3/0.5 + prototype noise + aux-loss cut.
    weight_decay: float = 8e-2
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
    # Auxiliary head weights lowered April 2026 after overfit audit:
    # boundary/prev/next heads are random-init and their gradients
    # flow back into the shared encoder, consuming encoder capacity
    # on near-random targets and contributing to the 17 pp train/val
    # MF1 gap. Keeping them non-zero preserves the regularization
    # signal (boundary saliency, adjacency consistency) without the
    # previous 45 % of the loss budget going to auxiliary tasks.
    boundary_weight: float = 0.10
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
    # April 2026: overfit audit raised ls 0.05 -> 0.10 to blunt the
    # overconfident logits that drive train MF1 towards 0.97.
    label_smoothing: float = 0.10

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
    adaptive_warmup: int = 2    # epochs with uniform weights before adapting
    adaptive_K: float = 10.0    # coefficient for weight formula
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
