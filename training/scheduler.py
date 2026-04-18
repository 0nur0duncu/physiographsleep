"""Learning rate scheduler factory."""

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from ..configs.train_config import SchedulerConfig


def build_scheduler(
    optimizer: optim.Optimizer,
    config: SchedulerConfig,
) -> LRScheduler:
    """Create linear-warmup + cosine-annealing LR schedule.

    Phase 1 (warmup_epochs): LR linearly ramps from ~0 to peak.
    Phase 2 (remaining):     Cosine decay from peak to eta_min.

    Warmup prevents over-confident early updates that cause val_loss to
    diverge from val_MF1 under FocalLoss + label_smoothing.  Standard
    in SleepTransformer / AttnSleep / ViT literature.

    Falls back to plain cosine when warmup_epochs == 0.
    """
    warmup = config.warmup_epochs
    if warmup <= 0:
        return CosineAnnealingLR(
            optimizer,
            T_max=config.t_max,
            eta_min=config.eta_min,
        )

    warmup_sched = LinearLR(
        optimizer,
        start_factor=1e-2,   # begin at 1% of peak LR
        end_factor=1.0,
        total_iters=warmup,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=config.t_max - warmup,
        eta_min=config.eta_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup],
    )
