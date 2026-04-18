"""Learning rate scheduler factory."""

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from ..configs.train_config import SchedulerConfig


def build_scheduler(
    optimizer: optim.Optimizer,
    config: SchedulerConfig,
) -> LRScheduler:
    """Create CosineAnnealingLR (no warm restarts).

    Warm restarts (CosineAnnealingWarmRestarts) caused val_loss spikes
    mid-training because LR jumped back to max at epoch T_0. Plain cosine
    over total training length gives a smoother descent and matches
    SleepTransformer / AttnSleep / XSleepNet scheduling.
    """
    return CosineAnnealingLR(
        optimizer,
        T_max=config.t_max,
        eta_min=config.eta_min,
    )
