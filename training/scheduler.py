"""Learning rate scheduler factory."""

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler

from ..configs.train_config import SchedulerConfig


def build_scheduler(
    optimizer: optim.Optimizer,
    config: SchedulerConfig,
) -> LRScheduler:
    """Create CosineAnnealingWarmRestarts scheduler."""
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.t_0,
        T_mult=config.t_mult,
        eta_min=config.eta_min,
    )
