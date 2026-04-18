"""Exponential Moving Average of model weights.

Used by SleepTransformer, XSleepNet, and most recent SOTA sleep staging
models to stabilise val metrics and gain ~0.01-0.02 MF1 over the raw
weights. Updates are decoupled from the optimizer and applied after each
optimizer step; evaluation uses the EMA weights via a lightweight
swap/unswap context.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    """Track an exponential moving average of model parameters.

    Uses the timm-style *adjusted* decay schedule: the effective decay
    starts at ~0 and ramps up to the target ``decay`` as training
    progresses, which avoids the "cold start" problem where EMA weights
    remain mostly random for the first epoch(s) and produce catastrophic
    first-epoch val metrics (e.g. MF1 ~0.04).

        d_eff(t) = min(decay, (1 + t) / (10 + t))

    At t=0  -> 0.1
    At t=100 -> 0.92
    At t=1000 -> 0.99
    Asymptotically converges to the target ``decay``.

    Args:
        model: The training model whose parameters are tracked.
        decay: EMA decay factor (0.999 = ~1000 step half-life).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        self.ema = deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad = False

    def _effective_decay(self) -> float:
        warmup = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.decay, warmup)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights in-place from the live model."""
        self.num_updates += 1
        d = self._effective_decay()
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    @contextmanager
    def swap_into(self, model: nn.Module):
        """Temporarily load EMA weights into `model` for evaluation."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema.state_dict(), strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)
