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

    Args:
        model: The training model whose parameters are tracked.
        decay: EMA decay factor (0.999 = ~1000 step half-life).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights in-place from the live model."""
        d = self.decay
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
