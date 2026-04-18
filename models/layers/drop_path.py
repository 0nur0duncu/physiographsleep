"""Stochastic Depth (DropPath) — drop entire residual branches during training.

Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
Standard regularizer for deep transformer stacks. Applied as a scaling on the
residual branch output; during eval it is an identity.
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop entire residual branch with probability `drop_prob`."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Broadcast mask over all dims except batch; we are at node level
        # (N, D) so treat dim 0 as batch-like.
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"
