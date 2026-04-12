"""Depthwise-separable convolution branch with SE attention."""

import torch
import torch.nn as nn

from .se_block import SqueezeExcitation


class DepthwiseSeparableConv(nn.Module):
    """Single depthwise-separable conv layer: DWConv → PWConv → activation."""

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, groups=channels, bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class DSConvBranch(nn.Module):
    """Two-layer depthwise-separable branch with SE block.

    DWConv → DWConv → SE → residual
    """

    def __init__(self, channels: int, kernel_size: int, se_reduction: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, kernel_size, dropout)
        self.conv2 = DepthwiseSeparableConv(channels, kernel_size, dropout)
        self.se = SqueezeExcitation(channels, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + residual
