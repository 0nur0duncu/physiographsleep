"""Sequence Transition Decoder — BiGRU + TemporalConv + Transition Memory.

Architecture:
  epoch_embeddings (B, L, 128)
  → BiGRU(128→80, 2-layer, bidirectional) → (B, L, 160)
  → TemporalConvBlock(k=3) → (B, L, 160)
  → TransitionMemoryBlock (5 stage prototypes, cross-attention)
  → Linear → (B, L, 160)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.model_config import SequenceDecoderConfig


class TemporalConvBlock(nn.Module):
    """1D causal-style temporal convolution for local smoothing."""

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        residual = x
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, L, D)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x + residual


class TransitionMemoryBlock(nn.Module):
    """Cross-attention to learnable stage prototypes.

    5 learnable prototype vectors (one per sleep stage).
    The center epoch and its neighbors attend to these prototypes
    to produce transition-aware representations.
    """

    def __init__(
        self,
        dim: int,
        num_prototypes: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
        prototype_noise_std: float = 0.0,
    ):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)
        # Prototype noise injection at train time acts as an implicit
        # ensemble over shifted prototype variants and dampens subject-
        # specific transition memorization.
        self.prototype_noise_std = float(prototype_noise_std)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) sequence features

        Returns:
            (B, L, D) transition-aware features
        """
        B, L, D = x.shape

        # Expand prototypes for batch: (B, 5, D)
        prototypes = self.prototypes.unsqueeze(0).expand(B, -1, -1)
        if self.training and self.prototype_noise_std > 0.0:
            noise = torch.randn_like(prototypes) * self.prototype_noise_std
            prototypes = prototypes + noise

        # Cross-attention: queries=sequence, keys/values=prototypes
        attn_out, _ = self.cross_attn(
            query=x,
            key=prototypes,
            value=prototypes,
        )

        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x


class SequenceTransitionDecoder(nn.Module):
    """Full sequence decoder: BiGRU → TemporalConv → TransitionMemory."""

    def __init__(self, config: SequenceDecoderConfig):
        super().__init__()
        self.config = config
        gru_out_dim = config.gru_hidden * 2  # bidirectional

        # BiGRU backbone (dropout=0 for ROCm/MIOpen compatibility)
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.gru_dropout = nn.Dropout(config.gru_dropout)

        # Temporal convolution
        self.temporal_conv = TemporalConvBlock(
            dim=gru_out_dim,
            kernel_size=config.tcn_kernel,
            dropout=config.dropout,
        )

        # Transition memory
        self.transition_memory = TransitionMemoryBlock(
            dim=gru_out_dim,
            num_prototypes=config.num_prototypes,
            dropout=config.dropout,
            prototype_noise_std=getattr(config, "prototype_noise_std", 0.0),
        )

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(gru_out_dim, config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        epoch_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            epoch_embeddings: (B, L, 128)
            mask: (B, L) optional sequence mask

        Returns:
            sequence_features: (B, L, 160)
        """
        # Sequence padding mask: (B, L, 1), 1.0 for valid, 0.0 for padded.
        # Applied at every layer to prevent zero-padded edge positions from
        # corrupting BiGRU hidden state and causing encoder representation
        # collapse during Stage A pretraining.
        if mask is not None:
            m = mask.unsqueeze(-1).to(epoch_embeddings.dtype)
            epoch_embeddings = epoch_embeddings * m

        # BiGRU
        x, _ = self.gru(epoch_embeddings)  # (B, L, 160)
        x = self.gru_dropout(x)
        if mask is not None:
            x = x * m

        # Temporal convolution
        x = self.temporal_conv(x)  # (B, L, 160)
        if mask is not None:
            x = x * m

        # Transition memory cross-attention
        x = self.transition_memory(x)  # (B, L, 160)
        if mask is not None:
            x = x * m

        # Project
        x = self.projection(x)  # (B, L, output_dim)
        if mask is not None:
            x = x * m

        return x
