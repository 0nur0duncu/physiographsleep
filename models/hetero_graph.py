"""Heterogeneous Epoch Graph Encoder.

Takes patch tokens (6, D) and band tokens (5, D), constructs a 12-node
heterogeneous graph, processes through Edge-Aware Graph Transformer blocks,
and produces a single epoch embedding.

Hetero-aware readout:
  summary_node ⊕ patch-attentive-pool ⊕ band-attentive-pool → Linear → D
  This preserves modality-specific signal (morphology vs. spectral) instead
  of a single global pool that could dilute either side.
"""

import torch
import torch.nn as nn

from ..configs.model_config import HeteroGraphConfig
from .layers import GraphTransformerBlock
from ..data.graph_builder import (
    NUM_NODES, SUMMARY_OFFSET, PATCH_OFFSET, BAND_OFFSET,
    NUM_PATCH, NUM_BAND, batch_epoch_graphs,
)


class AttentiveReadout(nn.Module):
    """Learnable attentive pooling over a subset of graph nodes."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, batch_id: torch.Tensor, batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, D) node features (subset)
            batch_id: (N,) graph membership for those nodes
            batch_size: B

        Returns:
            pooled: (B, D)
        """
        weights = self.gate(x)     # (N, 1)
        weighted = x * weights     # (N, D)

        out = torch.zeros(batch_size, x.shape[1], device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch_id.unsqueeze(-1).expand_as(weighted), weighted)
        return out


class HeteroGraphEncoder(nn.Module):
    """Heterogeneous intra-epoch graph encoder."""

    def __init__(self, config: HeteroGraphConfig):
        super().__init__()
        self.config = config

        # Summary token (learnable)
        self.summary_token = nn.Parameter(torch.randn(1, config.node_dim) * 0.02)

        # Graph Transformer blocks with linearly-increasing DropPath rate
        # (stochastic depth schedule from timm — deeper layers drop more).
        dp_rates = torch.linspace(
            0.0, float(config.drop_path), steps=max(config.num_layers, 1),
        ).tolist()
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                drop_path=dp_rates[i],
            )
            for i in range(config.num_layers)
        ])

        # Modality-specific attentive readouts
        self.patch_readout = AttentiveReadout(config.hidden_dim)
        self.band_readout = AttentiveReadout(config.hidden_dim)

        # Final projection: concat(summary, patch_pool, band_pool) → epoch embedding
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.out_dim),
            nn.LayerNorm(config.out_dim),
            nn.GELU(),
        )

        # Pre-compute per-epoch patch/band node offsets for vectorized index gather.
        patch_local = torch.arange(NUM_PATCH, dtype=torch.long) + PATCH_OFFSET
        band_local = torch.arange(NUM_BAND, dtype=torch.long) + BAND_OFFSET
        self.register_buffer("_patch_local", patch_local, persistent=False)
        self.register_buffer("_band_local", band_local, persistent=False)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        band_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, 6, D)
            band_tokens: (B, 5, D)

        Returns:
            epoch_embedding: (B, out_dim)
        """
        B = patch_tokens.shape[0]
        device = patch_tokens.device

        # Build batched graph
        x, edge_index, edge_type, batch_id = batch_epoch_graphs(
            patch_tokens, band_tokens,
        )

        # Indices of summary nodes within the batched graph (one per epoch)
        summary_indices = torch.arange(B, device=device) * NUM_NODES + SUMMARY_OFFSET

        # Replace zero summary tokens with learned parameter (vectorized)
        x[summary_indices] = self.summary_token.expand(B, -1)

        num_nodes = x.shape[0]

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, edge_index, edge_type, num_nodes)

        # --- Hetero-aware readout ------------------------------------------
        # Batch-offset indices for patch / band nodes
        batch_offsets = torch.arange(B, device=device) * NUM_NODES  # (B,)

        patch_idx = (batch_offsets.unsqueeze(-1) + self._patch_local).reshape(-1)
        band_idx = (batch_offsets.unsqueeze(-1) + self._band_local).reshape(-1)

        patch_nodes = x[patch_idx]  # (B*6, D)
        band_nodes = x[band_idx]    # (B*5, D)

        patch_batch = torch.repeat_interleave(
            torch.arange(B, device=device), NUM_PATCH,
        )
        band_batch = torch.repeat_interleave(
            torch.arange(B, device=device), NUM_BAND,
        )

        summary_out = x[summary_indices]                              # (B, D)
        patch_pool = self.patch_readout(patch_nodes, patch_batch, B)  # (B, D)
        band_pool = self.band_readout(band_nodes, band_batch, B)      # (B, D)

        combined = torch.cat([summary_out, patch_pool, band_pool], dim=-1)  # (B, 3D)
        return self.projection(combined)

