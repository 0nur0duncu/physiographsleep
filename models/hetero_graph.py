"""Heterogeneous Epoch Graph Encoder.

Takes patch tokens (6, D) and band tokens (5, D), constructs a 12-node
heterogeneous graph, processes through Edge-Aware Graph Transformer blocks,
and produces a single epoch embedding.

Readout: summary_token ⊕ attentive_pool → Linear → epoch embedding
"""

import torch
import torch.nn as nn

from ..configs.model_config import HeteroGraphConfig
from .layers import GraphTransformerBlock
from ..data.graph_builder import NUM_NODES, SUMMARY_OFFSET, batch_epoch_graphs


class AttentiveReadout(nn.Module):
    """Learnable attentive pooling over graph nodes."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, batch_id: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            x: (N, D) all node features
            batch_id: (N,) graph membership
            batch_size: B

        Returns:
            pooled: (B, D)
        """
        weights = self.gate(x)  # (N, 1)
        weighted = x * weights   # (N, D)

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

        # Graph Transformer blocks
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Attentive readout
        self.readout = AttentiveReadout(config.hidden_dim)

        # Final projection: concat(summary, pool) → epoch embedding
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.out_dim),
            nn.LayerNorm(config.out_dim),
            nn.GELU(),
        )

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
            epoch_embedding: (B, 128)
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

        # Readout: summary token + attentive pool
        summary_out = x[summary_indices]  # (B, D)

        pool_out = self.readout(x, batch_id, B)  # (B, D)

        combined = torch.cat([summary_out, pool_out], dim=-1)  # (B, 2D)
        return self.projection(combined)  # (B, out_dim)
