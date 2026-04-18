"""Graph Transformer block — attention + FFN with residual, norm, and DropPath."""

import torch
import torch.nn as nn

from .graph_attention import EdgeAwareAttention
from .drop_path import DropPath


class GraphTransformerBlock(nn.Module):
    """One layer of Edge-Aware Graph Transformer.

    attention → drop_path → residual + norm → FFN → drop_path → residual + norm
    DropPath (stochastic depth) improves generalization for deeper graph stacks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        ff_mult: int = 2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attention = EdgeAwareAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, D) node features
            edge_index: (2, E)
            edge_type: (E,)
            num_nodes: N
            edge_mask: optional (E,) bool — restrict this layer to a
                pathway-specific edge subset (scGraPhT §III-D).

        Returns:
            (N, D) updated features
        """
        # Multi-head attention + DropPath + residual
        x = x + self.drop_path(self.attention(
            self.norm1(x), edge_index, edge_type, num_nodes, edge_mask=edge_mask,
        ))
        # FFN + DropPath + residual
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
