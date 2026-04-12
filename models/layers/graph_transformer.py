"""Graph Transformer block — attention + FFN with residual and norm."""

import torch
import torch.nn as nn

from .graph_attention import EdgeAwareAttention


class GraphTransformerBlock(nn.Module):
    """One layer of Edge-Aware Graph Transformer.

    attention → residual + norm → FFN → residual + norm
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.2, ff_mult: int = 2):
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, D) node features
            edge_index: (2, E)
            edge_type: (E,)
            num_nodes: N

        Returns:
            (N, D) updated features
        """
        # Multi-head attention + residual
        x = x + self.attention(self.norm1(x), edge_index, edge_type, num_nodes)
        # FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x
