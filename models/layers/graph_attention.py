"""Edge-Aware Graph Transformer block for heterogeneous epoch graph.

Each block performs:
  1. Multi-head attention with edge-type bias
  2. Residual connection + LayerNorm
  3. Feed-forward network
  4. Residual connection + LayerNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_EDGE_TYPES = 4


class EdgeAwareAttention(nn.Module):
    """Multi-head attention with learnable edge-type bias."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Learnable bias per edge type per head
        self.edge_bias = nn.Embedding(NUM_EDGE_TYPES, num_heads)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, D) node features (all nodes in batch)
            edge_index: (2, E) source–target edge pairs
            edge_type: (E,) edge type indices
            num_nodes: total number of nodes N

        Returns:
            out: (N, D) updated node features
        """
        Q = self.q_proj(x)  # (N, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to multi-head
        Q = Q.view(-1, self.num_heads, self.head_dim)  # (N, H, d)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)

        src, tgt = edge_index  # (E,), (E,)

        # Compute attention scores for each edge
        q_src = Q[tgt]  # queries from target nodes
        k_tgt = K[src]  # keys from source nodes
        attn_scores = (q_src * k_tgt).sum(dim=-1) * self.scale  # (E, H)

        # Add edge-type bias
        bias = self.edge_bias(edge_type)  # (E, H)
        attn_scores = attn_scores + bias

        # Softmax per target node (scatter)
        attn_weights = self._edge_softmax(attn_scores, tgt, num_nodes)  # (E, H)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted aggregation of values
        v_src = V[src]  # (E, H, d)
        weighted = v_src * attn_weights.unsqueeze(-1)  # (E, H, d)

        # Scatter-add to target nodes
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim,
                          device=x.device, dtype=x.dtype)
        out.scatter_add_(0, tgt.unsqueeze(-1).unsqueeze(-1).expand_as(weighted), weighted)

        out = out.reshape(num_nodes, -1)  # (N, D)
        return self.out_proj(out)

    @staticmethod
    def _edge_softmax(
        scores: torch.Tensor,
        target: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute softmax of scores grouped by target node."""
        # Stability: subtract max per target
        max_vals = torch.full((num_nodes, scores.shape[1]), -1e9,
                              device=scores.device, dtype=scores.dtype)
        max_vals.scatter_reduce_(
            0, target.unsqueeze(-1).expand_as(scores), scores, reduce="amax",
        )
        scores = scores - max_vals[target]

        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(num_nodes, scores.shape[1],
                              device=scores.device, dtype=scores.dtype)
        sum_exp.scatter_add_(0, target.unsqueeze(-1).expand_as(exp_scores), exp_scores)

        return exp_scores / (sum_exp[target] + 1e-12)
