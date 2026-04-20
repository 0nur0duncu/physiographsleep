"""Heterogeneous intra-epoch graph construction.

Builds a typed graph with 12 nodes per epoch:
  - 6 patch nodes (morphology tokens from WaveformStem)
  - 5 band nodes (spectral tokens from SpectralEncoder)
  - 1 summary node (global aggregation)

Edge types:
  1. patch↔patch: temporal adjacency between consecutive patches
  2. band↔band: coupling edges between all frequency bands
  3. patch↔band: cross-modal edges (each patch connects to all bands)
  4. summary↔all: summary node connects to every other node
"""

import torch

# Node type offsets (within a 12-node graph)
PATCH_OFFSET = 0       # nodes 0–5
BAND_OFFSET = 6        # nodes 6–10
SUMMARY_OFFSET = 11    # node 11

NUM_PATCH = 6
NUM_BAND = 5
NUM_NODES = 12

# Edge type identifiers
EDGE_PATCH_PATCH = 0
EDGE_BAND_BAND = 1
EDGE_PATCH_BAND = 2
EDGE_SUMMARY = 3
EDGE_SELF = 4  # self-loop — each node attends to its own value (fixes
               # missing self-attention term; April 2026 diagnosis)


def build_edge_index() -> tuple[torch.Tensor, torch.Tensor]:
    """Build static edge index and edge type tensors for one epoch.

    Returns:
        edge_index: (2, E) — source–target pairs
        edge_type: (E,) — integer edge type for each edge
    """
    sources, targets, types = [], [], []

    # 1. Patch↔patch temporal (chain: 0-1, 1-2, ..., 4-5) — bidirectional
    for i in range(NUM_PATCH - 1):
        s = PATCH_OFFSET + i
        t = PATCH_OFFSET + i + 1
        sources.extend([s, t])
        targets.extend([t, s])
        types.extend([EDGE_PATCH_PATCH, EDGE_PATCH_PATCH])

    # 2. Band↔band coupling (fully connected among bands) — bidirectional
    for i in range(NUM_BAND):
        for j in range(i + 1, NUM_BAND):
            s = BAND_OFFSET + i
            t = BAND_OFFSET + j
            sources.extend([s, t])
            targets.extend([t, s])
            types.extend([EDGE_BAND_BAND, EDGE_BAND_BAND])

    # 3. Patch↔band cross-modal (each patch connects to every band)
    for i in range(NUM_PATCH):
        for j in range(NUM_BAND):
            s = PATCH_OFFSET + i
            t = BAND_OFFSET + j
            sources.extend([s, t])
            targets.extend([t, s])
            types.extend([EDGE_PATCH_BAND, EDGE_PATCH_BAND])

    # 4. Summary↔all
    for i in range(NUM_NODES - 1):  # connect to nodes 0–10
        sources.extend([SUMMARY_OFFSET, i])
        targets.extend([i, SUMMARY_OFFSET])
        types.extend([EDGE_SUMMARY, EDGE_SUMMARY])

    # 5. Self-loops — every node attends to itself (standard GAT/Transformer
    # practice; without this the attention aggregation excludes the node's
    # own value and relies purely on the residual connection to propagate
    # self-information). Preserved through pathway masking unconditionally
    # (see hetero_graph.py).
    for i in range(NUM_NODES):
        sources.append(i)
        targets.append(i)
        types.append(EDGE_SELF)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_type = torch.tensor(types, dtype=torch.long)
    return edge_index, edge_type


# Pre-compute static graph topology (same for every epoch)
STATIC_EDGE_INDEX, STATIC_EDGE_TYPE = build_edge_index()


def batch_epoch_graphs(
    patch_tokens_batch: torch.Tensor,
    band_tokens_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch multiple epoch graphs with offset edge indices.

    Args:
        patch_tokens_batch: (B, 6, D)
        band_tokens_batch: (B, 5, D)

    Returns:
        x: (B*12, D) batched node features
        edge_index: (2, B*E) batched edge indices
        edge_type: (B*E,) batched edge types
        batch_id: (B*12,) graph membership
    """
    B = patch_tokens_batch.shape[0]
    device = patch_tokens_batch.device
    d = patch_tokens_batch.shape[-1]

    edge_index_base = STATIC_EDGE_INDEX.to(device)
    edge_type_base = STATIC_EDGE_TYPE.to(device)
    E = edge_index_base.shape[1]

    # Vectorized graph batching — no Python loop
    summary = torch.zeros(B, 1, d, device=device)
    x = torch.cat([
        patch_tokens_batch,           # (B, 6, D)
        band_tokens_batch,            # (B, 5, D)
        summary,                      # (B, 1, D)
    ], dim=1).reshape(B * NUM_NODES, d)  # (B*12, D)

    offsets = (torch.arange(B, device=device) * NUM_NODES).view(B, 1, 1)  # (B, 1, 1)
    edge_index = (edge_index_base.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, B * E)  # (2, B*E)
    edge_type = edge_type_base.unsqueeze(0).expand(B, -1).reshape(B * E)  # (B*E,)
    batch_id = torch.arange(B, device=device).unsqueeze(1).expand(B, NUM_NODES).reshape(B * NUM_NODES)  # (B*12,)

    return x, edge_index, edge_type, batch_id
