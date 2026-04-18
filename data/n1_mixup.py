"""N1-targeted Mixup augmentation.

For sleep staging the chief difficulty is the N1 class (PMC 9726872 reports
+19% N1 accuracy from a DCGAN augmentation; SSC-SleepNet 2025 closes most
of its N1 gap with adaptive focal + class-balanced sampling).

This module implements a *batch-level* Mixup that ONLY mixes samples whose
center label is N1 with a random *other-class* sample drawn from the same
batch. Compared to vanilla Mixup it:
  • preserves all non-N1 samples untouched,
  • soft-labels only the focal-loss head (the auxiliary heads keep the
    center sample's hard label — augmentation targets N1 boundary
    disambiguation, not multi-task targets).

Returns the mixed batch and an optional `(soft_targets, lam)` tuple that
the loss function should consume in place of the hard `label` target.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..configs.train_config import N1MixupConfig

# Beta distribution sampling without scipy dependency
def _sample_beta(alpha: float, device: torch.device) -> torch.Tensor:
    if alpha <= 0:
        return torch.tensor(1.0, device=device)
    a = torch.distributions.Gamma(alpha, 1.0).sample().to(device)
    b = torch.distributions.Gamma(alpha, 1.0).sample().to(device)
    return a / (a + b + 1e-12)


def apply_n1_mixup(
    batch: dict[str, torch.Tensor],
    cfg: N1MixupConfig,
    num_classes: int = 5,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Mix N1 samples with random other-class peers from the same batch.

    Args:
        batch: dict containing at least 'signal' (B, L, C, T),
            'spectral' (B, L, 5, 42), 'label' (B,)
        cfg: N1MixupConfig
        num_classes: number of stage classes (for soft-label one-hot)

    Returns:
        (possibly mixed) batch, and a dict with mixup info or None if
        no mixup was applied this call.
    """
    if not cfg.enabled:
        return batch, None

    labels = batch["label"]
    device = labels.device

    # Stochastic activation per batch
    if torch.rand(1, device=device).item() > cfg.prob:
        return batch, None

    n1_mask = labels == cfg.n1_class_id
    other_mask = ~n1_mask
    if n1_mask.sum().item() == 0 or other_mask.sum().item() == 0:
        return batch, None

    n1_idx = torch.nonzero(n1_mask, as_tuple=False).flatten()
    other_pool = torch.nonzero(other_mask, as_tuple=False).flatten()

    # Random partner from the other-class pool for each N1 sample
    perm = torch.randint(0, other_pool.numel(), (n1_idx.numel(),), device=device)
    partner_idx = other_pool[perm]

    lam = _sample_beta(cfg.alpha, device).clamp(0.05, 0.95)

    new_batch = dict(batch)  # shallow copy

    # Mix continuous tensors (signal + spectral). All are float.
    for key in ("signal", "spectral"):
        if key not in batch:
            continue
        a = batch[key][n1_idx]
        b = batch[key][partner_idx]
        mixed = lam * a + (1.0 - lam) * b
        out = batch[key].clone()
        out[n1_idx] = mixed
        new_batch[key] = out

    # Build soft labels for the focal/CE stage head
    soft = F.one_hot(labels, num_classes=num_classes).float()
    partner_labels = labels[partner_idx]
    soft_partner = F.one_hot(partner_labels, num_classes=num_classes).float()
    soft[n1_idx] = lam * soft[n1_idx] + (1.0 - lam) * soft_partner

    info = {
        "soft_label": soft,
        "lam": lam.detach(),
        "mixed_indices": n1_idx,
    }
    return new_batch, info
