"""Tests for scGraPhT-inspired pathway subgraphs + λ-fusion + N1-Mixup."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physiographsleep.configs.model_config import (
    FusionConfig,
    HeteroGraphConfig,
    ModelConfig,
)
from physiographsleep.configs.train_config import LossConfig, N1MixupConfig
from physiographsleep.data.n1_mixup import apply_n1_mixup
from physiographsleep.models.fusion import LambdaFusion, WaveformOnlyClassifier
from physiographsleep.models.hetero_graph import HeteroGraphEncoder
from physiographsleep.models.losses import MultiTaskLoss
from physiographsleep.models.physiographsleep import PhysioGraphSleep


B, L, C, T = 4, 25, 1, 3000


def test_pathway_hetero_graph_runs():
    """3-layer pathway: hetero → homo → all-with-summary (scGraPhT §III-D)."""
    cfg = HeteroGraphConfig(
        num_layers=3,
        edge_pathways=[(2,), (0, 1), (0, 1, 2, 3)],
    )
    model = HeteroGraphEncoder(cfg)
    patch = torch.randn(B, 6, 96)
    band = torch.randn(B, 5, 96)
    out = model(patch, band)
    assert out.shape == (B, 128)
    print(f"✓ Pathway HeteroGraphEncoder: {out.shape}")


def test_pathway_length_validation():
    """edge_pathways length must equal num_layers."""
    cfg = HeteroGraphConfig(num_layers=3, edge_pathways=[(2,)])
    try:
        HeteroGraphEncoder(cfg)
    except AssertionError:
        print("✓ edge_pathways length validation triggers")
        return
    raise AssertionError("expected AssertionError")


def test_lambda_fusion_init_and_value():
    f = LambdaFusion(init_lambda=0.5)
    assert abs(f.lambda_value.item() - 0.5) < 1e-6
    f = LambdaFusion(init_lambda=0.7)
    assert abs(f.lambda_value.item() - 0.7) < 1e-5
    print(f"✓ LambdaFusion init: λ={f.lambda_value.item():.3f}")


def test_lambda_fusion_logit_combination():
    f = LambdaFusion(init_lambda=0.25)
    a = torch.zeros(2, 5)
    b = torch.ones(2, 5)
    out = f(a, b)
    expected = (1.0 - 0.25) * 1.0  # λ·a + (1-λ)·b = 0 + 0.75
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-5)
    print(f"✓ LambdaFusion logit combo correct (λ=0.25, out={out[0, 0].item():.3f})")


def test_full_model_with_fusion():
    cfg = ModelConfig()
    cfg.fusion = FusionConfig(enabled=True, init_lambda=0.5)
    model = PhysioGraphSleep(cfg)
    signals = torch.randn(B, L, C, T)
    spectral = torch.randn(B, L, 5, 42)
    out = model(signals, spectral)
    assert out["stage"].shape == (B, 5)
    assert "stage_gnn" in out
    assert "stage_transformer" in out
    assert "lambda" in out
    counts = model.count_parameters()
    assert counts["fusion"] >= 1
    assert counts["transformer_classifier"] > 0
    assert counts["total"] < 1_300_000, f"too many params: {counts['total']:,}"
    print(
        f"✓ Full model + fusion forward OK (λ={out['lambda'].item():.3f}, "
        f"total={counts['total']:,})"
    )


def test_full_model_with_pathway_and_fusion():
    """Combined: pathway hetero+homo + λ-fusion (full scGraPhT-inspired stack)."""
    cfg = ModelConfig()
    cfg.graph = HeteroGraphConfig(
        num_layers=3, edge_pathways=[(2,), (0, 1), (0, 1, 2, 3)],
    )
    cfg.fusion = FusionConfig(enabled=True, init_lambda=0.5)
    model = PhysioGraphSleep(cfg)
    signals = torch.randn(B, L, C, T)
    spectral = torch.randn(B, L, 5, 42)
    out = model(signals, spectral)
    assert out["stage"].shape == (B, 5)
    print("✓ Full model + pathway + fusion forward OK")


def test_n1_mixup_disabled_is_noop():
    cfg = N1MixupConfig(enabled=False)
    batch = {
        "signal": torch.randn(8, 25, 1, 3000),
        "spectral": torch.randn(8, 25, 5, 42),
        "label": torch.tensor([0, 1, 1, 2, 3, 4, 0, 1]),
    }
    out, info = apply_n1_mixup(batch, cfg)
    assert info is None
    assert torch.equal(out["signal"], batch["signal"])
    print("✓ N1-Mixup no-op when disabled")


def test_n1_mixup_mixes_only_n1():
    cfg = N1MixupConfig(enabled=True, prob=1.0, alpha=0.4)
    torch.manual_seed(0)
    sig = torch.randn(8, 25, 1, 3000)
    labels = torch.tensor([0, 1, 1, 2, 3, 4, 0, 1])
    batch = {
        "signal": sig.clone(),
        "spectral": torch.randn(8, 25, 5, 42),
        "label": labels,
    }
    out, info = apply_n1_mixup(batch, cfg)
    assert info is not None
    n1_idx = (labels == 1).nonzero(as_tuple=False).flatten().tolist()
    other_idx = (labels != 1).nonzero(as_tuple=False).flatten().tolist()
    # N1 rows must have changed
    for i in n1_idx:
        assert not torch.equal(out["signal"][i], sig[i]), f"N1 idx {i} unchanged"
    # Non-N1 rows must NOT have changed
    for i in other_idx:
        assert torch.equal(out["signal"][i], sig[i]), f"non-N1 idx {i} changed"
    # Soft label rows for N1 should sum to 1
    soft = info["soft_label"]
    assert torch.allclose(soft.sum(dim=-1), torch.ones(8), atol=1e-5)
    print(f"✓ N1-Mixup mixes only N1 ({len(n1_idx)} N1 rows mixed, λ={info['lam']:.3f})")


def test_n1_mixup_skips_when_no_n1():
    cfg = N1MixupConfig(enabled=True, prob=1.0)
    batch = {
        "signal": torch.randn(4, 25, 1, 3000),
        "spectral": torch.randn(4, 25, 5, 42),
        "label": torch.tensor([0, 2, 3, 4]),  # no N1
    }
    out, info = apply_n1_mixup(batch, cfg)
    assert info is None
    print("✓ N1-Mixup skips batch with no N1 samples")


def test_focal_loss_soft_targets():
    """FocalLoss must produce a finite scalar for soft (B, K) targets."""
    cfg = LossConfig()
    loss_fn = MultiTaskLoss(cfg)
    torch.manual_seed(1)
    logits_stage = torch.randn(8, 5, requires_grad=True)
    soft = torch.softmax(torch.randn(8, 5), dim=-1)
    preds = {
        "stage": logits_stage,
        "boundary": torch.randn(8),
        "prev": torch.randn(8, 5),
        "next": torch.randn(8, 5),
        "n1": torch.randn(8),
    }
    targets = {
        "label": torch.randint(0, 5, (8,)),
        "label_soft": soft,
        "boundary": torch.randint(0, 2, (8,)).float(),
        "prev_label": torch.randint(0, 5, (8,)),
        "next_label": torch.randint(0, 5, (8,)),
        "n1_label": torch.randint(0, 2, (8,)).float(),
    }
    losses = loss_fn(preds, targets)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert logits_stage.grad is not None
    print(f"✓ FocalLoss soft-target path: total={losses['total'].item():.4f}")


if __name__ == "__main__":
    test_pathway_hetero_graph_runs()
    test_pathway_length_validation()
    test_lambda_fusion_init_and_value()
    test_lambda_fusion_logit_combination()
    test_full_model_with_fusion()
    test_full_model_with_pathway_and_fusion()
    test_n1_mixup_disabled_is_noop()
    test_n1_mixup_mixes_only_n1()
    test_n1_mixup_skips_when_no_n1()
    test_focal_loss_soft_targets()
    print("\n✓ All scGraPhT-inspired feature tests passed")
