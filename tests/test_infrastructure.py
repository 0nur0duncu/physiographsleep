"""Tests for ablation / CV / pretraining infrastructure and Grad-CAM."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from physiographsleep.configs import ExperimentConfig
from physiographsleep.configs.model_config import FusionConfig
from physiographsleep.models.physiographsleep import PhysioGraphSleep
from physiographsleep.scripts.run_ablation import patch_config, PATHWAY_3LAYER
from physiographsleep.scripts.run_cv import _build_fold_splits
from physiographsleep.training.pretraining import (
    MaskedPatchPretrainer, PretrainConfig,
)


# ---------------------------------------------------------------------------
# Ablation config patching
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["A", "B", "C", "D", "E", "F"])
def test_ablation_patch_produces_valid_config(name):
    cfg = ExperimentConfig()
    name_out, patched = patch_config(name, cfg)
    assert name_out == name

    # Model must instantiate for every ablation
    model = PhysioGraphSleep(patched.model)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_ablation_pathway_3layer_used_in_d_e_f():
    for name in ["D", "E", "F"]:
        cfg = ExperimentConfig()
        _, patched = patch_config(name, cfg)
        assert patched.model.graph.edge_pathways == PATHWAY_3LAYER
        assert patched.model.graph.num_layers == 3


def test_ablation_fusion_only_in_e_and_f():
    e_cfg = ExperimentConfig()
    _, e_patched = patch_config("E", e_cfg)
    assert e_patched.model.fusion is not None

    f_cfg = ExperimentConfig()
    _, f_patched = patch_config("F", f_cfg)
    assert f_patched.model.fusion is not None
    assert f_patched.train.n1_mixup is not None


def test_ablation_a_b_c_have_fusion_disabled():
    for name in ["A", "B", "C"]:
        cfg = ExperimentConfig()
        _, patched = patch_config(name, cfg)
        assert patched.model.fusion is None
        assert patched.train.n1_mixup is None


# ---------------------------------------------------------------------------
# CV fold splitting
# ---------------------------------------------------------------------------
def test_cv_fold_splits_are_disjoint():
    subjects = [f"SC{400 + i:03d}" for i in range(20)]
    splits = _build_fold_splits(subjects, n_folds=20, seed=42)

    assert len(splits) == 20
    test_subjects_seen = set()
    for s in splits:
        # Disjoint within each fold
        assert not (set(s["train"]) & set(s["val"]))
        assert not (set(s["train"]) & set(s["test"]))
        assert not (set(s["val"]) & set(s["test"]))
        # Train > val + test (sanity)
        assert len(s["train"]) > len(s["val"]) + len(s["test"])
        test_subjects_seen.update(s["test"])

    # Across all folds, every subject should appear in test exactly once
    assert test_subjects_seen == set(subjects)


def test_cv_fold_splits_kfold():
    subjects = [f"SC{400 + i:03d}" for i in range(20)]
    splits = _build_fold_splits(subjects, n_folds=10, seed=0)
    assert len(splits) == 10
    # Each test fold has 2 subjects (20 / 10)
    for s in splits:
        assert len(s["test"]) == 2


# ---------------------------------------------------------------------------
# Pretraining harness — smoke test
# ---------------------------------------------------------------------------
def test_pretrainer_forward_smoke():
    cfg = ExperimentConfig()
    encoder = PhysioGraphSleep(cfg.model)
    pretrainer = MaskedPatchPretrainer(
        encoder=encoder,
        cfg=PretrainConfig(mask_ratio=0.3, contrastive_weight=0.0),
    )

    B, L = 2, cfg.data.seq_len
    C = cfg.data.num_input_channels
    T = cfg.data.epoch_duration * cfg.data.sampling_rate
    sig = torch.randn(B, L, C, T)
    spec = torch.randn(B, L, 5, 42)

    out = pretrainer(sig, spec)
    assert "recon_loss" in out
    assert "total" in out
    assert torch.isfinite(out["total"])
    out["total"].backward()


def test_pretrainer_with_contrastive():
    cfg = ExperimentConfig()
    encoder = PhysioGraphSleep(cfg.model)
    pretrainer = MaskedPatchPretrainer(
        encoder=encoder,
        cfg=PretrainConfig(mask_ratio=0.15, contrastive_weight=0.1),
    )

    B, L = 2, cfg.data.seq_len
    C = cfg.data.num_input_channels
    T = cfg.data.epoch_duration * cfg.data.sampling_rate
    sig = torch.randn(B, L, C, T)
    spec = torch.randn(B, L, 5, 42)

    out = pretrainer(sig, spec)
    assert "contrastive_loss" in out
    assert torch.isfinite(out["contrastive_loss"])


# ---------------------------------------------------------------------------
# Pretraining → supervised hand-off
# ---------------------------------------------------------------------------
def test_pretrained_weights_load_into_supervised_model():
    cfg = ExperimentConfig()
    encoder = PhysioGraphSleep(cfg.model)
    pretrainer = MaskedPatchPretrainer(encoder, PretrainConfig())

    # Run a single backward step to perturb weights
    B, L = 2, cfg.data.seq_len
    sig = torch.randn(B, L, cfg.data.num_input_channels,
                       cfg.data.epoch_duration * cfg.data.sampling_rate)
    spec = torch.randn(B, L, 5, 42)
    out = pretrainer(sig, spec)
    out["total"].backward()

    # New supervised model loads encoder state_dict
    fresh = PhysioGraphSleep(cfg.model)
    incompat = fresh.load_state_dict(encoder.state_dict(), strict=False)
    # Encoder modules should match exactly; only pretraining-specific
    # extras (recon_head, mask_token) live outside the encoder so are
    # absent from the encoder.state_dict() — incompat should be empty.
    assert len(incompat.missing_keys) == 0
    assert len(incompat.unexpected_keys) == 0
