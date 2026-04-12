"""Smoke tests — verify forward pass, shapes, and parameter counts."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physiographsleep.configs.model_config import (
    HeadsConfig,
    HeteroGraphConfig,
    ModelConfig,
    SequenceDecoderConfig,
    SpectralEncoderConfig,
    WaveformStemConfig,
)
from physiographsleep.models.heads import MultiTaskHeads
from physiographsleep.models.hetero_graph import HeteroGraphEncoder
from physiographsleep.models.physiographsleep import PhysioGraphSleep
from physiographsleep.models.sequence_decoder import SequenceTransitionDecoder
from physiographsleep.models.spectral_encoder import SpectralTokenEncoder
from physiographsleep.models.waveform_stem import WaveformStem


B, L, C, T = 4, 25, 1, 3000


def test_waveform_stem():
    cfg = WaveformStemConfig()
    model = WaveformStem(cfg)
    x = torch.randn(B, C, T)
    out = model(x)
    assert out.shape == (B, 6, 96), f"Expected (4,6,96), got {out.shape}"
    print(f"✓ WaveformStem: {out.shape}")


def test_spectral_encoder():
    cfg = SpectralEncoderConfig()
    model = SpectralTokenEncoder(cfg)
    x = torch.randn(B, 5, 42)
    out = model(x)
    assert out.shape == (B, 5, 96), f"Expected (4,5,96), got {out.shape}"
    print(f"✓ SpectralTokenEncoder: {out.shape}")


def test_hetero_graph():
    cfg = HeteroGraphConfig()
    model = HeteroGraphEncoder(cfg)
    patch = torch.randn(B, 6, 96)
    band = torch.randn(B, 5, 96)
    out = model(patch, band)
    assert out.shape == (B, 128), f"Expected (4,128), got {out.shape}"
    print(f"✓ HeteroGraphEncoder: {out.shape}")


def test_sequence_decoder():
    cfg = SequenceDecoderConfig()
    model = SequenceTransitionDecoder(cfg)
    x = torch.randn(B, L, 128)
    out = model(x)
    assert out.shape == (B, L, 160), f"Expected (4,25,160), got {out.shape}"
    print(f"✓ SequenceTransitionDecoder: {out.shape}")


def test_heads():
    cfg = HeadsConfig()
    model = MultiTaskHeads(cfg)
    x = torch.randn(B, 160)
    out = model(x)
    assert out["stage"].shape == (B, 5)
    assert out["boundary"].shape == (B,)
    assert out["prev"].shape == (B, 5)
    assert out["next"].shape == (B, 5)
    assert out["n1"].shape == (B,)
    print(f"✓ MultiTaskHeads: stage={out['stage'].shape}")


def test_full_model():
    cfg = ModelConfig()
    model = PhysioGraphSleep(cfg)
    signals = torch.randn(B, L, C, T)
    spectral = torch.randn(B, L, 5, 42)
    out = model(signals, spectral)
    assert out["stage"].shape == (B, 5)
    print(f"✓ Full model forward pass OK")

    counts = model.count_parameters()
    print("\n📊 Parameter counts:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")
    assert counts["total"] < 1_200_000, f"Too many params: {counts['total']:,}"
    print(f"\n✓ Total parameters: {counts['total']:,} (< 1.2M budget)")


if __name__ == "__main__":
    print("=" * 60)
    print("PhysioGraphSleep — Forward Pass Smoke Tests")
    print("=" * 60)
    test_waveform_stem()
    test_spectral_encoder()
    test_hetero_graph()
    test_sequence_decoder()
    test_heads()
    test_full_model()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
