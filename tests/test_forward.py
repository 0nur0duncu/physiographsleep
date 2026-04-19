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


def test_sequence_decoder_mask_zeroes_padding():
    """Padded positions must be zeroed at decoder output (mask propagation)."""
    cfg = SequenceDecoderConfig()
    model = SequenceTransitionDecoder(cfg)
    model.eval()
    x = torch.randn(B, L, 128)
    mask = torch.ones(B, L)
    mask[1, :5] = 0.0   # left padding on sample 1
    mask[1, -3:] = 0.0  # right padding on sample 1
    with torch.no_grad():
        out = model(x, mask)
    padded = out[mask == 0]
    valid = out[mask == 1]
    assert padded.abs().max().item() < 1e-6, \
        f"padded outputs not zeroed: max|.|={padded.abs().max().item():.3e}"
    assert valid.abs().max().item() > 1e-3, \
        "valid outputs collapsed to zero"
    print("✓ SequenceTransitionDecoder mask zeroes padded positions")


def test_dataset_boundary_symmetry():
    """First/last epochs with stage transition must be labeled boundary=1."""
    import numpy as np
    from physiographsleep.configs.data_config import DataConfig
    from physiographsleep.data.dataset import SleepEDFDataset

    cfg = DataConfig()
    cfg.seq_len = 5
    # labels: W, W, N1, N1, N2  -> boundary at idx 1 (W->N1 next), 2 (W->N1 prev),
    #                              and idx 3 (N1->N2 next), idx 4 (N1->N2 prev)
    labels = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    epochs = np.zeros((len(labels), 1, cfg.epoch_samples), dtype=np.float32)
    ds = SleepEDFDataset(epochs, labels, config=cfg)
    boundaries = [int(ds[i]["boundary"].item()) for i in range(len(labels))]
    assert boundaries[1] == 1, f"idx=1 should be boundary, got {boundaries}"
    assert boundaries[3] == 1, f"idx=3 should be boundary, got {boundaries}"
    assert boundaries[0] == 0, f"idx=0 W->W should NOT be boundary, got {boundaries}"
    print(f"✓ Dataset boundary symmetric labels: {boundaries}")


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


# ---------------------------------------------------------------------------
# Two-channel (EEG + EOG) support — guarantees that flipping use_eog=True
# yields a functional forward pass and does not break the 1-channel path.
# ---------------------------------------------------------------------------
def test_waveform_stem_two_channel():
    cfg = WaveformStemConfig()
    cfg.in_channels = 2
    model = WaveformStem(cfg)
    x = torch.randn(B, 2, T)
    out = model(x)
    assert out.shape == (B, 6, 96), f"2ch stem: {out.shape}"
    # The 2-channel path must introduce more parameters than 1ch
    # (depthwise groups=2 → separate filter bank per channel).
    one_ch = WaveformStem(WaveformStemConfig())
    assert sum(p.numel() for p in model.parameters()) > sum(p.numel() for p in one_ch.parameters())
    print(f"✓ WaveformStem 2ch: {out.shape}")


def test_spectral_encoder_two_channel():
    from physiographsleep.configs.model_config import SpectralEncoderConfig
    cfg = SpectralEncoderConfig()
    cfg.features_per_band = 84  # 42 features × 2 channels
    model = SpectralTokenEncoder(cfg)
    x = torch.randn(B, 5, 84)
    out = model(x)
    assert out.shape == (B, 5, 96), f"2ch spectral: {out.shape}"
    print(f"✓ SpectralTokenEncoder 2ch: {out.shape}")


def test_full_model_two_channel_via_sync():
    """End-to-end 2ch forward pass using sync_channel_config()."""
    from physiographsleep.configs import ExperimentConfig, sync_channel_config
    cfg = ExperimentConfig()
    cfg.data.use_eog = True
    sync_channel_config(cfg)
    # After sync both waveform and spectral dims must match 2ch.
    assert cfg.model.waveform.in_channels == 2
    assert cfg.model.spectral.features_per_band == 84
    model = PhysioGraphSleep(cfg.model)
    signals = torch.randn(B, L, 2, T)
    spectral = torch.randn(B, L, 5, 84)
    out = model(signals, spectral)
    assert out["stage"].shape == (B, 5)
    print(f"✓ Full model 2ch forward OK (spectral={spectral.shape})")


def test_sync_channel_config_idempotent():
    """Calling sync_channel_config multiple times is safe."""
    from physiographsleep.configs import ExperimentConfig, sync_channel_config
    cfg = ExperimentConfig()
    # 1ch → defaults
    assert cfg.model.waveform.in_channels == 1
    assert cfg.model.spectral.features_per_band == 42
    # Flip on
    cfg.data.use_eog = True
    sync_channel_config(cfg)
    sync_channel_config(cfg)
    assert cfg.model.waveform.in_channels == 2
    assert cfg.model.spectral.features_per_band == 84
    # Flip off — must restore 1ch dims.
    cfg.data.use_eog = False
    sync_channel_config(cfg)
    assert cfg.model.waveform.in_channels == 1
    assert cfg.model.spectral.features_per_band == 42
    print("✓ sync_channel_config is idempotent and reversible")


def test_spectral_extractor_two_channel():
    """SpectralFeatureExtractor.extract_batch handles (B, C, T) inputs."""
    import numpy as np
    from physiographsleep.configs.data_config import DataConfig
    from physiographsleep.data.spectral import SpectralFeatureExtractor
    cfg = DataConfig()
    extractor = SpectralFeatureExtractor(cfg)
    # 1ch (B, T)
    out1 = extractor.extract_batch(np.random.randn(3, cfg.epoch_samples).astype(np.float32))
    assert out1.shape == (3, 5, 42)
    # 1ch (B, 1, T)
    out2 = extractor.extract_batch(np.random.randn(3, 1, cfg.epoch_samples).astype(np.float32))
    assert out2.shape == (3, 5, 42)
    # 2ch (B, 2, T)
    out3 = extractor.extract_batch(np.random.randn(3, 2, cfg.epoch_samples).astype(np.float32))
    assert out3.shape == (3, 5, 84), f"2ch extractor: {out3.shape}"
    print(f"✓ SpectralFeatureExtractor 1ch={out1.shape} 2ch={out3.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("PhysioGraphSleep — Forward Pass Smoke Tests")
    print("=" * 60)
    test_waveform_stem()
    test_spectral_encoder()
    test_hetero_graph()
    test_sequence_decoder()
    test_sequence_decoder_mask_zeroes_padding()
    test_dataset_boundary_symmetry()
    test_heads()
    test_full_model()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
