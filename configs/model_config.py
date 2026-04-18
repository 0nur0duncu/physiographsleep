"""Model architecture configuration."""

from dataclasses import dataclass, field


@dataclass
class WaveformStemConfig:
    """Waveform Stem — multi-scale depthwise-separable CNN."""

    in_channels: int = 1
    base_channels: int = 32
    embed_dim: int = 96
    num_patches: int = 6
    kernel_sizes: list[int] = field(default_factory=lambda: [25, 75, 225])
    initial_kernel: int = 25
    initial_stride: int = 2
    se_reduction: int = 4
    dropout: float = 0.1


@dataclass
class SpectralEncoderConfig:
    """Spectral Token Encoder — per-band MLP."""

    num_bands: int = 5
    features_per_band: int = 42  # 6 patches × 7 features/band
    hidden_dim: int = 64
    embed_dim: int = 96
    dropout: float = 0.1


@dataclass
class HeteroGraphConfig:
    """Heterogeneous Epoch Graph Encoder."""

    node_dim: int = 96
    hidden_dim: int = 96
    out_dim: int = 128
    num_heads: int = 6
    num_layers: int = 3
    dropout: float = 0.2
    drop_path: float = 0.1
    num_patch_nodes: int = 6
    num_band_nodes: int = 5
    num_summary_nodes: int = 1


@dataclass
class SequenceDecoderConfig:
    """Sequence Transition Decoder — BiGRU + Transition Memory."""

    input_dim: int = 128
    gru_hidden: int = 80
    gru_layers: int = 2
    gru_dropout: float = 0.3
    tcn_kernel: int = 3
    num_prototypes: int = 5  # one per sleep stage
    output_dim: int = 160
    dropout: float = 0.2


@dataclass
class HeadsConfig:
    """Multi-task classification heads."""

    input_dim: int = 160
    num_classes: int = 5
    dropout: float = 0.3


@dataclass
class ModelConfig:
    """Full PhysioGraphSleep model configuration."""

    waveform: WaveformStemConfig = field(default_factory=WaveformStemConfig)
    spectral: SpectralEncoderConfig = field(default_factory=SpectralEncoderConfig)
    graph: HeteroGraphConfig = field(default_factory=HeteroGraphConfig)
    decoder: SequenceDecoderConfig = field(default_factory=SequenceDecoderConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
