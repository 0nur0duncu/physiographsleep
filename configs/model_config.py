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
    """Heterogeneous Epoch Graph Encoder.

    `edge_pathways` (scGraPhT-inspired, IEEE TSIPN 2025 §III-D) restricts
    each layer to a subset of edge types so the network composes a
    pathway of homogeneous + heterogeneous subgraph layers instead of
    fully-fused multi-relational message passing in every layer.

    Edge type ids (see data/graph_builder.py):
      0 = patch↔patch (homo)   1 = band↔band (homo)
      2 = patch↔band (hetero)  3 = summary↔all

    Default 3-layer scGraPhT pathway:
        hetero-only → homo-only → all-edges-with-summary
    Set `edge_pathways=None` to use all edge types in every layer
    (only used by the ablation runner).
    """

    node_dim: int = 96
    hidden_dim: int = 96
    out_dim: int = 128
    num_heads: int = 6
    num_layers: int = 3
    dropout: float = 0.2
    # drop_path 0.1: validated default. 0.2 was tested April 2026 and
    # killed F1_N1 (signal too sparse through 3 graph layers).
    drop_path: float = 0.1
    num_patch_nodes: int = 6
    num_band_nodes: int = 5
    num_summary_nodes: int = 1
    edge_pathways: list[tuple[int, ...]] | None = field(
        default_factory=lambda: [(2,), (0, 1), (0, 1, 2, 3)]
    )


@dataclass
class FusionConfig:
    """λ-interpolation fusion of transformer (waveform-only) and GNN heads.

    scGraPhT (IEEE TSIPN 2025) Eq. (1):
        P_final = λ · P_transformer + (1 − λ) · P_GNN

    `λ` is a learnable scalar passed through sigmoid so it lives in (0, 1).
    `init_lambda=0.3` deliberately favours the GNN at start (the auxiliary
    transformer head is small and untrained), then λ adapts during training.
    """

    init_lambda: float = 0.3
    transformer_dropout: float = 0.3
    # When True, the auxiliary transformer head receives gradients only
    # via its own logits (does NOT corrupt GNN graph embedding). This is
    # the scGraPhT _EL_ semantics — embedding & logit fusion. Enable when
    # the val loss diverges between the two branches.
    detach_gnn_for_lambda: bool = True


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
    """Full PhysioGraphSleep model configuration.

    Set `fusion=None` to disable the auxiliary transformer head and use
    the GNN logits directly (used by the ablation runner).
    """

    waveform: WaveformStemConfig = field(default_factory=WaveformStemConfig)
    spectral: SpectralEncoderConfig = field(default_factory=SpectralEncoderConfig)
    graph: HeteroGraphConfig = field(default_factory=HeteroGraphConfig)
    decoder: SequenceDecoderConfig = field(default_factory=SequenceDecoderConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    fusion: FusionConfig | None = field(default_factory=FusionConfig)
