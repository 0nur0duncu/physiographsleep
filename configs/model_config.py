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

    Default 2-layer, no pathway restriction (April 2026):
        both layers see all edge types (patch-patch, band-band, patch-band,
        summary, self-loop).
    Previously used [(2,), (0, 1, 2, 3)] — restricted layer 1 to hetero-only
    edges. At 2-layer depth this starves patch-patch temporal and band-band
    spectral coupling in layer 1 with no chance to recover. Removing the
    restriction lets attention allocate capacity itself.
    GNN-specific bug fixed simultaneously: self-loops (EDGE_SELF=4) are now
    part of the static edge index so each node includes its own value in
    attention (was previously missing — see data/graph_builder.py).
    Change is channel-agnostic (1ch/2ch identical GNN — tokens flow from
    WaveformStem / SpectralEncoder which handle channel count upstream).
    Set `edge_pathways` explicitly (e.g. scGraPhT 3-layer pathway) only for
    ablation comparisons; self-loops are always preserved regardless.
    """

    node_dim: int = 96
    hidden_dim: int = 96
    out_dim: int = 128
    num_heads: int = 6
    num_layers: int = 2
    dropout: float = 0.2
    # drop_path 0.1: validated default. 0.2 was tested April 2026 and
    # killed F1_N1 (signal too sparse through graph layers).
    drop_path: float = 0.1
    num_patch_nodes: int = 6
    num_band_nodes: int = 5
    num_summary_nodes: int = 1
    edge_pathways: list[tuple[int, ...]] | None = None


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
    """Sequence Transition Decoder — BiGRU + Transition Memory.

    gru_layers=1 (April 2026): reduced from 2 after Sleep-EDF-20 runs
    showed val MF1 ceiling ~0.775 regardless of graph changes, while
    train MF1 climbed to 0.88 (10 pp gap). BiGRU probe contribution was
    +7.8 pp (dominant) and decoder held 424K of 710K params — so the
    overfit driver was here, not in the GNN. Single-layer BiGRU matches
    DeepSleepNet / AttnSleep practice and halves the sequence-level
    subject-memorization capacity without losing the transition cues
    TCN + TransitionMemory recover. Change is channel-agnostic.
    """

    input_dim: int = 128
    gru_hidden: int = 80
    gru_layers: int = 1
    # gru_dropout 0.5 (April 2026 overfit fix): raised from 0.3 after
    # physiographsleep.log showed 17 pp train/val MF1 gap (train 0.97 /
    # val 0.80) on Sleep-EDF-20. Single-layer BiGRU's internal
    # `dropout=` arg is inert (PyTorch applies it only between stacked
    # layers), so the *only* regularizer on the 130K-param GRU output
    # is this external nn.Dropout. 0.5 matches DeepSleepNet's BiLSTM
    # setting. Revert to 0.3 if Val MF1 drops > 1 pp.
    gru_dropout: float = 0.5
    tcn_kernel: int = 3
    num_prototypes: int = 5  # one per sleep stage
    # prototype_noise_std 0.05 (April 2026 overfit fix): Gaussian noise
    # added to the 5 learnable transition prototypes at train time. The
    # 5 * hidden_dim prototypes are a prime subject-memorization surface
    # (learnable nn.Parameter, no other regularizer). Random perturbation
    # during training acts as an ensemble over prototype slightly-shifted
    # variants; disabled at eval (self.training gate).
    prototype_noise_std: float = 0.05
    output_dim: int = 160
    # dropout 0.3 (April 2026 overfit fix): up from 0.2 for TemporalConv,
    # TransitionMemory cross-attn, and final projection.
    dropout: float = 0.3


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
