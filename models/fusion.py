"""λ-interpolation fusion of waveform-transformer head and GNN head.

Inspired by scGraPhT (Koç et al., IEEE TSIPN 2025) Eq. (1):
    P_final = λ · P_transformer + (1 − λ) · P_GNN

Here we operate on *logits* (linear combination preserves CE/focal loss
gradients better than mixing softmax probabilities). λ is a learnable
scalar squashed by sigmoid so it lives in (0, 1).

The auxiliary "transformer" head sees only the WaveformStem patch tokens
of the center epoch — it is a deliberately small, single-epoch baseline
that the GNN+sequence model is fused with.
"""

import torch
import torch.nn as nn

from ..configs.model_config import FusionConfig, WaveformStemConfig, HeadsConfig


class WaveformOnlyClassifier(nn.Module):
    """Mean-pool patch tokens → MLP → stage logits.

    No graph, no sequence — single-epoch transformer-style baseline.
    Produces ~6K params (negligible vs. full model budget).
    """

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, num_patches, embed_dim) center-epoch tokens
        Returns:
            logits: (B, num_classes)
        """
        pooled = patch_tokens.mean(dim=1)  # (B, embed_dim)
        return self.classifier(pooled)


class LambdaFusion(nn.Module):
    """Learnable scalar interpolation between two logit tensors.

        out = sigmoid(λ) · trans_logits + (1 − sigmoid(λ)) · gnn_logits
    """

    def __init__(self, init_lambda: float = 0.5):
        super().__init__()
        # Initialise the *unsquashed* parameter so that sigmoid(p) ≈ init_lambda
        init_lambda = max(min(init_lambda, 1.0 - 1e-6), 1e-6)
        init_logit = torch.logit(torch.tensor(init_lambda))
        self.lambda_param = nn.Parameter(init_logit)

    @property
    def lambda_value(self) -> torch.Tensor:
        return torch.sigmoid(self.lambda_param)

    def forward(
        self, trans_logits: torch.Tensor, gnn_logits: torch.Tensor,
    ) -> torch.Tensor:
        lam = self.lambda_value
        return lam * trans_logits + (1.0 - lam) * gnn_logits


def build_fusion(
    fusion_cfg: FusionConfig | None,
    waveform_cfg: WaveformStemConfig,
    heads_cfg: HeadsConfig,
) -> tuple[WaveformOnlyClassifier | None, LambdaFusion | None]:
    """Build the (transformer-only classifier, λ-fusion) pair.

    Returns (None, None) when `fusion_cfg is None` — used by the ablation
    runner to structurally disable the auxiliary head.
    """
    if fusion_cfg is None:
        return None, None
    classifier = WaveformOnlyClassifier(
        embed_dim=waveform_cfg.embed_dim,
        num_classes=heads_cfg.num_classes,
        dropout=fusion_cfg.transformer_dropout,
    )
    fusion = LambdaFusion(init_lambda=fusion_cfg.init_lambda)
    return classifier, fusion
