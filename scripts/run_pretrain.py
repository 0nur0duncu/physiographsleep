"""Self-supervised pretraining driver for PhysioGraphSleep.

Trains the masked-patch reconstruction objective for N epochs and saves
the encoder weights to a `.pt` checkpoint that can be loaded into the
supervised `Trainer` via `model.load_state_dict(ckpt, strict=False)`.

Usage:
    python -m physiographsleep.scripts.run_pretrain --epochs 20 \
        --output checkpoints/pretrain.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Channels contain different.*")
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from physiographsleep.configs import ExperimentConfig
from physiographsleep.data.dataset import SleepEDFDataset
from physiographsleep.data.loader import load_sleep_edf
from physiographsleep.models.physiographsleep import PhysioGraphSleep
from physiographsleep.training.pretraining import (
    MaskedPatchPretrainer, PretrainConfig,
)
from physiographsleep.utils.reproducibility import get_device, set_seed


def _build_sequence_batch(
    batch: dict[str, torch.Tensor], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a SleepEDFDataset batch (B, C, T) to (B, L, C, T)."""
    sig = batch["signal"].to(device)
    spec = batch["spectral"].to(device)
    if sig.dim() == 3:        # single-epoch — wrap into seq of length 1
        sig = sig.unsqueeze(1)
        spec = spec.unsqueeze(1)
    return sig, spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="checkpoints/pretrain.pt",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    config = ExperimentConfig()
    config.data.batch_size = args.batch_size

    print("Loading data (uses standard split cache)...")
    data = load_sleep_edf(config.data)
    train_ds = SleepEDFDataset(
        data["train"]["epochs"], data["train"]["labels"],
        config=config.data, spectral=data["train"].get("spectral"),
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=config.data.num_workers, pin_memory=config.data.pin_memory,
    )

    encoder = PhysioGraphSleep(config.model).to(device)
    pretrainer = MaskedPatchPretrainer(
        encoder=encoder,
        cfg=PretrainConfig(
            mask_ratio=args.mask_ratio,
            contrastive_weight=args.contrastive_weight,
        ),
    ).to(device)

    optim = torch.optim.AdamW(pretrainer.parameters(), lr=args.lr)
    print(f"Pretraining {args.epochs} epochs on {len(train_ds)} samples")

    for ep in range(1, args.epochs + 1):
        pretrainer.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in loader:
            sig, spec = _build_sequence_batch(batch, device)
            losses = pretrainer(sig, spec)
            loss = losses["total"]
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 5.0)
            optim.step()
            total_loss += float(loss.item())
            n_batches += 1
        elapsed = time.time() - t0
        print(
            f"[ep {ep:02d}] recon_loss={total_loss / max(1, n_batches):.4f} "
            f"({elapsed:.0f}s)"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), out_path)
    print(f"Saved encoder weights → {out_path}")


if __name__ == "__main__":
    main()
