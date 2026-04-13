"""Sleep-EDF-20 dataset loader with subject-wise splitting and sequence windowing."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..configs.data_config import DataConfig

# Sleep-EDF annotation mapping
ANNOTATION_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # merge S3+S4 → N3
    "Sleep stage R": 4,
}
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def get_subject_ids(num_subjects: int = 20) -> list[str]:
    """Return sorted subject IDs for Sleep-EDF-20 SC.

    File naming: SC4SSNE0-PSG.edf where SS=subject(00-19), N=night(1-2).
    Subject ID = first 5 chars = 'SC4SS', e.g. SC400, SC401, ..., SC419.
    """
    return [f"SC4{i:02d}" for i in range(num_subjects)]


def split_subjects(
    subject_ids: list[str],
    train_n: int = 14,
    val_n: int = 3,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Subject-wise random split with no overlap."""
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(subject_ids).tolist()
    return {
        "train": sorted(shuffled[:train_n]),
        "val": sorted(shuffled[train_n : train_n + val_n]),
        "test": sorted(shuffled[train_n + val_n :]),
    }


class SleepEDFDataset(Dataset):
    """Sleep-EDF-20 dataset with sequence windowing.

    Each sample is a sequence of `seq_len` consecutive epochs.
    The label corresponds to the center epoch.
    """

    def __init__(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        config: DataConfig,
        transform=None,
        spectral: np.ndarray | None = None,
    ):
        """
        Args:
            epochs: (N, C, T) — all epochs for a split. C=1 or 2, T=3000
            labels: (N,) — integer labels 0–4
            config: data configuration
            transform: optional augmentation callable
            spectral: (N, 5, 42) — pre-computed spectral features
        """
        self.epochs = epochs
        self.labels = labels
        self.spectral = spectral
        self.seq_len = config.seq_len
        self.half = config.seq_len // 2
        self.transform = transform
        self.num_epochs = len(labels)

    def __len__(self) -> int:
        return self.num_epochs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx - self.half
        end = idx + self.half + 1

        # Determine valid vs padding indices
        seq_indices = np.arange(start, end)
        valid_mask = (seq_indices >= 0) & (seq_indices < self.num_epochs)
        valid_indices = seq_indices[valid_mask]

        # Allocate sequence arrays
        L = self.seq_len
        C, T = self.epochs.shape[1], self.epochs.shape[2]
        seq_data = np.zeros((L, C, T), dtype=self.epochs.dtype)
        seq_labels = np.zeros(L, dtype=np.int64)
        mask = np.zeros(L, dtype=np.float32)

        # Vectorized copy of valid epochs
        valid_positions = np.where(valid_mask)[0]
        seq_data[valid_positions] = self.epochs[valid_indices]
        seq_labels[valid_positions] = self.labels[valid_indices]
        mask[valid_positions] = 1.0

        # Apply augmentation to entire sequence at once
        if self.transform is not None:
            seq_data[valid_positions] = self.transform.transform_sequence(
                seq_data[valid_positions]
            )

        center_label = self.labels[idx]

        # Boundary label: does center epoch differ from neighbors?
        is_boundary = 0
        if idx > 0 and self.labels[idx] != self.labels[idx - 1]:
            is_boundary = 1
        elif idx < self.num_epochs - 1 and self.labels[idx] != self.labels[idx + 1]:
            is_boundary = 1

        # Previous / next stage labels
        prev_label = self.labels[idx - 1] if idx > 0 else center_label
        next_label = self.labels[idx + 1] if idx < self.num_epochs - 1 else center_label

        # N1-vs-rest binary
        n1_label = 1 if center_label == 1 else 0

        result = {
            "signal": torch.from_numpy(seq_data).float(),         # (L, C, T)
            "label": torch.tensor(center_label, dtype=torch.long),
            "seq_labels": torch.from_numpy(seq_labels).long(),    # (L,)
            "mask": torch.from_numpy(mask).float(),               # (L,)
            "boundary": torch.tensor(is_boundary, dtype=torch.float),
            "prev_label": torch.tensor(prev_label, dtype=torch.long),
            "next_label": torch.tensor(next_label, dtype=torch.long),
            "n1_label": torch.tensor(n1_label, dtype=torch.float),
        }

        if self.spectral is not None:
            spec_data = np.zeros((L, self.spectral.shape[1], self.spectral.shape[2]),
                                 dtype=self.spectral.dtype)
            spec_data[valid_positions] = self.spectral[valid_indices]
            result["spectral"] = torch.from_numpy(spec_data).float()  # (L, 5, 42)

        return result
