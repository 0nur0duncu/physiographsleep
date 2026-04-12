"""Visualization utilities — confusion matrices, hypnograms, embeddings."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_hypnogram(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Hypnogram",
) -> None:
    """Plot ground truth vs predicted hypnogram."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    epochs = np.arange(len(true_labels))

    for ax, labels, label_text in [
        (axes[0], true_labels, "Ground Truth"),
        (axes[1], pred_labels, "Predicted"),
    ]:
        ax.step(epochs, labels, where="mid", linewidth=0.7)
        ax.set_ylabel("Stage")
        ax.set_yticks(range(5))
        ax.set_yticklabels(STAGE_NAMES)
        ax.set_title(label_text)
        ax.invert_yaxis()

    axes[1].set_xlabel("Epoch")
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> None:
    """Plot training loss and validation metrics over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if "train_loss" in history:
        axes[0].plot(history["train_loss"])
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")

    if "val_acc" in history:
        axes[1].plot(history["val_acc"])
        axes[1].set_title("Validation Accuracy")
        axes[1].set_xlabel("Epoch")

    if "val_mf1" in history:
        axes[2].plot(history["val_mf1"])
        axes[2].set_title("Validation Macro-F1")
        axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
