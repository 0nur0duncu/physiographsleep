"""Sleep-EDF data loading — download, preprocess, and cache.

Handles MNE-based EDF reading, annotation parsing, epoch extraction,
bandpass filtering, and caching.
"""

from pathlib import Path
import warnings

import mne
import numpy as np
from tqdm import tqdm

from ..configs.data_config import DataConfig
from ..data.dataset import ANNOTATION_MAP, split_subjects, get_subject_ids
from ..data.download import ensure_dataset

# Suppress noisy warnings
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", message=".*Channels contain different.*")
warnings.filterwarnings("ignore", message=".*Highpass cutoff frequency.*")
warnings.filterwarnings("ignore", message=".*Lowpass cutoff frequency.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
warnings.filterwarnings("ignore", category=FutureWarning)


def _find_edf_dir(data_dir: Path) -> Path:
    """Find directory containing SC*.edf files (may be in sleep-cassette/)."""
    # Check data_dir directly
    if list(data_dir.glob("SC*-PSG.edf")):
        return data_dir
    # Check sleep-cassette subdirectory
    sc_dir = data_dir / "sleep-cassette"
    if sc_dir.exists() and list(sc_dir.glob("SC*-PSG.edf")):
        return sc_dir
    # Search recursively
    for edf in data_dir.rglob("SC*-PSG.edf"):
        return edf.parent
    return data_dir


def load_sleep_edf(config: DataConfig) -> dict[str, dict[str, np.ndarray]]:
    """Load and preprocess Sleep-EDF-20 data.

    Downloads dataset if not present. Returns cached data if available.

    Returns:
        dict with 'train', 'val', 'test' keys, each containing:
            'epochs': (N, C, T) float32
            'labels': (N,) int64
            'spectral': (N, 5, 42) float32  — pre-computed spectral features
    """
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache key includes channel + EOG state so EEG-only and EEG+EOG caches
    # never collide. Switching `use_eog` automatically forces re-extraction.
    eog_tag = "_eog" if config.use_eog else ""
    cache_file = cache_dir / (
        f"sleepedf20_ch{config.channel.replace(' ', '_')}{eog_tag}"
        f"_wt{config.wake_trim_minutes}.npz"
    )
    if cache_file.exists():
        return _load_from_cache(cache_file)

    # Ensure dataset is downloaded
    ensure_dataset(data_dir=config.data_dir, study="SC")

    # Load raw data
    subject_ids = get_subject_ids(config.num_subjects)
    splits = split_subjects(
        subject_ids,
        train_n=config.train_subjects,
        val_n=config.val_subjects,
        seed=config.seed,
    )

    data_dir = _find_edf_dir(Path(config.data_dir))
    print(f"Loading EDF files from: {data_dir}")
    needed_subjects = set(subject_ids)
    all_epochs, all_labels = _load_all_subjects(data_dir, config, needed_subjects)

    # Pre-compute spectral features
    from .spectral import SpectralFeatureExtractor
    spectral_ext = SpectralFeatureExtractor(config)

    # Split by subject
    result = {}
    for split_name, split_ids in splits.items():
        split_epochs = []
        split_labels = []
        for sid in split_ids:
            if sid in all_epochs:
                split_epochs.append(all_epochs[sid])
                split_labels.append(all_labels[sid])
        if split_epochs:
            epochs = np.concatenate(split_epochs, axis=0)
            labels = np.concatenate(split_labels, axis=0)
            # Compute spectral: extract from channel 0
            print(f"  {split_name}: computing spectral for {len(labels)} epochs...")
            spectral = spectral_ext.extract_batch(epochs[:, 0, :])
            print(f"  {split_name}: {len(labels)} epochs, spectral {spectral.shape}")
            result[split_name] = {
                "epochs": epochs,
                "labels": labels,
                "spectral": spectral.astype(np.float32),
            }

    _save_to_cache(cache_file, result)
    return result


def load_sleep_edf_per_subject(
    config: DataConfig,
) -> dict[str, dict[str, np.ndarray]]:
    """Load Sleep-EDF and return per-subject (epochs, labels, spectral).

    Used by the CV runner so each fold can build its own train/val/test
    split without re-running the EDF→epoch pipeline.

    Returns:
        {subject_id: {"epochs": (N,C,T), "labels": (N,), "spectral": (N,5,42)}}
    """
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    eog_tag = "_eog" if config.use_eog else ""
    cache_file = cache_dir / (
        f"sleepedf20_persubj_ch{config.channel.replace(' ', '_')}{eog_tag}"
        f"_wt{config.wake_trim_minutes}.npz"
    )
    if cache_file.exists():
        loaded = np.load(cache_file)
        result: dict[str, dict[str, np.ndarray]] = {}
        keys = sorted({k.split("__", 1)[0] for k in loaded.files})
        for sid in keys:
            entry = {
                "epochs": loaded[f"{sid}__epochs"],
                "labels": loaded[f"{sid}__labels"],
            }
            sk = f"{sid}__spectral"
            if sk in loaded.files:
                entry["spectral"] = loaded[sk]
            result[sid] = entry
        return result

    ensure_dataset(data_dir=config.data_dir, study="SC")
    subject_ids = get_subject_ids(config.num_subjects)
    data_dir = _find_edf_dir(Path(config.data_dir))
    needed = set(subject_ids)
    all_epochs, all_labels = _load_all_subjects(data_dir, config, needed)

    from .spectral import SpectralFeatureExtractor
    spectral_ext = SpectralFeatureExtractor(config)

    save_dict: dict[str, np.ndarray] = {}
    result = {}
    for sid in subject_ids:
        if sid not in all_epochs:
            continue
        epochs = all_epochs[sid]
        labels = all_labels[sid]
        spectral = spectral_ext.extract_batch(epochs[:, 0, :]).astype(np.float32)
        result[sid] = {
            "epochs": epochs,
            "labels": labels,
            "spectral": spectral,
        }
        save_dict[f"{sid}__epochs"] = epochs
        save_dict[f"{sid}__labels"] = labels
        save_dict[f"{sid}__spectral"] = spectral

    np.savez_compressed(cache_file, **save_dict)
    return result


def _load_all_subjects(
    data_dir: Path,
    config: DataConfig,
    needed_subjects: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load PSG+Hypnogram file pairs for specified subjects."""
    all_epochs = {}
    all_labels = {}

    psg_files = sorted(data_dir.glob("SC*-PSG.edf"))
    if not psg_files:
        raise FileNotFoundError(
            f"No SC*-PSG.edf files found in {data_dir}. "
            f"Check data_dir setting or run dataset download."
        )

    # Pre-filter to only needed subjects' recordings
    if needed_subjects:
        psg_files = [p for p in psg_files if p.name[:5] in needed_subjects]
    print(f"Loading {len(psg_files)} recordings for {len(needed_subjects or [])} subjects")

    for psg_path in tqdm(psg_files, desc="Loading subjects"):
        subject_id = psg_path.name[:5]  # e.g., SC400 (subject 0)

        # Match hypnogram by recording prefix (subject+night), e.g. SC4001
        recording_prefix = psg_path.name[:6]
        hyp_pattern = f"{recording_prefix}*-Hypnogram.edf"
        hyp_files = list(data_dir.glob(hyp_pattern))
        if not hyp_files:
            print(f"Warning: No hypnogram for {subject_id}, skipping")
            continue

        hyp_path = hyp_files[0]
        epochs, labels = _load_single_recording(psg_path, hyp_path, config)

        if epochs is not None:
            if subject_id in all_epochs:
                all_epochs[subject_id] = np.concatenate(
                    [all_epochs[subject_id], epochs], axis=0,
                )
                all_labels[subject_id] = np.concatenate(
                    [all_labels[subject_id], labels], axis=0,
                )
            else:
                all_epochs[subject_id] = epochs
                all_labels[subject_id] = labels

    return all_epochs, all_labels


def _load_single_recording(
    psg_path: Path,
    hyp_path: Path,
    config: DataConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load one PSG recording and its hypnogram."""
    try:
        raw = mne.io.read_raw_edf(str(psg_path), preload=True)
        annot = mne.read_annotations(str(hyp_path))
        raw.set_annotations(annot)

        # Pick channel
        channels = [config.channel]
        if config.use_eog and "EOG horizontal" in raw.ch_names:
            channels.append("EOG horizontal")
        raw.pick(channels)

        # Bandpass filter
        raw.filter(config.bandpass_low, config.bandpass_high, fir_design="firwin")

        # Create epochs from annotations
        events, event_id = mne.events_from_annotations(
            raw, event_id=ANNOTATION_MAP, chunk_duration=config.epoch_duration,
        )

        tmin = 0.0
        tmax = config.epoch_duration - 1.0 / config.sampling_rate

        mne_epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=tmin, tmax=tmax,
            baseline=None, preload=True,
        )

        data = mne_epochs.get_data()  # (N, C, T)
        labels = mne_epochs.events[:, -1]

        # Wake-trim (literature standard: keep ±30 min of W around sleep period)
        if config.wake_trim_minutes > 0:
            data, labels = _trim_wake(
                data, labels,
                wake_label=ANNOTATION_MAP["Sleep stage W"],
                trim_minutes=config.wake_trim_minutes,
                epoch_duration=config.epoch_duration,
            )
            if data is None or len(labels) == 0:
                return None, None

        # Normalize per epoch
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True) + 1e-8
        data = (data - mean) / std

        return data.astype(np.float32), labels.astype(np.int64)

    except Exception as e:
        print(f"Warning: Failed to load {psg_path.name}: {e}")
        return None, None


def _trim_wake(
    data: np.ndarray,
    labels: np.ndarray,
    wake_label: int,
    trim_minutes: int,
    epoch_duration: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Keep only `trim_minutes` of W before first and after last sleep epoch.

    This is the canonical Sleep-EDF preprocessing used by DeepSleepNet,
    TinySleepNet, AttnSleep, SleepTransformer, XSleepNet, etc. It removes
    long awake periods at the beginning/end of recordings, which would
    otherwise dominate (~70% W -> ~30% W) and inflate accuracy while
    suppressing macro-F1.
    """
    sleep_mask = labels != wake_label
    if not sleep_mask.any():
        return None, np.array([], dtype=labels.dtype)

    sleep_idx = np.where(sleep_mask)[0]
    first_sleep, last_sleep = sleep_idx[0], sleep_idx[-1]

    # epochs_per_minute = 60 / epoch_duration
    keep_pad = (trim_minutes * 60) // epoch_duration
    start = max(0, first_sleep - keep_pad)
    end = min(len(labels), last_sleep + keep_pad + 1)
    return data[start:end], labels[start:end]


def _save_to_cache(path: Path, data: dict) -> None:
    """Save split data to .npz cache."""
    save_dict = {}
    for split_name, split_data in data.items():
        save_dict[f"{split_name}_epochs"] = split_data["epochs"]
        save_dict[f"{split_name}_labels"] = split_data["labels"]
        if "spectral" in split_data:
            save_dict[f"{split_name}_spectral"] = split_data["spectral"]
    np.savez_compressed(path, **save_dict)


def _load_from_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load split data from .npz cache."""
    loaded = np.load(path)
    result = {}
    for split in ["train", "val", "test"]:
        key_epochs = f"{split}_epochs"
        key_labels = f"{split}_labels"
        key_spectral = f"{split}_spectral"
        if key_epochs in loaded:
            result[split] = {
                "epochs": loaded[key_epochs],
                "labels": loaded[key_labels],
            }
            if key_spectral in loaded:
                result[split]["spectral"] = loaded[key_spectral]
    return result
