"""Sleep-EDF dataset downloader via PhysioNet S3 (boto3)."""

from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

BUCKET = "physionet-open"
PREFIX = "sleep-edfx/1.0.0/"


def get_dataset_path(base_dir: str | None = None) -> Path:
    """Resolve dataset directory."""
    if base_dir:
        return Path(base_dir)
    return Path("physiographsleep/dataset/sleep-edfx")


def check_dataset_exists(data_dir: Path, study: str = "SC", min_files: int = 10) -> bool:
    """Check if enough EDF files already exist."""
    if not data_dir.exists():
        return False

    study_dir = data_dir / ("sleep-cassette" if study == "SC" else "sleep-telemetry")
    prefix = "SC" if study == "SC" else "ST"

    for search_dir in [study_dir, data_dir]:
        if not search_dir.exists():
            continue
        edf_files = [f for f in search_dir.glob(f"{prefix}*.edf") if "Hypnogram" not in f.name]
        hypno_files = list(search_dir.glob(f"{prefix}*Hypnogram.edf"))
        if len(edf_files) >= min_files and len(hypno_files) >= min_files:
            return True
    return False


def ensure_dataset(
    data_dir: str | None = None,
    study: str = "SC",
    force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """Download Sleep-EDF dataset if not already present.

    Args:
        data_dir: target directory
        study: 'SC' (sleep-cassette) or 'ST' (sleep-telemetry)
        force_download: re-download even if exists
        verbose: print progress

    Returns:
        Path to dataset directory
    """
    output_dir = get_dataset_path(data_dir)

    if not force_download and check_dataset_exists(output_dir, study):
        if verbose:
            print(f"✓ Dataset exists: {output_dir}")
        return output_dir

    if verbose:
        print(f"Downloading dataset to: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Only download the requested study subdirectory
    study_subdir = "sleep-cassette" if study == "SC" else "sleep-telemetry"
    study_prefix = f"{PREFIX}{study_subdir}/"

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET, Prefix=study_prefix)

    files = []
    for page in pages:
        for obj in page.get("Contents", []):
            files.append(obj)

    if verbose:
        print(f"Found {len(files)} files for {study_subdir}")

    iterator = tqdm(files, desc=f"Downloading {study_subdir}") if verbose else files

    for obj in iterator:
        key = obj["Key"]
        relative_path = key[len(PREFIX):]
        if not relative_path:
            continue

        local_path = output_dir / relative_path
        if local_path.exists():
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(local_path))

    if verbose:
        print(f"\n✓ Download complete: {output_dir}")

    return output_dir


if __name__ == "__main__":
    ensure_dataset(verbose=True)
