from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HARData:
    trainX: np.ndarray  # (7352, 128, 9)
    trainY: np.ndarray  # (7352, 128, 6)  one-hot, constant label per sequence
    testX: np.ndarray   # (2947, 128, 9)
    testY: np.ndarray   # (2947, 128, 6)


_EXPECTED = [
    "training_data_X.npy", "training_data_Y.npy",
    "testing_data_X.npy",  "testing_data_Y.npy",
]


def _extract_zip_if_needed(data_dir: Path) -> None:
    """Extract data.zip into data_dir if the raw .npy files are not present."""
    if all((data_dir / f).exists() for f in _EXPECTED):
        return
    zip_path = data_dir / "data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing HAR data in {data_dir}.\n"
            f"Expected raw files {_EXPECTED}, or a data.zip containing them."
        )
    print(f"[har] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)


def load_har_from_dir(data_dir: Path) -> HARData:
    """
    Expects training_data_X.npy, training_data_Y.npy,
    testing_data_X.npy, testing_data_Y.npy in data_dir,
    or a data.zip containing those files (auto-extracted on first use).
    """
    _extract_zip_if_needed(data_dir)
    return HARData(
        trainX=np.load(data_dir / "training_data_X.npy").astype(np.float32),
        trainY=np.load(data_dir / "training_data_Y.npy").astype(np.float32),
        testX=np.load(data_dir / "testing_data_X.npy").astype(np.float32),
        testY=np.load(data_dir / "testing_data_Y.npy").astype(np.float32),
    )
