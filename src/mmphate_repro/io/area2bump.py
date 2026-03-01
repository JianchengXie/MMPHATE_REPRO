from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Area2BumpData:
    trainX: np.ndarray
    trainy: np.ndarray
    testX: np.ndarray
    testy: np.ndarray


_EXPECTED = ["trainX.npy", "trainy.npy", "testX.npy", "testy.npy"]


def _extract_zip_if_needed(data_dir: Path) -> None:
    """Extract data.zip into data_dir if the raw .npy files are not present."""
    if all((data_dir / f).exists() for f in _EXPECTED):
        return
    zip_path = data_dir / "data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing area2bump data in {data_dir}.\n"
            f"Expected raw files {_EXPECTED}, or a data.zip containing them."
        )
    print(f"[area2bump] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)


def load_area2bump_from_dir(data_dir: Path) -> Area2BumpData:
    """
    Expects trainX.npy, trainy.npy, testX.npy, testy.npy in data_dir,
    or a data.zip containing those files (auto-extracted on first use).
    """
    _extract_zip_if_needed(data_dir)
    trainX = np.load(data_dir / "trainX.npy")
    trainy = np.load(data_dir / "trainy.npy")
    testX  = np.load(data_dir / "testX.npy")
    testy  = np.load(data_dir / "testy.npy")
    return Area2BumpData(trainX=trainX, trainy=trainy, testX=testX, testy=testy)