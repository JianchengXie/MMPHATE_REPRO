from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Area2BumpData:
    trainX: np.ndarray
    trainy: np.ndarray
    testX: np.ndarray
    testy: np.ndarray


def load_area2bump_from_dir(data_dir: Path) -> Area2BumpData:
    """
    Expects:
      trainX.npy, trainy.npy, testX.npy, testy.npy
    """
    trainX = np.load(data_dir / "trainX.npy")
    trainy = np.load(data_dir / "trainy.npy")
    testX  = np.load(data_dir / "testX.npy")
    testy  = np.load(data_dir / "testy.npy")
    return Area2BumpData(trainX=trainX, trainy=trainy, testX=testX, testy=testy)