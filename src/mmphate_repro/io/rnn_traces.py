from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RNNTraces:
    """Container for rnn_traces.npz contents used in synthetic suite pipelines."""
    npz_path: Path
    data: np.ndarray  # (ET, H, S)
    meta: dict[str, Any]


def load_rnn_traces_npz(npz_path: str | Path) -> RNNTraces:
    """
    Load an `rnn_traces.npz` file produced by your synthetic suite.

    Expected keys:
      - "data": array shaped (ET, H, S)
    Additional keys are stored in `meta`.
    """
    npz_path = Path(npz_path)
    z = np.load(npz_path, allow_pickle=True)
    if "data" not in z:
        raise KeyError(f"{npz_path} missing required key 'data'.")

    data = z["data"]
    if data.ndim != 3:
        raise ValueError(f"{npz_path} expected data.ndim==3, got {data.ndim}.")

    meta: dict[str, Any] = {}
    for k in z.files:
        if k == "data":
            continue
        meta[k] = z[k]

    return RNNTraces(npz_path=npz_path, data=data, meta=meta)


def scenario_tag_from_npz(npz_path: str | Path) -> str:
    """Use parent folder name as scenario tag (matches your original script)."""
    npz_path = Path(npz_path)
    return npz_path.parent.name