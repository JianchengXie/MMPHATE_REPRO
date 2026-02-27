from __future__ import annotations

from itertools import permutations
import numpy as np


def standardize_trace_tensor(raw: np.ndarray, *, n_samples: int, n_timesteps: int, n_units: int) -> np.ndarray:
    """
    Convert TraceHistory stack into a standardized tensor of shape (E, T, H, S).

    raw is expected to be 4D: (E, a, b, c) with some permutation of (S, T, H).

    We infer which axis corresponds to samples/timesteps/units by matching sizes.
    """
    if raw.ndim != 4:
        raise ValueError(f"Expected raw.ndim==4, got {raw.ndim} with shape {raw.shape}")

    E = raw.shape[0]
    dims = raw.shape[1:]

    # Find a permutation of axes (1,2,3) that matches (T,H,S)
    # raw[:, axT, axH, axS] => (E,T,H,S)
    target = (n_timesteps, n_units, n_samples)

    for perm in permutations(range(3), 3):
        cand = (dims[perm[0]], dims[perm[1]], dims[perm[2]])
        if cand == target:
            axT, axH, axS = perm
            out = raw.transpose(0, axT + 1, axH + 1, axS + 1)
            return out

    # Sometimes raw might match (H,T,S) or (S,T,H) etc. Try all permutations to match a set.
    # Fallback: match by unique dimension sizes
    idxS = [i for i, d in enumerate(dims) if d == n_samples]
    idxT = [i for i, d in enumerate(dims) if d == n_timesteps]
    idxH = [i for i, d in enumerate(dims) if d == n_units]
    if len(idxS) == 1 and len(idxT) == 1 and len(idxH) == 1:
        axS, axT, axH = idxS[0], idxT[0], idxH[0]
        out = raw.transpose(0, axT + 1, axH + 1, axS + 1)
        return out

    raise ValueError(
        "Could not infer (T,H,S) axes from raw trace tensor.\n"
        f"raw shape: {raw.shape}, expected sizes: n_samples={n_samples}, n_timesteps={n_timesteps}, n_units={n_units}"
    )