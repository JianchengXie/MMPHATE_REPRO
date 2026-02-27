from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def zscore_across_samples(T: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score across samples axis (last axis).
    Input:  (ET, H, S)
    Output: (ET, H, S)
    """
    mean = T.mean(axis=2, keepdims=True)
    std = T.std(axis=2, ddof=1, keepdims=True)
    return (T - mean) / (std + eps)


def _neighbor_overlap_scores(source: np.ndarray, target: np.ndarray, k: int) -> list[float]:
    """
    For each point i, compute fraction of overlap between its k-NN sets
    in source-space vs target-space (excluding self).
    """
    n = source.shape[0]
    n_req = min(k + 1, n)  # +1 to include self then remove
    nn_s = NearestNeighbors(metric="euclidean").fit(source)
    nn_t = NearestNeighbors(metric="euclidean").fit(target)

    idx_s = nn_s.kneighbors(source, n_neighbors=n_req, return_distance=False)
    idx_t = nn_t.kneighbors(target, n_neighbors=n_req, return_distance=False)

    scores: list[float] = []
    for i in range(n):
        ns = [j for j in idx_s[i] if j != i][:k]
        nt = [j for j in idx_t[i] if j != i][:k]
        ke = min(len(ns), len(nt))
        if ke > 0:
            scores.append(len(set(ns[:ke]) & set(nt[:ke])) / float(ke))
    return scores


def neighborhood_preservation(T: np.ndarray, emb: np.ndarray, k: int = 10) -> tuple[float, float]:
    """
    Compute intra-step and inter-step neighbor preservation.

    T:   (ET, H, S)  z-scored traces in sample space
    emb: either (ET*H, D) or (ET, H, D)

    Returns:
      intra_mean: mean overlap within each timepoint across units
      inter_mean: mean overlap within each unit across timepoints
    """
    ET, H, _ = T.shape

    if emb.ndim == 2:
        emb = emb.reshape(ET, H, -1)

    intra_scores: list[float] = []
    for tau in range(ET):
        intra_scores.extend(_neighbor_overlap_scores(T[tau], emb[tau], k))

    inter_scores: list[float] = []
    for i in range(H):
        inter_scores.extend(_neighbor_overlap_scores(T[:, i, :], emb[:, i, :], k))

    intra_mean = float(np.mean(intra_scores)) if intra_scores else float("nan")
    inter_mean = float(np.mean(inter_scores)) if inter_scores else float("nan")
    return intra_mean, inter_mean