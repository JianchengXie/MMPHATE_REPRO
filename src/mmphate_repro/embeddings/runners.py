from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import m_phate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def center_columns(X: np.ndarray) -> np.ndarray:
    """Column-center a 2D array (n_samples, n_features)."""
    return X - X.mean(axis=0, keepdims=True)


def load_npy_if_exists(path: Path) -> Optional[np.ndarray]:
    return np.load(path) if path.exists() else None


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


@dataclass(frozen=True)
class TSNEConfig:
    n_components: int = 3
    perplexity: float = 50.0
    n_iter: int = 2000
    random_state: int = 24
    learning_rate: str = "auto"
    init: str = "pca"
    verbose: int = 0


def embed_mmphate(
    trace_data: np.ndarray,
    *,
    n_components: int = 3,
    n_jobs: int = 1,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Run M_PHATE on a tensor of shape (ET, H, S).
    Returns an embedding of shape (ET*H, n_components).
    """
    if cache_path is not None:
        cached = load_npy_if_exists(cache_path)
        if cached is not None:
            return cached

    ET, H, _ = trace_data.shape
    op = m_phate.M_PHATE(n_jobs=n_jobs)
    op.set_params(n_components=n_components)

    emb = op.fit_transform(trace_data)

    # Some versions may return (ET, H, D); unify to (ET*H, D).
    if emb.ndim == 3 and emb.shape[:2] == (ET, H):
        emb = emb.reshape(ET * H, -1)

    if cache_path is not None:
        save_npy(cache_path, emb)
    return emb


def embed_pca(
    X: np.ndarray,
    *,
    n_components: int = 3,
    random_state: int = 24,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    PCA on a 2D matrix X of shape (N, S).
    """
    if cache_path is not None:
        cached = load_npy_if_exists(cache_path)
        if cached is not None:
            return cached

    emb = PCA(n_components=n_components, random_state=random_state).fit_transform(X)

    if cache_path is not None:
        save_npy(cache_path, emb)
    return emb


def embed_tsne(
    X: np.ndarray,
    *,
    cfg: TSNEConfig = TSNEConfig(),
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    t-SNE on a 2D matrix X of shape (N, S).
    """
    if cache_path is not None:
        cached = load_npy_if_exists(cache_path)
        if cached is not None:
            return cached

    emb = TSNE(
        n_components=cfg.n_components,
        perplexity=cfg.perplexity,
        learning_rate=cfg.learning_rate,
        init=cfg.init,
        n_iter=cfg.n_iter,
        random_state=cfg.random_state,
        verbose=cfg.verbose,
    ).fit_transform(X)

    if cache_path is not None:
        save_npy(cache_path, emb)
    return emb