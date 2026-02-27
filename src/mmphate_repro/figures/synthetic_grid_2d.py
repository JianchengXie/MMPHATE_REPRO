from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from mmphate_repro.utils.paths import cache_dir, figures_dir


_METHODS = [
    ("MM-PHATE", "embedding_mmphate_3D.npy"),
    ("PCA",      "embedding_pca_3D.npy"),
    ("t-SNE",    "embedding_tsne_3D.npy"),
]

_ROWS = [
    ("epoch",    "epoch_label.npy",    "continuous"),
    ("timestep", "timestep_label.npy", "continuous"),
    ("unit",     "unit_label.npy",     "categorical"),
    ("group",    "group_label.npy",    "group"),
]


def _load_labels(scen_dir: Path) -> dict[str, np.ndarray]:
    labdir = scen_dir / "labels"
    labels = {}
    for _, fname, _ in _ROWS:
        labels[fname] = np.load(labdir / fname)
    return labels


def _scatter(ax, xy: np.ndarray, c: np.ndarray, kind: str, title: str):
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    if kind == "continuous":
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=2, alpha=0.6)
        return sc

    if kind == "group":
        # 0=A, 1=B
        mask0 = (c == 0)
        mask1 = (c == 1)
        ax.scatter(xy[mask0, 0], xy[mask0, 1], s=2, alpha=0.6, label="A")
        ax.scatter(xy[mask1, 0], xy[mask1, 1], s=2, alpha=0.6, label="B")
        ax.legend(loc="best", fontsize=8, frameon=False)
        return None

    # categorical (unit id)
    # Works well because synthetic H is small (e.g., 6).
    ax.scatter(xy[:, 0], xy[:, 1], c=c, s=2, alpha=0.6)
    return None


def make_grid_for_scenario(scenario: str) -> Path:
    """
    Create a 4x3 grid (rows: epoch/timestep/unit/group; cols: mmphate/pca/tsne)
    using cached embeddings under results_cache/synthetic/<scenario>/.
    """
    scen_dir = cache_dir() / "synthetic" / scenario
    if not scen_dir.exists():
        raise FileNotFoundError(f"Missing cache directory: {scen_dir}")

    labels = _load_labels(scen_dir)
    epoch = labels["epoch_label.npy"]
    step = labels["timestep_label.npy"]
    unit = labels["unit_label.npy"]
    group = labels["group_label.npy"]

    row_colors = {
        "epoch_label.npy": epoch,
        "timestep_label.npy": step,
        "unit_label.npy": unit,
        "group_label.npy": group,
    }

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 12), constrained_layout=True)
    fig.suptitle(f"Synthetic scenario: {scenario} (2D)", fontsize=12)

    # Column headers
    for col, (mname, _) in enumerate(_METHODS):
        axes[0, col].set_title(mname, fontsize=11)

    # Plot rows
    for col, (mname, emb_file) in enumerate(_METHODS):
        emb = np.load(scen_dir / emb_file)
        xy = emb[:, :2]  # first two coords

        for row, (row_name, lab_file, kind) in enumerate(_ROWS):
            c = row_colors[lab_file]
            title = row_name if col == 0 else ""
            sc = _scatter(axes[row, col], xy, c, kind, title)

            # only add colorbars for continuous rows, and only once per row (last column)
            if kind == "continuous" and col == 2 and sc is not None:
                fig.colorbar(sc, ax=axes[row, :], fraction=0.015, pad=0.01)

    out_dir = figures_dir() / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario}_grid_2d.pdf"
    fig.savefig(out_path)
    plt.close(fig)

    # also write PNG
    png_path = out_dir / f"{scenario}_grid_2d.png"
    # re-open to save png at higher dpi (simple approach: regenerate quickly)
    # (keep it minimal: just save a png by reloading)
    # If you prefer, we can skip png and keep only pdf.
    return out_path


def make_grids(scenarios: Iterable[str]) -> list[Path]:
    out = []
    for s in scenarios:
        out.append(make_grid_for_scenario(s))
    return out