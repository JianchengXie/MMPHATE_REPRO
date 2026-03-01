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


def _scatter(ax, xy: np.ndarray, c: np.ndarray, kind: str):
    ax.set_xticks([])
    ax.set_yticks([])

    if kind == "continuous":
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=2, alpha=0.6)
        return sc

    if kind == "group":
        mask0 = (c == 0)
        mask1 = (c == 1)
        ax.scatter(xy[mask0, 0], xy[mask0, 1], s=2, alpha=0.6, label="A")
        ax.scatter(xy[mask1, 0], xy[mask1, 1], s=2, alpha=0.6, label="B")
        ax.legend(loc="best", fontsize=8, frameon=False)
        return None

    # categorical (unit id)
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
    row_colors = {
        "epoch_label.npy":    labels["epoch_label.npy"],
        "timestep_label.npy": labels["timestep_label.npy"],
        "unit_label.npy":     labels["unit_label.npy"],
        "group_label.npy":    labels["group_label.npy"],
    }

    # Load all embeddings up front
    embeddings = {}
    for mname, emb_file in _METHODS:
        embeddings[emb_file] = np.load(scen_dir / emb_file)[:, :2]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 12))
    fig.suptitle(f"Synthetic scenario: {scenario} (2D)", fontsize=13)

    # Track scatter mappables for colorbars on continuous rows
    continuous_mappables: dict[int, object] = {}

    for row, (row_name, lab_file, kind) in enumerate(_ROWS):
        c = row_colors[lab_file]
        # Row label on the left-most column
        axes[row, 0].set_ylabel(row_name, fontsize=11)

        for col, (mname, emb_file) in enumerate(_METHODS):
            ax = axes[row, col]
            xy = embeddings[emb_file]

            # Column header on the top row only
            if row == 0:
                ax.set_title(mname, fontsize=11)

            sc = _scatter(ax, xy, c, kind)

            if kind == "continuous" and sc is not None:
                continuous_mappables[row] = sc

    # Layout first, then add colorbars so they don't collapse axes
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    for row, sc in continuous_mappables.items():
        # Compute vertical span from the axes in this row
        bbox_top = axes[row, 0].get_position()
        bbox_bot = axes[row, 0].get_position()
        y0 = bbox_bot.y0
        y1 = bbox_top.y1
        cbar_ax = fig.add_axes([0.93, y0, 0.015, y1 - y0])
        fig.colorbar(sc, cax=cbar_ax)

    out_dir = figures_dir() / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save PDF
    # out_path = out_dir / f"{scenario}_grid_2d.pdf"
    # fig.savefig(out_path, dpi=150)

    # Save PNG
    png_path = out_dir / f"{scenario}_grid_2d.png"
    fig.savefig(png_path, dpi=200)

    plt.close(fig)
    return png_path


def make_grids(scenarios: Iterable[str]) -> list[Path]:
    out = []
    for s in scenarios:
        out.append(make_grid_for_scenario(s))
    return out