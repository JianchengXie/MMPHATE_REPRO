from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mmphate_repro.embeddings.runners import center_columns, embed_mmphate, embed_pca, embed_tsne, TSNEConfig
from mmphate_repro.metrics.neighborhood import zscore_across_samples, neighborhood_preservation
from mmphate_repro.utils.paths import cache_dir


@dataclass(frozen=True)
class SyntheticSuiteConfig:
    ks: tuple[int, ...] = (10, 40)
    tsne: TSNEConfig = TSNEConfig()
    n_jobs_mphate: int = 1


def _scenario_tag(npz_path: Path) -> str:
    return npz_path.parent.name


def _scenario_cache_dir(npz_path: Path) -> Path:
    # Central cache location (cleaner than writing into data folders)
    return cache_dir() / "synthetic" / _scenario_tag(npz_path)


def _load_npz(npz_path: Path) -> dict:
    return dict(np.load(npz_path, allow_pickle=True))


def run_one(npz_path: Path, cfg: SyntheticSuiteConfig) -> pd.DataFrame:
    z = _load_npz(npz_path)
    trace_data = z["data"]  # (ET, H, S)
    ET, H, S = trace_data.shape

    # Flatten for PCA/t-SNE
    X = trace_data.reshape(ET * H, S)
    Xc = center_columns(X)

    out_dir = _scenario_cache_dir(npz_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Embeddings (cached)
    mm = embed_mmphate(
        trace_data,
        n_components=3,
        n_jobs=cfg.n_jobs_mphate,
        cache_path=out_dir / "embedding_mmphate_3D.npy",
    )
    pca = embed_pca(Xc, n_components=3, random_state=24, cache_path=out_dir / "embedding_pca_3D.npy")
    tsne = embed_tsne(Xc, cfg=cfg.tsne, cache_path=out_dir / "embedding_tsne_3D.npy")

    # Neighborhood preservation on z-scored traces
    Tz = zscore_across_samples(trace_data.copy())
    rows = []
    for name, emb in [("MM-PHATE", mm), ("PCA", pca), ("t-SNE", tsne)]:
        for k in cfg.ks:
            intra, inter = neighborhood_preservation(Tz, emb, k=k)
            rows.append(dict(method=name, k=k, intra=intra, inter=inter))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "neighborhood_preservation.csv", index=False)

    # Scenario info (light provenance)
    info_keys = ["system_a", "system_b", "units_a", "units_b", "warp_state", "warp_time"]
    info = {k: str(z[k]) for k in info_keys if k in z}
    info["npz_path"] = str(npz_path)
    if "mu_a" in z:
        info["mu_a_range"] = f"{float(z['mu_a'][0]):.2f} -> {float(z['mu_a'][-1]):.2f}"
    if "mu_b" in z:
        info["mu_b_range"] = f"{float(z['mu_b'][0]):.2f} -> {float(z['mu_b'][-1]):.2f}"
    pd.Series(info).to_csv(out_dir / "scenario_info.csv")

    return df


def run_suite(root_dir: Path, cfg: SyntheticSuiteConfig) -> pd.DataFrame:
    npz_files = sorted(root_dir.glob("**/rnn_traces.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No rnn_traces.npz found under {root_dir}")

    all_dfs = []
    for f in npz_files:
        tag = _scenario_tag(f)
        print(f"[{tag}] {f}")
        df = run_one(f, cfg=cfg)
        df.insert(0, "scenario", tag)
        all_dfs.append(df)

    summary = pd.concat(all_dfs, ignore_index=True)

    suite_out = cache_dir() / "synthetic"
    suite_out.mkdir(parents=True, exist_ok=True)
    summary_path = suite_out / "all_neighborhood_preservation.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nSummary saved to {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str, help="Root directory containing scenario subfolders with rnn_traces.npz")
    ap.add_argument("--n_jobs", default=1, type=int, help="n_jobs for m_phate.M_PHATE")
    args = ap.parse_args()

    cfg = SyntheticSuiteConfig(n_jobs_mphate=args.n_jobs)
    run_suite(Path(args.root), cfg)


if __name__ == "__main__":
    main()