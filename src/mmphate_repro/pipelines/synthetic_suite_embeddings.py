from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mmphate_repro.embeddings.runners import center_columns, embed_mmphate, embed_pca, embed_tsne, TSNEConfig
from mmphate_repro.metrics.neighborhood import zscore_across_samples, neighborhood_preservation
from mmphate_repro.utils.paths import cache_dir
import fnmatch
import json

def _infer_E_T(z: dict, ET: int) -> tuple[int, int]:
    """
    Infer number of epochs E and timesteps per epoch T from npz metadata.
    Uses mu_a or mu_b length if available; otherwise falls back to (ET, 1).
    """
    E = None
    for k in ("n_epochs", "E", "epochs"):
        if k in z:
            try:
                E = int(z[k])
                break
            except Exception:
                pass

    if E is None:
        for k in ("mu_a", "mu_b"):
            if k in z:
                try:
                    E = int(len(z[k]))
                    break
                except Exception:
                    pass

    if E is None or E <= 0:
        return ET, 1

    if ET % E != 0:
        # fall back if inconsistent metadata
        return ET, 1

    T = ET // E
    return E, T


def _build_labels(ET: int, H: int, E: int, T: int, units_a: int) -> dict[str, np.ndarray]:
    """
    Build 1D label arrays of length N=ET*H for coloring.
    Assumes flattened axis is epoch-major then timestep (t=0..T-1 within each epoch).
    """
    epoch_ET = np.repeat(np.arange(E, dtype=int), T)          # length ET
    time_ET  = np.tile(np.arange(T, dtype=int), E)            # length ET

    epoch = np.repeat(epoch_ET, H)                            # length N
    timestep = np.repeat(time_ET, H)                          # length N
    unit = np.tile(np.arange(H, dtype=int), ET)               # length N
    group = (unit >= int(units_a)).astype(int)                # 0=A, 1=B
    return dict(epoch=epoch, timestep=timestep, unit=unit, group=group)


def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


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

    # IMPORTANT: define cache directory immediately
    out_dir = _scenario_cache_dir(npz_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Cache label vectors for plotting (epoch/timestep/unit/group)
    # ------------------------------------------------------------
    E, T = _infer_E_T(z, ET)

    epoch_per_ET = np.repeat(np.arange(E, dtype=int), T)[:ET]     # (ET,)
    step_per_ET  = np.tile(np.arange(T, dtype=int), E)[:ET]       # (ET,)

    epoch_label    = np.repeat(epoch_per_ET, H)                   # (ET*H,)
    timestep_label = np.repeat(step_per_ET, H)                    # (ET*H,)
    unit_label     = np.tile(np.arange(H, dtype=int), ET)         # (ET*H,)

    units_a = int(z["units_a"]) if "units_a" in z else H // 2
    units_b = int(z["units_b"]) if "units_b" in z else (H - units_a)
    group_per_unit = np.zeros(H, dtype=int)
    group_per_unit[units_a:] = 1                                  # 0=A, 1=B
    group_label = np.tile(group_per_unit, ET)                     # (ET*H,)

    meta = {
        "ET": int(ET), "H": int(H), "S": int(S),
        "E": int(E), "T": int(T),
        "units_a": int(units_a), "units_b": int(units_b),
        "npz_path": str(npz_path),
    }

    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels" / "epoch_label.npy", epoch_label)
    np.save(out_dir / "labels" / "timestep_label.npy", timestep_label)
    np.save(out_dir / "labels" / "unit_label.npy", unit_label)
    np.save(out_dir / "labels" / "group_label.npy", group_label)
    (out_dir / "labels" / "meta.json").write_text(json.dumps(meta, indent=2))

    # flat matrix for PCA / t-SNE
    X = trace_data.reshape(ET * H, S)
    Xc = center_columns(X)

    # embeddings (cached)
    mm = embed_mmphate(
        trace_data, n_components=3, n_jobs=cfg.n_jobs_mphate,
        cache_path=out_dir / "embedding_mmphate_3D.npy",
    )
    pca = embed_pca(
        Xc, n_components=3, random_state=24,
        cache_path=out_dir / "embedding_pca_3D.npy",
    )
    tsne = embed_tsne(
        Xc, cfg=cfg.tsne,
        cache_path=out_dir / "embedding_tsne_3D.npy",
    )

    # neighborhood preservation
    Tz = zscore_across_samples(trace_data.copy())
    rows = []
    for name, emb in [("MM-PHATE", mm), ("PCA", pca), ("t-SNE", tsne)]:
        for k in cfg.ks:
            intra, inter = neighborhood_preservation(Tz, emb, k=k)
            rows.append(dict(method=name, k=k, intra=intra, inter=inter))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "neighborhood_preservation.csv", index=False)

    # scenario info (light provenance)
    info_keys = ["system_a", "system_b", "units_a", "units_b", "warp_state", "warp_time"]
    info = {k: str(z[k]) for k in info_keys if k in z}
    if "mu_a" in z:
        info["mu_a_range"] = f"{float(z['mu_a'][0]):.2f} -> {float(z['mu_a'][-1]):.2f}"
    if "mu_b" in z:
        info["mu_b_range"] = f"{float(z['mu_b'][0]):.2f} -> {float(z['mu_b'][-1]):.2f}"
    pd.Series(info).to_csv(out_dir / "scenario_info.csv")

    return df


def run_suite(root_dir: Path, cfg: SyntheticSuiteConfig, include: tuple[str, ...] | None = None) -> pd.DataFrame:
    """
    Run embeddings/metrics for scenarios under root_dir.

    include:
      Optional list of scenario tags or glob patterns to include, e.g.
      ("hopf_bif__pitchfork_bif__no_warp", "hopf_bif__pitchfork_bif__warp", "hopf_big_clean_vs_warped")
      Patterns like "*pitchfork*" are allowed.
    """
    npz_files = sorted(root_dir.glob("**/rnn_traces.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No rnn_traces.npz found under {root_dir}")

    if include is not None and len(include) > 0:
        kept = []
        for f in npz_files:
            tag = _scenario_tag(f)
            if any(fnmatch.fnmatch(tag, pat) for pat in include):
                kept.append(f)
        npz_files = kept

        if not npz_files:
            raise FileNotFoundError(
                f"No scenarios matched include={include}. "
                f"Check folder names under {root_dir}."
            )

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