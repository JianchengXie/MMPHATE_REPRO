"""
One-command orchestrator for reproducing all paper figures.

Cache mode  (fast, ~minutes):
  Reads pre-computed embeddings from results_cache/ and regenerates figures.
  Requires: pre-populated results_cache/ for each dataset.

Full mode   (slow, ~hours):
  Generates synthetic data, trains models, computes embeddings, then figures.
  Requires: raw data files under data/ (see README for layout).
"""
from __future__ import annotations

import json
from pathlib import Path

from mmphate_repro.utils.paths import ensure_dirs, data_dir


def reproduce_main(mode: str, manifest_path: str, n_jobs: int = 1) -> None:
    ensure_dirs()
    cfg = json.loads(Path(manifest_path).read_text())

    _run_synthetic(mode, cfg.get("synthetic", {}), n_jobs)
    _run_area2bump(mode, cfg.get("area2bump", {}), n_jobs)
    _run_har(mode, cfg.get("har", {}), n_jobs)


# ---------------------------------------------------------------
# Synthetic suite
# ---------------------------------------------------------------

def _run_synthetic(mode: str, cfg: dict, n_jobs: int) -> None:
    scenarios = cfg.get("scenarios", [
        "hopf_bif__pitchfork_bif__no_warp__seed24",
        "hopf_bif__pitchfork_bif__warp__seed24",
        "hopf_bif__clean_vs_warped__seed24",
    ])

    if mode == "full":
        from mmphate_repro.pipelines.synthetic_generate import run_generate_suite
        from mmphate_repro.pipelines.synthetic_suite_embeddings import run_suite, SyntheticSuiteConfig

        syn_data = data_dir() / "synthetic"
        print("[synthetic] Generating ODE traces...")
        run_generate_suite(syn_data)

        emb_cfg = SyntheticSuiteConfig(n_jobs_mphate=n_jobs)
        print("[synthetic] Computing embeddings and metrics...")
        run_suite(syn_data, emb_cfg, include=tuple(scenarios))

    from mmphate_repro.figures.synthetic_grid_2d import make_grids
    print("[synthetic] Generating figures...")
    for p in make_grids(scenarios):
        print(f"  -> {p}")


# ---------------------------------------------------------------
# Area2Bump
# ---------------------------------------------------------------

def _run_area2bump(mode: str, cfg: dict, n_jobs: int) -> None:
    run_id = cfg.get("run_id", "paper_run")

    if mode == "full":
        from mmphate_repro.pipelines.area2bump_full import run_area2bump_full, Area2BumpRunConfig

        a2_cfg = Area2BumpRunConfig(
            epochs=cfg.get("epochs", 200),
            n_units=cfg.get("n_units", 20),
            lr=cfg.get("lr", 1e-4),
            n_jobs_mphate=n_jobs,
        )
        print(f"[area2bump] Training LSTM and computing embeddings (run_id={run_id})...")
        run_area2bump_full(data_dir() / "area2bump", run_id, a2_cfg)

    from mmphate_repro.figures.area2bump_grid_2d import make_area2bump_grid
    from mmphate_repro.figures.area2bump_grid_3d import make_area2bump_grid_3d
    print(f"[area2bump] Generating figures (run_id={run_id})...")
    print(f"  -> {make_area2bump_grid(run_id)}")
    for p in make_area2bump_grid_3d(run_id):
        print(f"  -> {p}")


# ---------------------------------------------------------------
# HAR
# ---------------------------------------------------------------

def _run_har(mode: str, cfg: dict, n_jobs: int) -> None:
    run_id = cfg.get("run_id", "paper_run")

    if mode == "full":
        from mmphate_repro.pipelines.har_full import run_har_full, HARRunConfig

        har_cfg = HARRunConfig(
            epochs=cfg.get("epochs", 1000),
            n_units=cfg.get("n_units", 30),
            lr=cfg.get("lr", 2e-5),
            n_jobs_mphate=n_jobs,
        )
        print(f"[har] Training LSTM and computing embeddings (run_id={run_id})...")
        run_har_full(data_dir() / "har", run_id, har_cfg)

    from mmphate_repro.figures.har_grid_2d import make_har_grid
    from mmphate_repro.figures.har_grid_3d import make_har_grid_3d
    print(f"[har] Generating figures (run_id={run_id})...")
    print(f"  -> {make_har_grid(run_id)}")
    for p in make_har_grid_3d(run_id):
        print(f"  -> {p}")
