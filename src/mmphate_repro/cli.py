import argparse
from pathlib import Path

from mmphate_repro.pipelines.reproduce import reproduce_main
from mmphate_repro.pipelines.synthetic_suite_embeddings import run_suite, SyntheticSuiteConfig
from mmphate_repro.figures.synthetic_grid_2d import make_grids
from mmphate_repro.figures.area2bump_grid_2d import make_area2bump_grid
from mmphate_repro.figures.area2bump_grid_3d import make_area2bump_grid_3d


def main():
    parser = argparse.ArgumentParser(prog="mmphate-repro")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -----------------------
    # reproduce (figure manifest)
    # -----------------------
    p_rep = sub.add_parser("reproduce", help="Reproduce all paper figures (cache or full mode)")
    p_rep.add_argument("--mode", choices=["cache", "full"], required=True,
                       help="cache: figures from pre-computed embeddings; full: train+embed+plot")
    p_rep.add_argument("--config", default="configs/figure_manifest.json",
                       help="Path to reproduce config JSON (default: configs/figure_manifest.json)")
    p_rep.add_argument("--n_jobs", type=int, default=1,
                       help="Parallel jobs for MM-PHATE (speeds up full mode on multi-core)")

    # -----------------------
    # synthetic-generate (generate rnn_traces.npz from ODE simulations)
    # -----------------------
    p_sgen = sub.add_parser(
        "synthetic-generate",
        help="Generate synthetic rnn_traces.npz files (Hopf/Pitchfork bifurcation suite)",
    )
    p_sgen.add_argument(
        "--out_dir",
        default=None,
        help="Root output directory (default: data/synthetic/)",
    )
    p_sgen.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario names to generate (default: all). See SCENARIO_NAMES in synthetic_generate.py.",
    )
    p_sgen.add_argument("--epochs",   type=int,   default=101)
    p_sgen.add_argument("--timesteps", type=int,  default=80)
    p_sgen.add_argument("--samples",  type=int,   default=10)
    p_sgen.add_argument("--seed",     type=int,   default=24)

    # -----------------------
    # synthetic-suite (embeddings + neighborhood metrics)
    # -----------------------
    p_syn = sub.add_parser("synthetic-suite", help="Run synthetic suite embeddings + metrics")
    p_syn.add_argument(
        "--root",
        required=True,
        help="Root directory containing scenario folders with rnn_traces.npz",
    )
    p_syn.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="n_jobs for m_phate.M_PHATE",
    )
    p_syn.add_argument(
        "--include",
        nargs="+",
        default=[
            "hopf_bif__pitchfork_bif__no_warp__seed24",
            "hopf_bif__pitchfork_bif__warp__seed24",
            "hopf_bif__clean_vs_warped__seed24",
        ],
        help="Scenario tags or glob patterns to include. Default: main-text scenarios only.",
    )

    # -----------------------
    # synthetic-figures (embeddings + neighborhood metrics)
    # -----------------------
    p_sfig = sub.add_parser("synthetic-figures", help="Generate synthetic 2D grid figures from cache")
    p_sfig.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        help="Scenario folder names (must exist under results_cache/synthetic/<scenario>/)",
    )

    # Area2Bump full pipeline
    p_a2 = sub.add_parser("area2bump-run", help="Train Area2Bump LSTM, trace, embed, and cache artifacts")
    p_a2.add_argument("--run_id", required=True, help="Name for results_cache/area2bump/<run_id>/")
    p_a2.add_argument("--data_root", default=None, help="Directory containing trainX/trainy/testX/testy npy files")
    p_a2.add_argument("--epochs", type=int, default=200)
    p_a2.add_argument("--units", type=int, default=20)
    p_a2.add_argument("--seed", type=int, default=42)
    p_a2.add_argument("--lr", type=float, default=1e-4)
    p_a2.add_argument("--n_jobs", type=int, default=1)

    # -----------------------
    # har-run (train LSTM, trace, embed, cache)
    # -----------------------
    p_har = sub.add_parser("har-run", help="Train HAR LSTM, trace, embed, and cache artifacts")
    p_har.add_argument("--run_id", required=True, help="Name for results_cache/har/<run_id>/")
    p_har.add_argument("--data_root", default=None, help="Directory containing training/testing_data_X/Y.npy")
    p_har.add_argument("--epochs", type=int, default=1000)
    p_har.add_argument("--units",  type=int, default=30)
    p_har.add_argument("--seed",   type=int, default=42)
    p_har.add_argument("--lr",     type=float, default=2e-5)
    p_har.add_argument("--batch_size", type=int, default=32)
    p_har.add_argument("--n_jobs", type=int, default=1)

    # area2bump-import (convert legacy run directory into project cache format)
    p_a2i = sub.add_parser("area2bump-import", help="Import a legacy Area2Bump run directory into project cache format")
    p_a2i.add_argument("--run_id",     required=True, help="Name for results_cache/area2bump/<run_id>/")
    p_a2i.add_argument("--legacy_dir", required=True, help="Path to the existing run directory")

    # har-import (convert legacy run directory into project cache format)
    p_hari = sub.add_parser("har-import", help="Import a legacy HAR run directory into project cache format")
    p_hari.add_argument("--run_id",     required=True, help="Name for results_cache/har/<run_id>/")
    p_hari.add_argument("--legacy_dir", required=True, help="Path to the existing run directory")
    p_hari.add_argument("--n_jobs",     type=int, default=1)

    # har-figures (plot from cache)
    p_harf = sub.add_parser("har-figures", help="Generate HAR 2D grid figure from cached artifacts")
    p_harf.add_argument("--run_id", required=True)

    # har-figures3d
    p_harf3 = sub.add_parser("har-figures3d", help="Generate HAR 3D embedding plots from cache")
    p_harf3.add_argument("--run_id", required=True)
    p_harf3.add_argument("--elev", type=float, default=25)
    p_harf3.add_argument("--azim", type=float, default=-60)

    # Area2Bump figures from cache
    # 2D
    p_a2f = sub.add_parser("area2bump-figures", help="Generate Area2Bump 2D grid figure from cached artifacts")
    p_a2f.add_argument("--run_id", required=True)
    # 3D
    p_a2f3 = sub.add_parser("area2bump-figures3d", help="Generate Area2Bump 3D embedding plots from cache")
    p_a2f3.add_argument("--run_id", required=True)
    p_a2f3.add_argument("--elev", type=float, default=25)
    p_a2f3.add_argument("--azim", type=float, default=-60)


    args = parser.parse_args()

    if args.cmd == "synthetic-generate":
        from mmphate_repro.pipelines.synthetic_generate import (
            run_generate_suite, SuiteConfig, SCENARIOS, SCENARIO_NAMES,
        )
        from mmphate_repro.utils.paths import data_dir

        out_dir = Path(args.out_dir) if args.out_dir else (data_dir() / "synthetic")

        scenario_subset = None
        if args.scenarios:
            unknown = set(args.scenarios) - set(SCENARIO_NAMES)
            if unknown:
                parser.error(f"Unknown scenario(s): {unknown}. Valid names: {SCENARIO_NAMES}")
            scenario_subset = [(label, overrides) for label, overrides in SCENARIOS
                               if label in args.scenarios]

        base_cfg = SuiteConfig(
            epochs=args.epochs,
            timesteps=args.timesteps,
            samples=args.samples,
            seed=args.seed,
        )
        print(f"Generating synthetic data -> {out_dir}")
        saved = run_generate_suite(out_dir, base_cfg=base_cfg, scenarios=scenario_subset)
        print(f"\nDone. {len(saved)} scenario(s) saved.")
        return

    if args.cmd == "reproduce":
        reproduce_main(mode=args.mode, manifest_path=args.config, n_jobs=args.n_jobs)
        return

    if args.cmd == "synthetic-suite":
        cfg = SyntheticSuiteConfig(n_jobs_mphate=args.n_jobs)
        run_suite(Path(args.root), cfg, include=tuple(args.include) if args.include else None)
        return
    
    if args.cmd == "synthetic-figures":
        make_grids(args.scenarios)
        return
    
    if args.cmd == "area2bump-run":
        from mmphate_repro.pipelines.area2bump_full import run_area2bump_full, Area2BumpRunConfig
        from mmphate_repro.utils.paths import data_dir
        
        root = Path(args.data_root) if args.data_root else (data_dir() / "area2bump")
        cfg = Area2BumpRunConfig(seed=args.seed, epochs=args.epochs, lr=args.lr, n_units=args.units, n_jobs_mphate=args.n_jobs)
        run_area2bump_full(root, args.run_id, cfg)
        return

    if args.cmd == "area2bump-figures":
        make_area2bump_grid(args.run_id)
        return
    
    if args.cmd == "area2bump-figures3d":
        make_area2bump_grid_3d(args.run_id, view_elev=args.elev, view_azim=args.azim)
        return

    if args.cmd == "har-run":
        from mmphate_repro.pipelines.har_full import run_har_full, HARRunConfig
        from mmphate_repro.utils.paths import data_dir

        root = Path(args.data_root) if args.data_root else (data_dir() / "har")
        cfg = HARRunConfig(
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, n_units=args.units, n_jobs_mphate=args.n_jobs,
        )
        run_har_full(root, args.run_id, cfg)
        return

    if args.cmd == "area2bump-import":
        from mmphate_repro.pipelines.area2bump_full import import_area2bump_run
        out = import_area2bump_run(Path(args.legacy_dir), args.run_id)
        print(f"Imported to: {out}")
        return

    if args.cmd == "har-import":
        from mmphate_repro.pipelines.har_full import import_har_run
        import_har_run(Path(args.legacy_dir), args.run_id, n_jobs_mphate=args.n_jobs)
        return

    if args.cmd == "har-figures":
        from mmphate_repro.figures.har_grid_2d import make_har_grid
        out = make_har_grid(args.run_id)
        print(f"Saved: {out}")
        return

    if args.cmd == "har-figures3d":
        from mmphate_repro.figures.har_grid_3d import make_har_grid_3d
        for p in make_har_grid_3d(args.run_id, view_elev=args.elev, view_azim=args.azim):
            print(f"Saved: {p}")
        return