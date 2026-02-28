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
    p_rep = sub.add_parser("reproduce", help="Reproduce paper figures")
    p_rep.add_argument("--mode", choices=["cache", "full"], required=True)
    p_rep.add_argument(
        "--config",
        default="configs/figure_manifest.json",
        help="Path to figure manifest JSON",
    )

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
    if args.cmd == "reproduce":
        reproduce_main(mode=args.mode, manifest_path=args.config)
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