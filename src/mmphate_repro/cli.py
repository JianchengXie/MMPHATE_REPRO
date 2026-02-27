import argparse
from mmphate_repro.pipelines.reproduce import reproduce_main

def main():
    parser = argparse.ArgumentParser(prog="mmphate-repro")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("reproduce", help="Reproduce paper figures")
    p_rep.add_argument("--mode", choices=["cache", "full"], required=True)
    p_rep.add_argument("--config", default="configs/figure_manifest.json",
                       help="Path to figure manifest JSON")

    args = parser.parse_args()

    if args.cmd == "reproduce":
        reproduce_main(mode=args.mode, manifest_path=args.config)