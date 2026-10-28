"""Command line:  fsphysics run|plot CONFIG [--stages a,b] [--set KEY=VALUE]"""

import argparse
import sys

import matplotlib


def main(argv=None):
    matplotlib.use("Agg")                 # write files, never open windows
    from .pipeline import plot, run

    parser = argparse.ArgumentParser(prog="fsphysics")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("run", "run stages and draw their figures"),
                       ("plot", "redraw figures from saved tables")):
        p = sub.add_parser(name, help=text)
        p.add_argument("config", help="the dataset's YAML file")
        p.add_argument("--stages", help="comma-separated, e.g. selection")
        p.add_argument("--set", action="append", default=[],
                       metavar="KEY=VALUE",
                       help="override one setting, e.g. selection.top_n=10")
    args = parser.parse_args(argv)
    stages = args.stages.split(",") if args.stages else None
    func = run if args.command == "run" else plot
    try:
        func(args.config, stages=stages, overrides=args.set)
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")


if __name__ == "__main__":
    main()
