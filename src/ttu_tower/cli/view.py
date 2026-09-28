"""ttu-view: browse a run's results, and the raw data behind them."""
import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ttu-view", description=__doc__)
    parser.add_argument("tag", nargs="?", help="Registered run tag (default: the last run viewed, or the first registered).")
    parser.add_argument("--run-dir", help="Open a run directory directly instead of by tag.")
    parser.add_argument("--test", action="store_true", help="Look the tag up among --test runs.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        from ttu_tower.viewer.ui.app import main as run_app
    except ImportError as exc:
        sys.exit(f"ttu-view needs the viewer extra: pip install -e \".[viewer]\" ({exc})")
    sys.exit(run_app(tag=args.tag, run_dir=args.run_dir, test=args.test))


if __name__ == "__main__":
    main()
