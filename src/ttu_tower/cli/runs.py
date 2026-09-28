"""ttu-runs: list the registered runs."""
import argparse

import pandas as pd

from ttu_tower.results import list_runs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ttu-runs", description=__doc__)
    parser.add_argument("--test", action="store_true", help="List the --test runs instead.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runs = list_runs(test=args.test)
    if runs.empty:
        print("no registered runs")
        return
    runs = runs.assign(stages=runs["stages"].map(", ".join))
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(runs.to_string(index=False))


if __name__ == "__main__":
    main()
