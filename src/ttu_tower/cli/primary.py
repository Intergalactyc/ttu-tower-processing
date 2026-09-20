"""ttu-primary: run the primary stage."""
import argparse

from ttu_tower.config.load import load_config
from ttu_tower.primary.runner import run_primary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ttu-primary", description=__doc__)
    parser.add_argument("config", help="Path to the run's TOML config file.")
    parser.add_argument("--nproc", type=int, default=None, help="Worker process count (default: [primary].nproc, or 1).")
    parser.add_argument("--test", action="store_true", help="Process only the first batch, in a separate test run directory.")
    parser.add_argument("--redo-failures", action="store_true", help="Retry units marked failed in the manifest.")
    parser.add_argument("--force", action="store_true",
                         help="Clear this stage's outputs on a config-hash mismatch, or re-register a tag pointed elsewhere.")
    parser.add_argument("--allow-non-parquet", action="store_true",
                         help="Allow accepted files that are still raw .csv/.csv.gz/.zip (slow; for small runs).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_primary(cfg, args)


if __name__ == "__main__":
    main()
