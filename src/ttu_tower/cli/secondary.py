"""ttu-secondary: run the secondary stage."""
import argparse

from ttu_tower.config.load import load_config
from ttu_tower.secondary.runner import run_secondary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ttu-secondary", description=__doc__)
    parser.add_argument("config", help="Path to the run's TOML config file.")
    parser.add_argument("--test", action="store_true", help="Process only the first batch, in a separate test run directory.")
    parser.add_argument("--force", action="store_true",
                         help="Clear this stage's outputs on a config-hash mismatch, or re-register a tag pointed elsewhere.")
    parser.add_argument("--fresh", action="store_true",
                         help="Clear this stage's outputs unconditionally before running, regardless of the config hash "
                              "(for a code-only change, which a hash match would otherwise skip batch-by-batch).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_secondary(cfg, args)


if __name__ == "__main__":
    main()
