"""ttu-tertiary: run the tertiary stage."""
import argparse

from ttu_tower.config.load import load_config
from ttu_tower.tertiary.runner import run_tertiary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ttu-tertiary", description=__doc__)
    parser.add_argument("config", help="Path to the run's TOML config file.")
    parser.add_argument("--test", action="store_true", help="Process only the first batch, in a separate test run directory.")
    parser.add_argument("--force", action="store_true",
                         help="Clear this stage's outputs on a config-hash mismatch, or re-register a tag pointed elsewhere.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_tertiary(cfg, args)


if __name__ == "__main__":
    main()
