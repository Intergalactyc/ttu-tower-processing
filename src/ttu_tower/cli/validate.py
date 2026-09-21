"""ttu-validate: run first-run validation checks and write their CSVs."""
import argparse

from ttu_tower.config.load import load_config
from ttu_tower.io.runs import run_dir_for
from ttu_tower.validation.checks import CHECK_NAMES, write_validation_reports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ttu-validate", description=__doc__)
    parser.add_argument("config", help="Path to the run's TOML config file.")
    parser.add_argument("--check", action="append", choices=CHECK_NAMES, dest="checks",
                         help="Run only this check (repeatable). Default: run every check.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for checks with a stochastic component.")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Worker process count for despike_calibration, the one check that parallelizes (default: every available core).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_dir = run_dir_for(cfg, test=False)
    write_validation_reports(run_dir, cfg, checks=args.checks, seed=args.seed, nproc=args.nproc)
    names = args.checks or CHECK_NAMES
    print(f"wrote validation reports for {', '.join(names)} to {run_dir / 'reports' / 'validation'}")


if __name__ == "__main__":
    main()
