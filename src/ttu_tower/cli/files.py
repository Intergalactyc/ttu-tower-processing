"""ttu-files: build and print the raw file table."""
import argparse

import pandas as pd

from ttu_tower import __version__
from ttu_tower.config.load import load_config
from ttu_tower.io import store
from ttu_tower.io.rawfiles import RawFilesNotAllowed, build_file_table, check_raw_allowed
from ttu_tower.io.runs import register_run
from ttu_tower.timegrid import time_to_half_hour


def _period_half_hours(cfg) -> tuple[int, int] | None:
    if not cfg.period.start or not cfg.period.end:
        return None
    start = pd.Timestamp(cfg.period.start, tz=cfg.output.timezone)
    end = pd.Timestamp(cfg.period.end, tz=cfg.output.timezone)
    return int(time_to_half_hour(start)), int(time_to_half_hour(end))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ttu-files", description=__doc__)
    parser.add_argument("config", help="Path to the run's TOML config file.")
    parser.add_argument(
        "--allow-non-parquet", action="store_true",
        help="Allow accepted files that are still raw .csv/.csv.gz/.zip (slow; for small runs).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_dir = register_run(cfg, args.config, package_version=__version__)

    table = build_file_table(cfg.paths.raw_dirs, cfg.files)
    try:
        check_raw_allowed(table, args.allow_non_parquet)
    except RawFilesNotAllowed as e:
        raise SystemExit(str(e))

    store.write_fragment(run_dir / "primary", "files", table)

    print(f"{len(table)} files found:")
    for status, count in table["status"].value_counts().items():
        print(f"  {status}: {count}")

    bounds = _period_half_hours(cfg)
    if bounds is not None:
        h_start, h_end = bounds
        n_half_hours = h_end - h_start
        accepted = table[
            (table["status"] == "accepted") & (table["half_hour"] >= h_start) & (table["half_hour"] < h_end)
        ]
        fraction = len(accepted) / n_half_hours if n_half_hours else float("nan")
        print(f"period coverage: {len(accepted)}/{n_half_hours} = {fraction:.1%} of half-hours")


if __name__ == "__main__":
    main()
