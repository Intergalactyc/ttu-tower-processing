"""ttu-runall: run primary -> secondary -> tertiary as subprocesses."""
import argparse
import subprocess
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ttu-runall", description=__doc__)
    parser.add_argument("config", help="Path to the run's TOML config file.")
    parser.add_argument("--nproc", type=int, default=None, help="Passed through to primary.")
    parser.add_argument("--redo-failures", action="store_true", help="Passed through to primary.")
    parser.add_argument("--allow-non-parquet", action="store_true", help="Passed through to primary.")
    parser.add_argument("--skip-primary", action="store_true",
                         help="Skip the primary stage (e.g. it already finished and you only want secondary/tertiary).")
    parser.add_argument("--test", action="store_true", help="Passed through to every stage.")
    parser.add_argument("--force", action="store_true", help="Passed through to every stage.")
    parser.add_argument("--fresh", action="store_true",
                         help="Passed through to every stage run (primary, unless --skip-primary; secondary; "
                              "tertiary): clears that stage's outputs unconditionally before running.")
    return parser.parse_args(argv)


def _run(module: str, config_path: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, config_path, *extra_args]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(argv=None):
    args = parse_args(argv)

    common = []
    if args.test:
        common.append("--test")
    if args.force:
        common.append("--force")
    if args.fresh:
        common.append("--fresh")

    if not args.skip_primary:
        primary_args = list(common)
        if args.nproc is not None:
            primary_args += ["--nproc", str(args.nproc)]
        if args.redo_failures:
            primary_args.append("--redo-failures")
        if args.allow_non_parquet:
            primary_args.append("--allow-non-parquet")
        _run("ttu_tower.cli.primary", args.config, primary_args)
    else:
        print("Skipping primary (--skip-primary).")

    _run("ttu_tower.cli.secondary", args.config, common)
    _run("ttu_tower.cli.tertiary", args.config, common)

    print("Full pipeline run complete.")


if __name__ == "__main__":
    main()
