"""ttu-report: write the QC/yield reports for a run's completed stages."""
import argparse

from ttu_tower.config.load import load_config
from ttu_tower.io.runs import run_dir_for
from ttu_tower.post.report import write_stability_reports
from ttu_tower.primary.report import write_primary_reports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ttu-report", description=__doc__)
    parser.add_argument("config", help="Path to the run's TOML config file.")
    parser.add_argument("--stability", action="store_true", help="Also write the post/stability reports.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_dir = run_dir_for(cfg, test=False)
    reports_dir = run_dir / "reports"

    primary_dir = run_dir / "primary"
    if (primary_dir / "run_summary.json").is_file():
        write_primary_reports(primary_dir, reports_dir, cfg)
        print(f"wrote primary reports to {reports_dir}")
    else:
        print("primary stage has no run_summary.json yet; skipping its reports")

    if args.stability:
        tertiary_dir = run_dir / "tertiary"
        if (tertiary_dir / "run_summary.json").is_file():
            write_stability_reports(run_dir, reports_dir, cfg)
            print(f"wrote stability reports to {reports_dir}")
        else:
            print("tertiary stage has no run_summary.json yet; skipping stability reports")


if __name__ == "__main__":
    main()
