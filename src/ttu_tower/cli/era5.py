"""ttu-era5-download: fetch ERA5 surface, pressure-level or model-level data from the CDS.

Resumable: files already present are skipped, so rerun after an interruption
or a failed request.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

from ttu_tower.era5 import config as era5_config
from ttu_tower.era5 import download

_SUBDIRS = {"surface": "surface", "pressure": "pressure", "model": "model"}


def parse_download_args(argv=None):
    p = argparse.ArgumentParser(prog="ttu-era5-download", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=list(download.BUILDERS))
    p.add_argument("--config", default="configs/era5.toml", help="era5 config (default: %(default)s)")
    p.add_argument("--start", type=dt.date.fromisoformat, help="first UTC day (default: config period)")
    p.add_argument("--end", type=dt.date.fromisoformat, help="last UTC day (default: config period)")
    p.add_argument("--out-dir", type=Path, help="output directory (default: <data_dir>/<source>)")
    p.add_argument("--parallel", type=int, help="concurrent requests (default: config)")
    p.add_argument("--dry-run", action="store_true", help="list the requests and exit")
    return p.parse_args(argv)


def download_main(argv=None):
    args = parse_download_args(argv)
    cfg = era5_config.load(args.config)
    start, end = args.start or cfg.start, args.end or cfg.end
    out_dir = args.out_dir or cfg.data_dir / _SUBDIRS[args.source]
    reqs = download.BUILDERS[args.source](cfg, out_dir, start, end)
    if args.dry_run:
        for r in reqs:
            print(f"{r.target}  {'(present)' if r.target.exists() else ''}")
        print(f"{len(reqs)} requests; example:\n{reqs[-1].request}")
        return
    failed = download.run(reqs, args.parallel or cfg.download.parallel)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    download_main()
