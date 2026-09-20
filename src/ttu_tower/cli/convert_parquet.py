"""CLI entry point: convert raw data files (.csv, .csv.gz, zip-wrapped .csv) to
parquet, with the canonical positional-to-header schema already applied.
"""
import argparse
import dataclasses
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ttu_tower.io.convert import convert_raw_file

RAW_SUFFIXES = (".csv.gz", ".zip", ".csv")
_DEFAULT_NPROC = max((os.cpu_count() or 2) - 1, 1)


def _is_raw_data_file(path: Path) -> bool:
    return path.name.endswith(RAW_SUFFIXES)


def _output_stem(path: Path) -> str:
    name = path.name
    for suffix in RAW_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"not a recognized raw data file: {path}")


def discover_raw_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and _is_raw_data_file(p))


def _resolve_tree_dest(src: Path, root: Path, output_root: Path, flatten: bool) -> Path:
    stem = _output_stem(src)
    if flatten:
        return output_root / f"{stem}.parquet"
    return output_root / src.parent.relative_to(root) / f"{stem}.parquet"


def _resolve_single_file_dest(src: Path, output_path: Path, output_path_str: str) -> Path:
    # Path() strips trailing separators on construction, so the "did the user
    # write a trailing slash" check has to look at the raw string.
    stem = _output_stem(src)
    if output_path.is_dir() or output_path_str.endswith(("/", "\\")):
        return output_path / f"{stem}.parquet"
    return output_path


def _write_parquet_atomic(df, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, dest)


@dataclasses.dataclass
class ConversionResult:
    src: str
    success: bool
    error: str | None = None


def _convert_one(src: str, dest: str, temp_dir: str, in_place: bool) -> ConversionResult:
    """Runs in a worker process (top-level and picklable, per
    ProcessPoolExecutor's requirements) - converts one file and reports the
    outcome instead of printing, so worker output doesn't interleave.
    """
    try:
        df = convert_raw_file(src, temp_dir)
        _write_parquet_atomic(df, Path(dest))
        if in_place:
            os.unlink(src)
        return ConversionResult(src=src, success=True)
    except Exception as e:
        return ConversionResult(src=src, success=False, error=str(e))


def _copy_other_files(root: Path, raw_files: set[Path], output_root: Path) -> tuple[int, int]:
    copied = skipped = 0
    for src in root.rglob("*"):
        if not src.is_file() or src in raw_files:
            continue
        dest = output_root / src.relative_to(root)
        if dest.exists():
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    return copied, skipped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ttu-convert-parquet", description=__doc__)
    parser.add_argument("input_path", type=str, help="Raw data file, or directory to scan recursively for raw data files.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-o", "--output-path", type=str, help="Directory (or file, for single-file input) to write converted output into. Existing files at the destination are left alone (not overwritten) unless -r/--reconvert-completed is passed.")
    mode.add_argument("-i", "--in-place", action="store_true", help="Replace each raw file with its converted parquet equivalent, in the same location (overwrites).")

    parser.add_argument("-r", "--reconvert-completed", action="store_true", help="Overwrite destination files that already exist, instead of skipping them - lets you force a redo of a previous run. Only meaningful with -o: with -i, a completed conversion deletes the source raw file, so there's nothing left to reconvert.")

    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("-f", "--flatten", action="store_true", help="With -o and a directory input: write every converted file directly into the output directory, without recreating subdirectory structure.")
    layout.add_argument("-c", "--copy-other-files", action="store_true", help="With -o and a directory input: preserve subdirectory structure and also copy non-raw-data files unchanged, so the output directory exactly mirrors the input.")

    parser.add_argument("-n", "--nproc", type=int, metavar="N", default=None, help=f"Number of worker processes to use (default: {_DEFAULT_NPROC}, i.e. one per CPU core minus one). Pass 1 to convert sequentially in the main process.")

    args = parser.parse_args(argv)

    if args.in_place and (args.flatten or args.copy_other_files):
        parser.error("-f/--flatten and -c/--copy-other-files require -o/--output-path, not -i/--in-place")

    if args.in_place and args.reconvert_completed:
        parser.error("-r/--reconvert-completed has no effect with -i/--in-place (a completed conversion deletes the source raw file, so there's nothing left to reconvert)")

    if args.nproc is not None and args.nproc < 1:
        parser.error(f"--nproc must be >= 1, got {args.nproc}")

    return args


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    is_dir_input = input_path.is_dir()
    if not is_dir_input and (args.flatten or args.copy_other_files):
        raise SystemExit("-f/--flatten and -c/--copy-other-files only apply when converting a directory, not a single file")

    if is_dir_input:
        root = input_path
        raw_files = discover_raw_files(input_path)
    else:
        if not _is_raw_data_file(input_path):
            raise SystemExit(f"Not a recognized raw data file (expected .csv, .csv.gz, or .zip): {input_path}")
        root = input_path.parent
        raw_files = [input_path]

    if not raw_files:
        print(f"No raw data files found under {input_path}")
        return

    print(f"Found {len(raw_files)} raw data file(s) to convert.")

    output_root = None
    if not args.in_place:
        output_root = Path(args.output_path)
        if is_dir_input:
            if output_root.exists() and not output_root.is_dir():
                raise SystemExit(f"Output path exists and is not a directory: {output_root}")
            output_root.mkdir(parents=True, exist_ok=True)

    # Resolve each source's destination and skip already-done/colliding work up
    # front, sequentially - this is cheap (just path math + .exists() checks)
    # and the collision check needs to see every destination chosen so far, so
    # it can't be done independently inside parallel workers.
    skipped = 0
    written_this_run: set[Path] = set()
    to_convert: list[tuple[Path, Path]] = []
    for src in raw_files:
        if args.in_place:
            dest = src.with_name(f"{_output_stem(src)}.parquet")
        elif is_dir_input:
            dest = _resolve_tree_dest(src, root, output_root, args.flatten)
        else:
            dest = _resolve_single_file_dest(src, output_root, args.output_path)

        if dest in written_this_run:
            print(f"Skipping (destination collides with another source file - rerun without --flatten to avoid this): {src}")
            skipped += 1
            continue
        if not args.in_place and not args.reconvert_completed and dest.exists():
            skipped += 1
            continue

        written_this_run.add(dest)
        to_convert.append((src, dest))

    converted = failed = 0
    nproc = args.nproc or _DEFAULT_NPROC

    with tempfile.TemporaryDirectory(prefix="ttu_convert_") as temp_dir:
        if to_convert:
            with ProcessPoolExecutor(max_workers=nproc) as executor:
                futures = {
                    executor.submit(_convert_one, str(src), str(dest), temp_dir, args.in_place): src
                    for src, dest in to_convert
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Converting", unit="file"):
                    result = future.result()
                    if result.success:
                        converted += 1
                    else:
                        print(f"FAILED: {result.src}: {result.error}")
                        failed += 1

        if args.copy_other_files:
            copied, other_skipped = _copy_other_files(root, set(raw_files), output_root)
            print(f"Copied {copied} non-raw file(s), skipped {other_skipped} already present at the destination.")

    print(f"Done. Converted {converted}, skipped {skipped} (already existed), failed {failed}.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
