"""Per-boom raw data loading."""
import numpy as np
import pyarrow.parquet as pq

from ttu_tower.constants import ROWS_PER_FILE
from ttu_tower.io.convert import convert_raw_file

_VARS = ("u", "v", "w", "ts", "t", "rh", "p")


class BadFileError(Exception):
    """A loaded file doesn't have the expected 90,000 rows."""


def load_boom(path, boom: int, temp_dir: str) -> dict[str, np.ndarray]:
    """Load one boom's seven raw variables (float64) from `path`. Parquet
    files are read directly; raw .csv/.csv.gz/.zip files (only reachable with
    --allow-non-parquet) are converted in memory first.
    """
    path = str(path)
    columns = [f"{v}_{boom}" for v in _VARS]

    if path.endswith(".parquet"):
        df = pq.read_table(path, columns=columns).to_pandas()
    else:
        df = convert_raw_file(path, temp_dir)

    arrays = {v: df[col].to_numpy(dtype=np.float64) for v, col in zip(_VARS, columns)}

    n = len(df)
    if n != ROWS_PER_FILE:
        raise BadFileError(f"{path}: {n} rows (expected {ROWS_PER_FILE})")

    return arrays
