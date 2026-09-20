import gzip
import zipfile

import numpy as np
import pytest

from ttu_tower.constants import HEADER_MAP_NEW, HEADER_MAP_OLD, SOURCE_HEADERS_NEW, SOURCE_HEADERS_OLD
from ttu_tower.io.convert import convert_raw_file, infer_source_schema

def _expected_mapped_columns(source_headers, header_map):
    out = set()
    for col in source_headers:
        prefix, boom = col.split("_")
        target = header_map.get(prefix)
        if target is not None:
            out.add(f"{target}_{boom}")
    return out


_NEW_MAPPED_COLUMNS = _expected_mapped_columns(SOURCE_HEADERS_NEW, HEADER_MAP_NEW)
_OLD_MAPPED_COLUMNS = _expected_mapped_columns(SOURCE_HEADERS_OLD, HEADER_MAP_OLD)


def test_infer_source_schema_new():
    headers, hmap = infer_source_schema(len(SOURCE_HEADERS_NEW))
    assert headers == SOURCE_HEADERS_NEW
    assert hmap == HEADER_MAP_NEW


def test_infer_source_schema_old():
    headers, hmap = infer_source_schema(len(SOURCE_HEADERS_OLD))
    assert headers == SOURCE_HEADERS_OLD
    assert hmap == HEADER_MAP_OLD


def test_infer_source_schema_unrecognized_count():
    with pytest.raises(ValueError):
        infer_source_schema(3)


def _write_csv(path, headers, n_rows=5, trailing_delimiter=False):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n_rows, len(headers))).astype(np.float32)
    lines = []
    for row in data:
        cells = ",".join(f"{v:.9g}" for v in row)  # enough digits to round-trip float32 exactly
        if trailing_delimiter:
            cells += ","
        lines.append(cells)
    path.write_text("\n".join(lines) + "\n")
    return data


def test_convert_raw_file_new_structure_csv(tmp_path):
    data = _write_csv(tmp_path / "raw.csv", SOURCE_HEADERS_NEW)
    df = convert_raw_file(str(tmp_path / "raw.csv"), str(tmp_path))

    assert set(df.columns) == _NEW_MAPPED_COLUMNS
    assert len(df) == 5

    src_col = SOURCE_HEADERS_NEW.index("TSN-TRANS_3")
    np.testing.assert_allclose(df["u_3"].to_numpy(), data[:, src_col])


def test_convert_raw_file_drops_trailing_empty_column(tmp_path):
    _write_csv(tmp_path / "raw.csv", SOURCE_HEADERS_NEW, trailing_delimiter=True)
    df = convert_raw_file(str(tmp_path / "raw.csv"), str(tmp_path))
    assert set(df.columns) == _NEW_MAPPED_COLUMNS


def test_convert_raw_file_old_structure_keeps_propeller_booms(tmp_path):
    _write_csv(tmp_path / "raw.csv", SOURCE_HEADERS_OLD)
    df = convert_raw_file(str(tmp_path / "raw.csv"), str(tmp_path))
    assert set(df.columns) == _OLD_MAPPED_COLUMNS
    assert "propu_3" in df.columns  # old structure has propeller booms on 3-10
    assert "propu_1" not in df.columns  # but not on 1-2


def test_convert_raw_file_gz(tmp_path):
    _write_csv(tmp_path / "raw.csv", SOURCE_HEADERS_NEW)
    with open(tmp_path / "raw.csv", "rb") as f_in, gzip.open(tmp_path / "raw.csv.gz", "wb") as f_out:
        f_out.write(f_in.read())
    df = convert_raw_file(str(tmp_path / "raw.csv.gz"), str(tmp_path))
    assert len(df) == 5


def test_convert_raw_file_zip(tmp_path):
    _write_csv(tmp_path / "raw.csv", SOURCE_HEADERS_NEW)
    zpath = tmp_path / "raw.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(tmp_path / "raw.csv", arcname="raw.csv")

    extract_dir = tmp_path / "extract"
    df = convert_raw_file(str(zpath), str(extract_dir))
    assert len(df) == 5
    assert not (extract_dir / "raw.csv").exists()  # cleaned up after reading


def test_convert_raw_file_zip_rejects_multi_member_archive(tmp_path):
    zpath = tmp_path / "raw.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.csv", "1,2,3\n")
        zf.writestr("b.csv", "1,2,3\n")
    with pytest.raises(ValueError):
        convert_raw_file(str(zpath), str(tmp_path / "extract"))
