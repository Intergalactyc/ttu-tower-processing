from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ttu_tower import schema
from ttu_tower.config import config_from_dict
from ttu_tower.constants import SAMPLE_HZ, STAGE_B_MARGIN_S
from ttu_tower.io.load import load_boom
from ttu_tower.io.rawfiles import build_file_table, resolve_period_slots
from ttu_tower.io.runs import find_home
from ttu_tower.io.store import read_manifest, read_table
from ttu_tower.primary.products import file_products
from ttu_tower.primary.runner import run_primary
from ttu_tower.primary.stage_a import StageA, stage_a
from ttu_tower.primary.stage_b import Span, StageBOut, stage_b
from ttu_tower.timegrid import SAMPLES_PER_HALF_HOUR, half_hour_to_time
from ttu_tower.validation.synthetic import write_raw_dataset

_HALF_HOURS = list(range(5000, 5012))  # 12 half-hours
_BOOMS = [1, 2]
_MISSING = [5005, 5008, 5009]  # 5005 isolated; 5008-5009 a 2-file outage
_TABLES = ("coverage", "means", "slot_qc", "flags", "mrd", "mrd_frame", "ladder", "ladder_coverage", "slot_boom")
_MARGIN = STAGE_B_MARGIN_S * SAMPLE_HZ


def _mutate(h, boom, df):
    df = df.copy()
    if h == 5003 and boom == 2:
        for col in df.columns:
            df[col] = np.nan  # boom 2's entire file is unusable
    if h == 5002 and boom == 1:
        df.loc[df.index[1000:1003], "u_1"] = 500.0  # a spike
    if h == 5006 and boom == 1:
        df.loc[df.index[2000:2400], "u_1"] = df["u_1"].iloc[2000]  # a stuck run
    if h == 5010 and boom == 1:
        df.loc[df.index[-300:], "u_1"] = np.nan  # a gap straddling the 5010/5011 boundary
    if h == 5011 and boom == 1:
        df.loc[df.index[:300], "u_1"] = np.nan
    return df


def _args(config, **overrides):
    base = dict(config=config, nproc=1, test=False, redo_failures=False, force=False, allow_non_parquet=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _cfg(raw_dir, tag, **sections):
    start = half_hour_to_time(_HALF_HOURS[0] + 1).strftime("%Y-%m-%d %H:%M")
    end = half_hour_to_time(_HALF_HOURS[-1]).strftime("%Y-%m-%d %H:%M")
    raw = {
        "tag": tag, "paths": {"raw_dirs": [str(raw_dir)]},
        "period": {"start": start, "end": end},
        "files": {"bad_records": [], "booms": _BOOMS},
        "tertiary": {"veer_reference_boom": 1, "fits": {k: [1] for k in
                     ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms")}},
    }
    raw.update(sections)
    return config_from_dict(raw)


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw_acceptance") / "raw"
    rng = np.random.default_rng(123)
    write_raw_dataset(d, _HALF_HOURS, _BOOMS, rng, missing=_MISSING, mutate=_mutate)
    return d


def _read_all_tables(run_dir):
    return {t: read_table(run_dir / "primary" / "data" / t) for t in _TABLES}


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key_cols = [c for c in ("slot", "boom", "variant", "variable", "stat", "layer", "test", "start", "spectrum", "scale_s", "rung_s", "family") if c in df.columns]
    # categorical columns can carry different (differently-ordered) category
    # sets between two otherwise-identical runs; sort by their string values.
    sort_df = df.copy()
    for c in key_cols:
        if isinstance(sort_df[c].dtype, pd.CategoricalDtype):
            sort_df[c] = sort_df[c].astype(str)
    order = sort_df.sort_values(key_cols).index
    return df.loc[order].reset_index(drop=True)


def _assert_tables_equal(a: dict, b: dict):
    for table in a:
        da, db = _sorted(a[table]), _sorted(b[table])
        assert list(da.columns) == list(db.columns), table
        assert len(da) == len(db), f"{table}: {len(da)} vs {len(db)} rows"
        for col in da.columns:
            if pd.api.types.is_float_dtype(da[col]):
                np.testing.assert_allclose(da[col].to_numpy(dtype=float), db[col].to_numpy(dtype=float),
                                            rtol=1e-12, atol=1e-15, equal_nan=True, err_msg=f"{table}.{col}")
            else:
                assert da[col].astype(str).tolist() == db[col].astype(str).tolist(), f"{table}.{col}"


def _empty_table(table: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(schema.TABLES[table].columns))


def _accepted_runs(file_table: pd.DataFrame) -> list[list[int]]:
    """Maximal runs of consecutive accepted half-hours, in order."""
    accepted = sorted(int(h) for h in file_table.loc[file_table["status"] == "accepted", "half_hour"])
    runs: list[list[int]] = []
    for h in accepted:
        if runs and h == runs[-1][-1] + 1:
            runs[-1].append(h)
        else:
            runs.append([h])
    return runs


def _whole_run_stage_b(stage_a_by_h: dict[int, StageA], run_half_hours: list[int], boom: int, cfg,
                        edge_margin: int) -> dict[int, StageBOut]:
    """Stage B for every half-hour in `run_half_hours`, each given the *whole
    run* as its span (not just the standard one-file-each-side margin) - the
    seam-invariance reference. The array still carries `edge_margin` samples
    of NaN padding beyond the run's own true ends, matching `build_span`'s
    own convention there (a span is always `[90000h - M, 90000(h+1) + M)`
    regardless of what's actually accepted beyond it): the point of this
    reference is to prove that *more available real context than the
    standard margin provides* doesn't change the result, not to give despike
    fewer array positions to consider than the real pipeline ever would.
    `series`/`file_run_id` are built once and shared; only
    `core_bounds_removed` differs per half-hour.
    """
    h0 = run_half_hours[0]
    run_samples = len(run_half_hours) * SAMPLES_PER_HALF_HOUR
    n = run_samples + 2 * edge_margin
    g0 = SAMPLES_PER_HALF_HOUR * h0 - edge_margin
    variables = ("ue", "vn", "w", "ts", "t", "rh", "p")
    series = {v: np.full(n, np.nan) for v in variables}
    file_run_id = np.full(n, -1, dtype=np.int64)
    file_run_id[edge_margin : edge_margin + run_samples] = h0  # one accepted run -> one tag throughout
    for i, h in enumerate(run_half_hours):
        a = stage_a_by_h[h]
        lo = edge_margin + i * SAMPLES_PER_HALF_HOUR
        for v in variables:
            series[v][lo : lo + SAMPLES_PER_HALF_HOUR] = getattr(a, v)

    out = {}
    for h in run_half_hours:
        span = Span(g0=g0, series=series, file_run_id=file_run_id, core_bounds_removed=stage_a_by_h[h].bounds_removed)
        out[h] = stage_b(span, h, boom, cfg)
    return out


def _reference_tables(raw_dir, cfg, boom: int, edge_margin: int) -> dict[str, pd.DataFrame]:
    """The seam-invariance reference for one boom: every accepted file run
    processed with the whole run as Stage B's span, then the same
    `file_products` the real pipeline uses, filtered to the period exactly
    as `run_unit` does.
    """
    file_table = build_file_table(cfg.paths.raw_dirs, cfg.files)
    accepted = file_table[file_table["status"] == "accepted"].set_index("half_hour")["path"]
    slot_a, slot_b = resolve_period_slots(cfg, file_table)
    h_a, h_b = slot_a // 3, (slot_b - 1) // 3

    stage_a_by_h = {int(h): stage_a(load_boom(p, boom, str(raw_dir)), boom, cfg.qc) for h, p in accepted.items()}

    B: dict[int, StageBOut] = {}
    for run in _accepted_runs(file_table):
        B.update(_whole_run_stage_b(stage_a_by_h, run, boom, cfg, edge_margin))

    collected: dict[str, list[pd.DataFrame]] = {t: [] for t in _TABLES}
    for h in range(h_a, h_b + 1):
        parts = file_products(h, B, boom, cfg)
        for table, df in parts.items():
            if table != "flags" and "slot" in df.columns:
                df = df[(df["slot"] >= slot_a) & (df["slot"] < slot_b)]
            collected[table].append(df)

    return {t: (pd.concat(collected[t], ignore_index=True) if collected[t] else _empty_table(t)) for t in _TABLES}


def test_seam_invariance_against_whole_run_reference(tmp_path, monkeypatch, raw_dir):
    # The streamed pipeline processes Stage B on a file +-10-min-margin span;
    # this proves that gives the same result as using the whole accepted file
    # run as context, which batch-partition invariance alone cannot show
    # (both configurations build Stage B the same margin-based way, so a
    # reach bug beyond the margin would be wrong identically in both).
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg = _cfg(raw_dir, "seam", primary={"batch_max_files": 3})
    run_primary(cfg, _args(str(tmp_path / "seam.toml")))

    streamed = _read_all_tables(find_home() / "results" / "seam")

    per_boom = [_reference_tables(raw_dir, cfg, boom, edge_margin=_MARGIN) for boom in _BOOMS]
    reference = {t: pd.concat([tables[t] for tables in per_boom], ignore_index=True) for t in _TABLES}
    _assert_tables_equal(streamed, reference)

    # A bigger edge margin (more available "real" context than the standard
    # margin ever provides) must give exactly the same result - otherwise the
    # match above would just be an artifact of matching build_span's specific
    # margin size, not genuine seam invariance.
    per_boom_bigger = [_reference_tables(raw_dir, cfg, boom, edge_margin=2 * _MARGIN) for boom in _BOOMS]
    reference_bigger = {t: pd.concat([tables[t] for tables in per_boom_bigger], ignore_index=True) for t in _TABLES}
    _assert_tables_equal(reference, reference_bigger)


def test_batch_partition_invariance(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg_small = _cfg(raw_dir, "batchsmall", primary={"batch_max_files": 2})
    cfg_big = _cfg(raw_dir, "batchbig", primary={"batch_max_files": 96})

    run_primary(cfg_small, _args(str(tmp_path / "small.toml")))
    run_primary(cfg_big, _args(str(tmp_path / "big.toml")))

    home = find_home()
    small = _read_all_tables(home / "results" / "batchsmall")
    big = _read_all_tables(home / "results" / "batchbig")
    _assert_tables_equal(small, big)


def test_parallel_equals_serial(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg1 = _cfg(raw_dir, "nproc1", primary={"batch_max_files": 3})
    cfg3 = _cfg(raw_dir, "nproc3", primary={"batch_max_files": 3})

    run_primary(cfg1, _args(str(tmp_path / "n1.toml"), nproc=1))
    run_primary(cfg3, _args(str(tmp_path / "n3.toml"), nproc=3))

    home = find_home()
    serial = _read_all_tables(home / "results" / "nproc1")
    parallel = _read_all_tables(home / "results" / "nproc3")
    _assert_tables_equal(serial, parallel)


def test_statuses_cover_every_slot_boom_of_the_period(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg = _cfg(raw_dir, "statuses")
    run_primary(cfg, _args(str(tmp_path / "statuses.toml")))

    run_dir = find_home() / "results" / "statuses"
    slot_boom = read_table(run_dir / "primary" / "data" / "slot_boom")

    period_half_hours = range(_HALF_HOURS[0] + 1, _HALF_HOURS[-1])
    expected_slots = {3 * h + i for h in period_half_hours for i in range(3)}
    assert set(zip(slot_boom["slot"], slot_boom["boom"])) == {(s, b) for s in expected_slots for b in _BOOMS}

    by_slot_boom = slot_boom.set_index(["slot", "boom"])["status"]
    for h in (5005, 5008, 5009):  # outage / isolated missing -> no_file
        for i in range(3):
            for b in _BOOMS:
                assert by_slot_boom[(3 * h + i, b)] == "no_file"
    for i in range(3):  # boom 2's all-NaN file -> no_data
        assert by_slot_boom[(3 * 5003 + i, 2)] == "no_data"
        assert by_slot_boom[(3 * 5003 + i, 1)] == "computed"  # boom 1 is unaffected


def test_context_beyond_isolated_missing_file(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg = _cfg(raw_dir, "context")
    run_primary(cfg, _args(str(tmp_path / "context.toml")))

    run_dir = find_home() / "results" / "context"
    mrd_frame = read_table(run_dir / "primary" / "data" / "mrd_frame")
    # the slot right after the isolated missing file 5005: its 80-min window
    # reaches into file 5004 and 5006, so momentum coverage should be
    # partial (not 0, not 1) rather than missing entirely.
    slot = 3 * 5006  # first slot of the file right after the gap
    row = mrd_frame[(mrd_frame["slot"] == slot) & (mrd_frame["boom"] == 1) & (mrd_frame["variant"] == "mrd")]
    assert not row.empty
    coverage = row["coverage_momentum"].iloc[0]
    assert 0.0 < coverage < 1.0, coverage


def test_period_boundary_not_half_hour_aligned_emits_exact_slot_set(tmp_path, monkeypatch, raw_dir):
    # A period boundary need only fall on a 10-min (slot) boundary, not a
    # half-hour one - Stage A/B still process the whole of half-hours 5001
    # and 5011 for context, but the emitted slots must stop exactly at the
    # period edge, matching slots.parquet exactly.
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    start = half_hour_to_time(_HALF_HOURS[1]) + pd.Timedelta(minutes=10)
    end = half_hour_to_time(_HALF_HOURS[-1]) + pd.Timedelta(minutes=20)
    cfg = _cfg(raw_dir, "boundary", period={
        "start": start.strftime("%Y-%m-%d %H:%M"), "end": end.strftime("%Y-%m-%d %H:%M"),
    })
    run_primary(cfg, _args(str(tmp_path / "boundary.toml")))

    run_dir = find_home() / "results" / "boundary"
    slot_boom = read_table(run_dir / "primary" / "data" / "slot_boom")
    slots_table = pd.read_parquet(run_dir / "primary" / "slots.parquet")

    expected_slots = set(slots_table["slot"].tolist())
    assert set(slot_boom["slot"].tolist()) == expected_slots
    assert 3 * _HALF_HOURS[1] not in expected_slots  # first slot of 5001: before the period
    assert 3 * _HALF_HOURS[1] + 1 in expected_slots  # second slot of 5001: the period's first slot
    assert 3 * _HALF_HOURS[-1] + 2 not in expected_slots  # last slot of 5011: after the period
    assert 3 * _HALF_HOURS[-1] + 1 in expected_slots  # second slot of 5011: the period's last slot

    # coverage/means/etc. (not flags) must be filtered the same way
    means = read_table(run_dir / "primary" / "data" / "means")
    assert set(means["slot"].tolist()) <= expected_slots
    assert 3 * _HALF_HOURS[1] not in set(means["slot"].tolist())


def test_resume_after_redo_failures(tmp_path, monkeypatch):
    # A `spawn` worker re-imports everything fresh, so a monkeypatch in this
    # process can't reach it - fail a unit for real instead, by corrupting
    # one accepted file's bytes so reading it raises (not the row-count
    # mismatch load_boom already treats as a missing file).
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    raw_dir = tmp_path / "raw"
    half_hours = list(range(6000, 6008))
    rng = np.random.default_rng(55)
    write_raw_dataset(raw_dir, half_hours, [1, 2], rng)  # boom 2's columns stay in the file untouched below

    from ttu_tower.timegrid import half_hour_to_time as hht
    cfg = config_from_dict({
        "tag": "resume", "paths": {"raw_dirs": [str(raw_dir)]},
        "period": {"start": hht(6001).strftime("%Y-%m-%d %H:%M"), "end": hht(6007).strftime("%Y-%m-%d %H:%M")},
        "files": {"bad_records": [], "booms": [1]},
        "tertiary": {"veer_reference_boom": 1, "fits": {k: [1] for k in
                     ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms")}},
        "primary": {"batch_max_files": 3},
    })

    # Drop boom 1's columns from one file: valid parquet, same row count (so
    # build_file_table still accepts it), but load_boom(..., boom=1, ...)
    # will raise a real (non-BadFileError) error selecting them - a genuine
    # work-unit failure, not the row-count mismatch load_boom already treats
    # as a missing file.
    target = sorted(raw_dir.glob("*.parquet"))[3]  # some file inside the period
    original = pd.read_parquet(target)
    boom_1_cols = [c for c in original.columns if c.endswith("_1")]
    original.drop(columns=boom_1_cols).to_parquet(target)

    summary1 = run_primary(cfg, _args(str(tmp_path / "resume.toml"), nproc=1))
    assert summary1["status_counts"]["failed"] > 0

    original.to_parquet(target)
    summary2 = run_primary(cfg, _args(str(tmp_path / "resume.toml"), nproc=1, redo_failures=True))
    assert summary2["status_counts"]["failed"] == 0

    run_dir = find_home() / "results" / "resume"
    manifest = read_manifest(run_dir / "primary" / "manifest")
    assert all(e["status"] == "success" for e in manifest.values())

    # no duplicate fragments: one file per unit per table
    for table in ("coverage", "means", "slot_boom"):
        frags = list((run_dir / "primary" / "data" / table).glob("*.parquet"))
        assert len(frags) == len(set(f.name for f in frags))


def test_manifest_summary_matches_slot_boom_fragment(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    cfg = _cfg(raw_dir, "summaries", primary={"batch_max_files": 3})
    run_primary(cfg, _args(str(tmp_path / "summaries.toml")))

    run_dir = find_home() / "results" / "summaries"
    manifest = read_manifest(run_dir / "primary" / "manifest")
    slot_boom = read_table(run_dir / "primary" / "data" / "slot_boom")

    for unit_id, entry in manifest.items():
        boom = int(unit_id[-2:])
        h_a, h_b = (int(x) for x in unit_id.split("_")[0][3:].split("-"))
        unit_slots = set(range(3 * h_a, 3 * (h_b + 1)))
        rows = slot_boom[(slot_boom["boom"] == boom) & (slot_boom["slot"].isin(unit_slots))]
        # a categorical column's value_counts() includes unobserved categories
        # at 0; run_unit's own slot_counts (a plain dict) omits them.
        expected_counts = {k: int(v) for k, v in rows["status"].astype(str).value_counts().items() if v > 0}
        assert entry["summary"]["slot_counts"] == expected_counts
