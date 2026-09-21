import pandas as pd

from ttu_tower.io import store
from ttu_tower.tertiary import wide

_TZ = "Etc/GMT+6"


def _slot_start(n):
    return pd.date_range("2014-04-01", periods=n, freq="10min", tz=_TZ)


def _empty(columns):
    return pd.DataFrame(columns=columns)


def test_column_naming_round_trip():
    slot_start = _slot_start(1)
    tables = {
        "boom_final": pd.DataFrame([
            {"slot": 1, "slot_start": slot_start[0], "boom": 4, "variant": "none", "variable": "ws", "stat": "mean", "value": 5.0},
            {"slot": 1, "slot_start": slot_start[0], "boom": 4, "variant": "mrd", "variable": "ti", "stat": None, "value": 0.2},
            {"slot": 1, "slot_start": slot_start[0], "boom": 4, "variant": "mrd", "variable": "ws", "stat": "std_5s", "value": 0.3},
        ]),
        "tau_final": pd.DataFrame([
            {"slot": 1, "slot_start": slot_start[0], "boom": 4, "variant": "mrd", "tau_s": 37.5,
             "source": "momentum", "source_status": "found"},
        ]),
        "pairs": pd.DataFrame([
            {"slot": 1, "slot_start": slot_start[0], "boom": 1, "boom2": 4, "variant": "none", "variable": "rib", "value": 0.1},
        ]),
        "profile": pd.DataFrame([
            {"slot": 1, "slot_start": slot_start[0], "variant": "none", "variable": "alpha", "value": 0.25},
            {"slot": 1, "slot_start": slot_start[0], "variant": "mrd", "variable": "gamma", "value": -0.1},
        ]),
        "slot_final": pd.DataFrame([
            {"slot": 1, "slot_start": slot_start[0], "variable": "sun_elevation", "value": 45.0},
            {"slot": 1, "slot_start": slot_start[0], "variable": "night", "value": 0.0},
        ]),
    }

    result = wide.pivot_tables(tables, slot_start)

    row = result.iloc[0]
    assert row["ws_mean_b4"] == 5.0
    assert row["ti_b4_mrd"] == 0.2
    assert row["ws_std_5s_b4_mrd"] == 0.3
    assert row["tau_b4_mrd"] == 37.5
    assert row["tau_source_b4_mrd"] == "momentum"
    assert row["rib_b1-b4"] == 0.1
    assert row["alpha"] == 0.25
    assert row["gamma_mrd"] == -0.1
    assert row["sun_elevation"] == 45.0
    assert row["night"] == 0.0


def test_monthly_files_concatenate_to_the_same_frame_as_a_single_pivot(tmp_path):
    starts = pd.DatetimeIndex(
        list(pd.date_range("2014-04-30 23:50", periods=2, freq="10min", tz=_TZ))
        + list(pd.date_range("2014-05-01 00:10", periods=2, freq="10min", tz=_TZ)),
    )
    boom_final = pd.DataFrame([
        {"slot": i, "slot_start": s, "boom": 4, "variant": "none", "variable": "ws", "stat": "mean", "value": float(i)}
        for i, s in enumerate(starts)
    ])
    tables = {
        "boom_final": boom_final,
        "tau_final": _empty(["slot", "slot_start", "boom", "variant", "tau_s", "source", "source_status"]),
        "pairs": _empty(["slot", "slot_start", "boom", "boom2", "variant", "variable", "value"]),
        "profile": _empty(["slot", "slot_start", "variant", "variable", "value"]),
        "slot_final": _empty(["slot", "slot_start", "variable", "value"]),
    }

    stage_dir = tmp_path / "tertiary"
    for name, df in tables.items():
        store.write_fragment(stage_dir / "data" / name, "batch0", df)

    wide.write_wide_export(stage_dir, _TZ)

    written = sorted((stage_dir / "wide").glob("*.parquet"))
    assert [p.stem for p in written] == ["2014-04", "2014-05"]

    from_files = pd.concat((pd.read_parquet(p) for p in written), ignore_index=True)
    from_files = from_files.sort_values("slot_start").reset_index(drop=True)

    direct = wide.pivot_tables(tables, pd.DatetimeIndex(sorted(starts)))
    direct = direct.sort_values("slot_start").reset_index(drop=True)

    pd.testing.assert_frame_equal(from_files, direct, check_like=True)


def test_pivot_tables_never_sees_more_than_one_months_rows(tmp_path, monkeypatch):
    """Regression test: `write_wide_export` used to hand `pivot_tables` the
    full year on every monthly iteration, only cropping to the month
    afterward via `.reindex()` - correct output, but doing the whole year's
    work twelve times. Each call must receive only that month's rows.
    """
    starts = pd.DatetimeIndex(
        list(pd.date_range("2014-04-30 23:30", periods=2, freq="10min", tz=_TZ))  # unambiguously April
        + list(pd.date_range("2014-05-01 00:10", periods=3, freq="10min", tz=_TZ)),  # unambiguously May
    )
    boom_final = pd.DataFrame([
        {"slot": i, "slot_start": s, "boom": 4, "variant": "none", "variable": "ws", "stat": "mean", "value": float(i)}
        for i, s in enumerate(starts)
    ])
    empty_tables = {
        "tau_final": _empty(["slot", "slot_start", "boom", "variant", "tau_s", "source", "source_status"]),
        "pairs": _empty(["slot", "slot_start", "boom", "boom2", "variant", "variable", "value"]),
        "profile": _empty(["slot", "slot_start", "variant", "variable", "value"]),
        "slot_final": _empty(["slot", "slot_start", "variable", "value"]),
    }

    stage_dir = tmp_path / "tertiary"
    store.write_fragment(stage_dir / "data" / "boom_final", "batch0", boom_final)
    for name, df in empty_tables.items():
        store.write_fragment(stage_dir / "data" / name, "batch0", df)

    seen_boom_final_sizes = []
    real_pivot_tables = wide.pivot_tables

    def spy(tables, index):
        seen_boom_final_sizes.append(len(tables["boom_final"]))
        return real_pivot_tables(tables, index)

    monkeypatch.setattr(wide, "pivot_tables", spy)
    wide.write_wide_export(stage_dir, _TZ)

    assert seen_boom_final_sizes == [2, 3], (
        "pivot_tables saw a row count other than each month's own - the full "
        "table is leaking into a call that should be scoped to one month"
    )
