import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset
import pytest

from ttu_tower.io.store import read_table
from ttu_tower.timegrid import SAMPLES_PER_SLOT
from ttu_tower.viewer import slotdata
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.reprocess import Reprocessor, tau_block_ids
from ttu_tower.viewer.run import RunHandle

from viewer_fixtures import FAULT_HALF_HOUR, FULL_BOOMS


def _computed_slots(run_dir, boom):
    sb = read_table(run_dir / "primary" / "data" / "slot_boom")
    return sorted(sb[(sb["boom"] == boom) & (sb["status"].astype(str) == "computed")]["slot"].tolist())


# --- primary-only run (fast) ------------------------------------------------------------

@pytest.fixture(scope="module")
def primary(primary_run):
    run = RunHandle.from_dir(primary_run.run_dir)
    return run, FragmentIndex(run.run_dir)


def test_slot_tables_equal_filtered_whole_table_reads(primary_run, primary):
    _, index = primary
    k = 3 * FAULT_HALF_HOUR + 1
    bundle = slotdata.load_slot(index, k, 1, booms=[1])
    f = pa_dataset.field
    for table in ("means", "coverage", "ladder", "ladder_coverage", "mrd_frame"):
        whole = read_table(primary_run.run_dir / "primary" / "data" / table, filter=(f("slot") == k) & (f("boom") == 1))
        assert len(bundle.tables[table]) == len(whole) > 0, table
    mrd = bundle.mrd[bundle.mrd["slot"] == k]
    assert set(mrd["spectrum"]) == set(slotdata.SPECTRA)
    assert set(bundle.mrd["slot"]) >= {k - 1, k + 1}  # neighbours for the overlay


def test_acf_reproduces_every_stored_its_exactly(primary):
    run, index = primary
    rp = Reprocessor(run)
    k = 3 * FAULT_HALF_HOUR + 1
    values, trace = rp.acf(1, k)
    ladder = slotdata.load_slot(index, k, 1, booms=[1]).ladder
    stored = ladder[(ladder["variant"] == "mrd") & (ladder["stat"] == "its")]
    checked = 0
    for rung, var, stat, v in values:
        if stat != "its":
            continue
        s = stored[(stored["rung_s"] == rung) & (stored["variable"] == var)]["value"].iloc[0]
        assert (np.isnan(v) and np.isnan(s)) or v == s
        assert trace["acf"][rung][var] is None or trace["acf"][rung][var][0] == 1.0
        checked += 1
    assert checked == 32


def test_verify_reports_no_difference(primary):
    run, index = primary
    k = 3 * FAULT_HALF_HOUR + 1
    bundle = slotdata.load_slot(index, k, 1, booms=[1])
    stored = dict(bundle.tables, mrd=bundle.mrd[(bundle.mrd["slot"] == k)])
    report = Reprocessor(run).verify(1, k, stored)
    assert set(report["table"]) == {"means", "coverage", "slot_qc", "mrd_frame", "mrd", "ladder", "ladder_coverage"}
    assert (report["max abs diff"] == 0).all() and (report["compared"] > 0).all()
    assert (report["only recomputed"] == 0).all() and (report["only stored"] == 0).all()


def test_verify_notices_a_changed_value(primary):
    run, index = primary
    k = 3 * FAULT_HALF_HOUR + 1
    bundle = slotdata.load_slot(index, k, 1, booms=[1])
    means = bundle.tables["means"].copy()
    means.loc[means.index[0], "value"] += 1.0
    report = Reprocessor(run).verify(1, k, dict(bundle.tables, means=means))
    row = report[report["table"] == "means"].iloc[0]
    assert row["max abs diff"] == pytest.approx(1.0)


def test_tau_blocks_tile_each_slot_from_its_start():
    k = 1000
    g0 = k * SAMPLES_PER_SLOT - 1000
    ids = tau_block_ids(g0, 3 * SAMPLES_PER_SLOT, 600.0)
    edges = np.flatnonzero(np.diff(ids)) + 1 + g0
    assert list(edges) == [k * SAMPLES_PER_SLOT, (k + 1) * SAMPLES_PER_SLOT, (k + 2) * SAMPLES_PER_SLOT]
    fine = tau_block_ids(k * SAMPLES_PER_SLOT, SAMPLES_PER_SLOT, 9.375)
    assert np.unique(fine).size == 64
    assert tau_block_ids(g0, 100, 1200.0) is None


def test_all_rows_lists_every_value_with_its_row_key():
    tables = {
        "slot_qc": pd.DataFrame({"slot": [5, 5], "boom": [1, 1], "variable": ["ts", "w"], "stat": ["skew", "kurt"],
                                 "value": [0.2, np.nan]}),
        "tau": pd.DataFrame({"slot": [5], "boom": [1], "variant": ["mrd"], "cospectrum": ["uw"], "status": ["ok"],
                             "tau_s": [312.0]}),
        "ladder": pd.DataFrame({"slot": [5], "boom": [1], "variant": ["mrd"], "rung_s": [9.375], "variable": ["uw"],
                                "stat": ["te"], "value": [0.4]}),
        "pairs": pd.DataFrame({"slot": [5, 5], "boom": [1, 2], "boom2": [2, 3], "variant": ["mrd", "mrd"],
                               "variable": ["ri_b", "ri_b"], "value": [0.1, 0.2]}),
        "filter_log": pd.DataFrame({"slot": [5], "boom": [1], "variant": ["mrd"], "group": ["g"], "criterion": ["c"],
                                    "variable": ["x"]}),
        "slot_stats": pd.DataFrame({"slot": [5], "sun_elevation": [12.5], "night": [False]}),
    }
    out = slotdata.all_rows(tables, 1)
    assert list(out.columns) == ["table", "key", "field", "value", "row"]
    rows = [tuple(r) for r in out.itertuples(index=False)]
    assert rows[0] == ("slot_qc", "variable=ts, stat=skew", "value", 0.2, 0)
    assert rows[1][:3] == ("slot_qc", "variable=w, stat=kurt", "value") and np.isnan(rows[1][3])
    assert rows[2:4] == [("tau", "variant=mrd, cospectrum=uw", "status", "ok", 0),
                         ("tau", "variant=mrd, cospectrum=uw", "tau_s", 312.0, 0)]
    assert rows[4][1] == "variant=mrd, rung_s=9.375, variable=uw, stat=te"
    assert [r[1] for r in rows[5:7]] == ["boom=1, boom2=2, variant=mrd, variable=ri_b",
                                         "boom=2, boom2=3, variant=mrd, variable=ri_b"]
    assert [r[4] for r in rows[5:7]] == [0, 1]
    assert rows[7] == ("filter_log", "variant=mrd, group=g, criterion=c, variable=x", "", "", 0)
    assert rows[8:] == [("slot_stats", "", "sun_elevation", 12.5, 0), ("slot_stats", "", "night", False, 0)]
    assert slotdata.all_rows({}, 1).empty


# --- full run (slow)--------------------------------------------------------------------------

@pytest.mark.slow
def test_detection_rerun_reproduces_every_stored_tau(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    tau_final = read_table(full_run.run_dir / "tertiary" / "data" / "tau_final")
    checked = 0
    for boom in FULL_BOOMS:
        for k in _computed_slots(full_run.run_dir, boom):
            bundle = slotdata.load_slot(index, k, boom, booms=FULL_BOOMS)
            for variant in ("mrd", "mrd_unexcised"):
                views, (tau_s, source, status) = slotdata.rerun_detection(bundle, variant, run.cfg.secondary)
                for view in views.values():
                    assert view.stored is not None and view.agrees, (k, boom, variant, view.family)
                stored = slotdata.stored_selection(bundle, variant)
                assert (source, status) == (stored["source"], stored["source_status"])
                assert (np.isnan(tau_s) and np.isnan(stored["tau_s"])) or tau_s == stored["tau_s"]
                row = tau_final[(tau_final["slot"] == k) & (tau_final["boom"] == boom)
                                & (tau_final["variant"].astype(str) == variant)]
                assert row["tau_s"].iloc[0] == stored["tau_s"] or np.isnan(stored["tau_s"])
                checked += 1
    assert checked > 20


@pytest.mark.slow
def test_rung_scales_define_ils_as_its_times_u_mean(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    k = _computed_slots(full_run.run_dir, 1)[3]
    bundle = slotdata.load_slot(index, k, 1, booms=FULL_BOOMS)
    scales = slotdata.rung_scales(bundle, "mrd")
    assert list(scales.index) == [9.375 * 2 ** j for j in range(8)]
    lad = bundle.ladder[(bundle.ladder["variant"] == "mrd")]
    u_mean = lad[(lad["variable"] == "u") & (lad["stat"] == "mean")].set_index("rung_s")["value"]
    np.testing.assert_allclose(scales["ils_w"], scales["its_w"] * u_mean.reindex(scales.index).abs())
    # the selected-τ ILS matches what secondary stored
    stats = bundle.table("boom_stats")
    tau = slotdata.stored_selection(bundle, "mrd")["tau_s"]
    stored_ils = stats[(stats["variant"] == "mrd") & (stats["variable"] == "ils_w") & pd.isna(stats["stat"])]
    assert scales.loc[tau, "ils_w"] == pytest.approx(stored_ils["value"].iloc[0], rel=1e-12)


@pytest.mark.slow
def test_inspector_spectra_and_scales_tabs_draw(full_run, qtbot):
    pytest.importorskip("pytestqt")
    from ttu_tower.viewer.ui.app import configure_pyqtgraph
    from ttu_tower.viewer.ui.main_window import MainWindow

    configure_pyqtgraph()
    window = MainWindow(RunHandle.from_dir(full_run.run_dir))
    qtbot.addWidget(window)
    qtbot.waitUntil(lambda: window.catalog is not None, timeout=60_000)
    k = _computed_slots(full_run.run_dir, 1)[3]
    inspector = window.inspect(1, k, window.catalog["boom_final|uw|cov"], "mrd")
    qtbot.waitUntil(lambda: inspector.bundle is not None, timeout=60_000)
    assert inspector.tabs.currentWidget() is inspector.spectra
    assert set(inspector.spectra.plots) == set(slotdata.SPECTRA) | {"te_uw", "te_wvpts"}
    assert inspector.spectra.emphasis == "uw"
    te = slotdata.transport_efficiency(inspector.bundle.te, k, 1, "mrd", "uw")
    assert te is not None and te[0].size == 8 and np.nanmax(np.abs(te[1])) <= 1.0
    assert set(inspector.bundle.te["boom"]) > {1} and set(inspector.bundle.te["slot"]) > {k}  # for the overlays
    for key in ("unexcised", "booms", "neighbours"):
        inspector.spectra.boxes[key].setChecked(True)
    spectra = inspector.spectra
    spectra.normalize["uw"].setChecked(True)
    curve = next(i for i in spectra.plots["uw"].listDataItems() if i.opts["symbol"] == "o")
    assert np.nansum(curve.yData) == pytest.approx(1.0) and "normalized" in spectra.plots["uw"].titleLabel.text
    assert "normalized" not in spectra.plots["wvpts"].titleLabel.text
    spectra.normalize["uw"].setChecked(False)
    spectra.groups["transport"].setChecked(False)
    assert set(spectra.plots) == set(slotdata.SPECTRA)
    spectra.groups["variances"].setChecked(False)
    spectra.groups["other_covariances"].setChecked(False)
    assert set(spectra.plots) == {"wvpts", "uw"} and not spectra.boxes["log"].isEnabled()
    assert [n for n, box in spectra.normalize.items() if not box.isHidden()] == ["wvpts", "uw"]
    spectra.groups["transport"].setChecked(True)
    qtbot.wait(50)
    widths = {name: plot.width() for name, plot in spectra.plots.items()}
    assert set(widths) == {"wvpts", "uw", "te_uw", "te_wvpts"}  # one row of four, equal widths
    assert max(widths.values()) - min(widths.values()) < 3
    assert sum(widths.values()) > 0.9 * spectra.graphics.width()  # filling the row
    for box in spectra.groups.values():
        box.setChecked(False)
    assert not spectra.plots and "no plots chosen" in spectra.info.text()
    inspector.spectra.mode.setCurrentIndex(1)  # the ogive
    assert inspector.scales.table.rowCount() == 8
    inspector.close()
    window.close()
    window.runner.wait(30_000)


def test_spectra_plot_rows_follow_the_checked_groups():
    from ttu_tower.viewer.ui.inspector.spectra_tab import plot_rows
    every = {"heat_momentum", "other_covariances", "variances", "transport"}
    assert plot_rows(every) == [["wvpts", "uw", "vw", "uv", "te_uw"], ["uu", "vv", "ww", "tsts", "te_wvpts"]]
    assert plot_rows(every - {"transport"}) == [["wvpts", "uw", "vw", "uv"], ["uu", "vv", "ww", "tsts"]]
    assert plot_rows({"heat_momentum", "variances"}) == [["wvpts", "uw"], ["uu", "vv", "ww", "tsts"]]
    assert plot_rows({"heat_momentum", "transport"}) == [["wvpts", "uw", "te_uw", "te_wvpts"]]
    assert plot_rows({"variances", "transport"}) == [["uu", "vv", "ww", "tsts"], ["te_uw", "te_wvpts"]]
    assert plot_rows(set()) == []
