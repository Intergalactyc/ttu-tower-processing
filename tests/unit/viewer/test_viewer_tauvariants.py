import dataclasses

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset
import pytest

from ttu_tower.viewer import stability as criteria
from ttu_tower.viewer import taustats, tauvariants as tv
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.run import RunHandle

from viewer_fixtures import FULL_BOOMS

pytestmark = pytest.mark.slow
_F = pa_dataset.field
_KEYS = {"boom_final": ["slot", "boom", "variable", "stat"], "boom_labels_final": ["slot", "boom", "label"],
         "tau_final": ["slot", "boom"], "profile": ["slot", "variable"],
         "filter_log": ["slot", "boom", "group", "criterion", "variable"]}


@pytest.fixture
def store(full_run, tmp_path):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    variants = tv.VariantStore(run, index, root=tmp_path / "variants", workers=0)
    index.variants = variants
    return variants


def _rows(index, table, variant):
    df = index.read(table, filter=_F("variant") == variant)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    if table == "profile":
        df = df[df["variable"].astype(str).str.startswith(("gamma", "wdgamma", "loglaw_z0_constrained"))]
    return df.drop(columns=["variant", "slot_start"], errors="ignore").sort_values(_KEYS[table]).reset_index(drop=True)


def _same(index, key, stored):
    for table in _KEYS:
        mine, theirs = _rows(index, table, key), _rows(index, table, stored)
        assert len(mine) == len(theirs) and len(theirs) > 0 or table == "filter_log", table
        pd.testing.assert_frame_equal(mine, theirs, check_dtype=False, obj=table)


def test_fixed_600_reproduces_the_runs_naive(store):
    key = tv.fixed_key(600.0)
    assert store.ensure(key) is not None and store.is_ready(key)
    _same(store.index, key, "naive")


def test_a_scheme_of_the_runs_own_selection_reproduces_mrd_and_unexcised(store):
    store.set_schemes({"own": tv.TauScheme(), "own unexcised": tv.TauScheme(data="mrd_unexcised")})
    _same(store.index, tv.scheme_key("own"), "mrd")
    _same(store.index, tv.scheme_key("own unexcised"), "mrd_unexcised")


def test_a_whatif_with_the_runs_settings_reproduces_mrd(store):
    cfg = store.run.cfg.secondary
    det = tuple(sorted((k, float(getattr(cfg.detection, k))) for k in tv.DETECTION_FIELDS))
    sel = (("fallback_tau_s", cfg.selection.fallback_tau_s), ("priority", ",".join(cfg.selection.priority)),
           ("rule", cfg.selection.rule))
    store.set_schemes({"w": tv.TauScheme(base="whatif", detection=det, selection=sel)})
    assert tv.TauScheme(base="whatif", detection=det, selection=sel).secondary_config(cfg) == cfg
    _same(store.index, tv.scheme_key("w"), "mrd")


def test_fixed_rungs_take_their_rung_everywhere(store):
    key = tv.fixed_key(1200.0)
    tau = store.index.read("tau_final", filter=_F("variant") == key)
    assert len(tau) > 0 and (tau["tau_s"] == 1200.0).all() and (tau["source"].astype(str) == "fixed").all()
    u = store.index.read("boom_final", filter=(_F("variant") == key) & (_F("variable") == "u") & (_F("stat") == "var"))
    ladder = store.index.read("ladder", filter=(_F("variant") == "mrd") & (_F("rung_s") == 1200.0)
                              & (_F("variable") == "u") & (_F("stat") == "var"))
    both = u.merge(ladder, on=["slot", "boom"], suffixes=("", "_ladder")).dropna(subset=["value"])
    assert len(both) > 0 and np.allclose(both["value"], both["value_ladder"])


def test_local_percent_classes_give_each_boom_its_share_and_their_choices(store):
    scheme = tv.TauScheme(classifier="local", local_mode="percent", percent=40.0, choices=("1200", "600", "run"),
                          otherwise="300")
    store.set_schemes({"local": scheme})
    key = tv.scheme_key("local")
    d = store.ensure(key)
    rules = tv.rules(d)
    tau = store.index.read("tau_final", filter=_F("variant") == key).merge(rules, on=["slot", "boom"])
    mrd = store.index.read("tau_final", filter=_F("variant") == "mrd")[["slot", "boom", "tau_s"]]
    zeta = store.index.read("boom_final", filter=(_F("variant") == "mrd") & (_F("variable") == "zeta"))
    for b in FULL_BOOMS:
        mine = tau[tau["boom"] == b]
        finite = np.isfinite(zeta.loc[zeta["boom"] == b, "value"]).sum()
        assert (mine["rule"] == "neutral").sum() == round(0.4 * finite)
        summary = store.summary(key)["booms"][str(b)]
        assert summary["threshold"] >= 0
    assert (tau.loc[tau["rule"] == "unstable", "tau_s"] == 1200.0).all()
    assert (tau.loc[tau["rule"] == "neutral", "tau_s"] == 600.0).all()
    assert (tau.loc[tau["rule"] == "otherwise", "tau_s"] == 300.0).all()
    stable = tau[tau["rule"] == "stable"].merge(mrd, on=["slot", "boom"], suffixes=("", "_mrd"))
    assert np.array_equal(stable["tau_s"].to_numpy(), stable["tau_s_mrd"].to_numpy(), equal_nan=True)
    # the classes follow ζ's sign outside the neutral share
    z = zeta[["slot", "boom", "value"]].merge(tau, on=["slot", "boom"])
    assert (z.loc[z["rule"] == "unstable", "value"] < 0).all() and (z.loc[z["rule"] == "stable", "value"] > 0).all()


def test_local_zeta0_and_a_fixed_rungs_zeta(store):
    scheme = tv.TauScheme(classifier="local", local_mode="zeta0", zeta0=0.1, zeta_variant=tv.fixed_key(300.0),
                          choices=("base", "600", "base"))
    store.set_schemes({"z0": scheme})
    key = tv.scheme_key("z0")
    store.ensure(key)
    assert store.is_ready(tv.fixed_key(300.0))  # its ζ was computed first
    tau = store.index.read("tau_final", filter=_F("variant") == key).merge(tv.rules(store.ready()[key]))
    zeta = store.index.read("boom_final", filter=(_F("variant") == tv.fixed_key(300.0)) & (_F("variable") == "zeta"))
    z = zeta[["slot", "boom", "value"]].merge(tau, on=["slot", "boom"])
    assert (z.loc[z["rule"] == "neutral", "value"].abs() <= 0.1).all()
    assert (z.loc[z["rule"].isin(["stable", "unstable"]), "value"].abs() > 0.1).all()


def test_global_classes_come_from_the_stability_scheme(store):
    spec = criteria.StabilitySpec("rib", tuple(FULL_BOOMS), "mrd", (-0.5, -0.01, 0.01, 0.5))
    scheme = tv.TauScheme(classifier="global", stability=spec, choices=("1200", "1200", "600", "300", "300"))
    store.set_schemes({"g": scheme})
    key = tv.scheme_key("g")
    d = store.ensure(key)
    classes = criteria.classify(criteria.values(store.index, spec), spec.classes())
    rules = tv.rules(d)
    rules["expected"] = rules["slot"].map(classes).fillna("otherwise")
    assert (rules["rule"] == rules["expected"]).all()


def test_scheme_errors_and_round_trip(tmp_path):
    bad = tv.TauScheme(classifier="local", zeta_variant="scheme:x", choices=("base",))
    assert any("another τ scheme" in e for e in bad.errors()) and any("one τ choice" in e for e in bad.errors())
    good = tv.TauScheme(base="whatif", detection=(("min_tau_s", 18.75),), kernel=(1.0, 1.0, 1.0),
                        selection=(("priority", "momentum,heat"), ("rule", "priority")), classifier="global",
                        stability=criteria.StabilitySpec(), stability_source="saved: x",
                        choices=("base", "run", "1200", "600", "9.375"))
    assert good.errors() == []
    store = tv.SchemeStore(tmp_path / "tau.json")
    store.put("a", good)
    assert tv.SchemeStore.open(tmp_path / "tau.json").items["a"] == good


def test_saved_schemes_classifying_by_the_retired_pair_ri_read_as_ri_g():
    for old in ("rib:layer", "rib:log"):
        scheme = tv.TauScheme.from_dict({"classifier": "local", "local_variable": old, "choices": ["base"] * 3})
        assert scheme.local_variable == "rig" and scheme.errors() == []
    assert tv.TauScheme.from_dict({}).local_variable == "zeta"


def test_reads_name_the_variants_they_want(store):
    key = tv.fixed_key(150.0)
    assert not store.is_ready(key)
    plain = store.index.read("boom_final", filter=_F("variant") == "mrd")
    assert not store.is_ready(key) and (plain["variant"].astype(str) == "mrd").all()
    both = store.index.read("tau_final", filter=_F("variant").isin([key, "mrd"]))
    assert store.is_ready(key) and set(both["variant"].astype(str)) == {key, "mrd"}
    unnamed = store.index.read("tau_final", columns=["slot", "variant"])
    assert key in set(unnamed["variant"].astype(str))  # computed ones join reads that name no variant
    store.clear()
    assert not store.is_ready(key)


def test_redetection_copies_mrd_into_unexcised_rows_without_their_own_spectra(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    data = taustats.load(run, index)
    det = dataclasses.replace(run.cfg.secondary.detection, peak_significance_se=0.5)
    sb = index.read("slot_boom", columns=["slot", "boom", "status", "unexcised_computed"])
    for b in FULL_BOOMS:
        out = taustats.redetect(index, data, b, det)
        rows = taustats.rows_of(data, b)
        copied = set(sb.loc[(sb["boom"] == b) & (sb["status"].astype(str) == "computed")
                            & ~sb["unexcised_computed"].astype(bool), "slot"])
        slot, variant = data.slot[rows], data.variant[rows]
        for f in taustats.SPECTRA:
            status, tau = out[f][0], out[f][1]
            for s in copied:
                m = np.flatnonzero((slot == s) & (variant == 0))
                u = np.flatnonzero((slot == s) & (variant == 1))
                if m.size and u.size:
                    assert status[u[0]] == status[m[0]]
                    assert tau[u[0]] == tau[m[0]] or (np.isnan(tau[u[0]]) and np.isnan(tau[m[0]]))


# --- the viewer --------------------------------------------------------------------------------------

def _idle(window):
    return not window.runner._callbacks and window.runner.pool.activeThreadCount() == 0


@pytest.fixture
def window(qtbot, full_run, tmp_path):
    pytest.importorskip("pytestqt")
    from ttu_tower.viewer.ui.app import configure_pyqtgraph
    from ttu_tower.viewer.ui.main_window import MainWindow
    configure_pyqtgraph()
    w = MainWindow(RunHandle.from_dir(full_run.run_dir))
    qtbot.addWidget(w)
    w.variants.root, w.variants.workers = tmp_path / "variants", 0
    w.tau_schemes.path = None  # memory only
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    qtbot.waitUntil(lambda: _idle(w) and w.datas[0] is not None, timeout=60_000)
    yield w
    w.runner.wait(30_000)


def test_fixed_rungs_are_in_every_variant_menu_and_load_once_computed(window, qtbot):
    panel = window.panels[0]
    panel.select("boom_final|ustar|", tv.fixed_key(1200.0), FULL_BOOMS)
    assert window.specs[0].variant == tv.fixed_key(1200.0)
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0] is not None
                    and window.datas[0].variant == tv.fixed_key(1200.0), timeout=120_000)
    assert any(np.isfinite(c.y).any() for c in window.datas[0].curves.values())
    for combo in (window.find.variant, window.qc_summary.variant, window.aniso_map.variant,
                  window.aniso_band.variant, window.stability_panel.variant):
        assert combo.findData(tv.fixed_key(300.0)) >= 0 and combo.findData(tv.fixed_key(600.0)) < 0
    inspector = window.inspect(FULL_BOOMS[0], int(window.datas[0].curves[FULL_BOOMS[0]].slots[0]),
                               variant=tv.fixed_key(1200.0))
    assert inspector.variant() == tv.fixed_key(1200.0)
    qtbot.waitUntil(lambda: _idle(window), timeout=60_000)
    assert "1200 s" in inspector.summary.text()
    inspector.close()


def test_a_saved_tau_scheme_becomes_a_variant_for_every_view(window, qtbot):
    p = window.tau_panel
    p.classifier.setCurrentIndex(p.classifier.findData("local"))
    p.local_percent.setChecked(True)
    p.percent.setValue(30.0)
    for box, c in zip(p.class_choices, ("1200", "600", "run")):
        box.setCurrentIndex(box.findData(c))
    scheme, problem = p.scheme_now()
    assert problem == "" and scheme.classifier == "local" and scheme.choices == ("1200", "600", "run")
    assert not p.compute_button.isEnabled()  # unsaved
    p.save_scheme("by ζ")
    key = tv.scheme_key("by ζ")
    assert p.chosen_key() == key and p.compute_button.isEnabled()
    window.panels[0].select("boom_final|ti|", "mrd", FULL_BOOMS)
    qtbot.waitUntil(lambda: _idle(window), timeout=60_000)
    assert window.panels[0].variant.findData(key) >= 0
    p.use_button.click()
    assert window.specs[0].variant == key and window.find.variant.currentData() == key
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0] is not None and window.datas[0].variant == key,
                    timeout=120_000)
    assert "computed for this run" in p.state.text() and p.counts.rowCount() > 0

    # a changed definition is computed afresh; a deleted one leaves the menus
    p.percent.setValue(60.0)
    p.save_scheme("by ζ")
    assert not window.variants.is_ready(key)
    qtbot.waitUntil(lambda: _idle(window) and window.variants.is_ready(key), timeout=120_000)
    assert window.variants.summary(key)["scheme"]["percent"] == 60.0
    p.delete_scheme("by ζ")
    assert window.find.variant.findData(key) < 0 and window.panels[0].variant.findData(key) < 0


def test_tau_panel_copies_the_summary_whatif_and_a_stability_scheme(window):
    p, view = window.tau_panel, window.tau_view
    view.priority.setCurrentIndex(view.priority.findData("momentum,heat"))
    view.fields["peak_significance_se"].setValue(3.5)
    p.copy_whatif.click()
    scheme, problem = p.scheme_now()
    assert problem == "" and scheme.base == "whatif"
    assert dict(scheme.selection)["priority"] == "momentum,heat"
    assert dict(scheme.detection)["peak_significance_se"] == 3.5
    p.classifier.setCurrentIndex(p.classifier.findData("global"))
    assert p.scheme_now()[0] is None  # nothing copied yet
    p.copy_global.click()
    scheme, problem = p.scheme_now()
    assert problem == "" and scheme.stability == window._stability_in_use()
    p.set_scheme(tv.TauScheme())
    assert p.scheme_now()[0] == tv.TauScheme()
