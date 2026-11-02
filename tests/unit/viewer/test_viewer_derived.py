import numpy as np
import pytest

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer import derived
from ttu_tower.viewer.catalog import Catalog, Quantity
from ttu_tower.viewer.derived import Definition, DerivedStore, Variable


def _q(key, kind="boom", variants=("none",), pairs=(), categorical=False):
    return Quantity(key=key, table="boom_final", variable=key, stat=None, kind=kind, variants=variants, label=key,
                    unit="", group="Wind", pairs=pairs, categorical=categorical)


def _catalog():
    cat = Catalog()
    for q in (_q("ws"), _q("ustar", variants=("none", "mrd")), _q("meso", kind="slot"),
              _q("rib", kind="pair", pairs=((2, 4), (4, 6))), _q("veer", kind="pair", pairs=((4, 6),)),
              _q("class", categorical=True)):
        cat.add(q)
    return cat


def test_a_definition_resolves_to_a_quantity_of_the_right_kind():
    cat = _catalog()
    q = derived.resolve(Definition("ratio", "x/y", (Variable("x", "ws"), Variable("y", "ustar", "mrd")), "1"), cat)
    assert (q.key, q.kind, q.group, q.unit, q.variants) == ("derived|ratio|", "boom", "Derived", "1", ("none",))
    q = derived.resolve(Definition("m", "x - y", (Variable("x", "meso"), Variable("y", "ws", member=5))), cat)
    assert q.kind == "slot"  # a fixed boom is one series
    q = derived.resolve(Definition("p", "x*y", (Variable("x", "rib"), Variable("y", "veer"))), cat)
    assert q.kind == "pair" and q.pairs == ((4, 6),)  # the pairs both have
    assert derived.resolve(Definition("h", "x/z", (Variable("x", "ws"),)), cat).kind == "boom"  # z: per boom only
    bad = [
        (Definition("2bad", "x", (Variable("x", "ws"),)), "name"),
        (Definition("r", "x", (Variable("z", "ws"),)), "can't name"),
        (Definition("r", "x", (Variable("x", "ws"), Variable("x", "ustar"))), "share a name"),
        (Definition("r", "x", (Variable("x", "nope"),)), "isn't in this run"),
        (Definition("r", "x", (Variable("x", "class"),)), "can't be used"),
        (Definition("r", "x", (Variable("x", "ws", "mrd"),)), "no mrd variant"),
        (Definition("r", "x*y", (Variable("x", "ws"), Variable("y", "rib"))), "mix"),
        (Definition("r", "x/z", (Variable("x", "meso"),)), "unknown name 'z'"),
        (Definition("r", "x y", (Variable("x", "ws"),)), "can't read"),
    ]
    for d, why in bad:
        with pytest.raises(ValueError, match=why):
            derived.resolve(d, cat)


def test_the_site_constants_are_the_pipelines_at_each_members_height():
    from ttu_tower.constants import SITE_ELEVATION, SITE_LATITUDE
    from ttu_tower.physics.earth import local_gravity

    top = derived.site_values("boom", 10)
    assert top["g"] == pytest.approx(local_gravity(SITE_LATITUDE, SITE_ELEVATION + 200.0))
    assert top["f"] == pytest.approx(2 * 7.2921e-5 * np.sin(np.deg2rad(SITE_LATITUDE)))
    assert top["kappa"] == 0.41 and top["cp"] == pytest.approx(1004.69, abs=0.01) and top["Rd"] == pytest.approx(287.006, abs=0.001)
    pair = derived.site_values("pair", (2, 4))["g"]
    assert pair == pytest.approx(local_gravity(SITE_LATITUDE, SITE_ELEVATION + (2.4 + 10.1) / 2))
    assert derived.site_values("slot", None)["g"] > pair > top["g"]  # less with height
    cat = _catalog()
    assert derived.resolve(Definition("n", "g/x*f/kappa", (Variable("x", "ws"),)), cat).kind == "boom"
    assert derived.resolve(Definition("s", "g*2", (Variable("g", "meso"),)), cat).kind == "slot"  # a variable may be g
    with pytest.raises(ValueError, match="unknown name 'G'"):
        derived.resolve(Definition("r", "G*x", (Variable("x", "ws"),)), cat)


def test_a_derived_quantity_may_use_others_but_never_itself(tmp_path):
    a = Definition("a", "x/y", (Variable("x", "ws"), Variable("y", "ustar", "mrd")))
    b = Definition("b", "2*p", (Variable("p", "derived|a|"),))
    c = Definition("c", "p + q", (Variable("p", "derived|b|"), Variable("q", "meso")))
    cat = _catalog()
    assert derived.add_to(cat, [c, b, a]) == []  # each after those it uses, whatever the order kept
    assert cat["derived|c|"].kind == "boom" and derived.uses(c) == {"b"}
    assert [d.name for d in derived.in_order([c, b, a])[0]] == ["a", "b", "c"]

    loop = Definition("a", "p", (Variable("p", "derived|b|"),))  # a → b → a
    me = Definition("m", "p", (Variable("p", "derived|m|"),))
    notes = derived.add_to(cat, [loop, b, c, me])
    assert "a: it uses itself (a → b → a)" in notes and "b: it uses itself (b → a → b)" in notes
    assert "m: it uses itself (m → m)" in notes
    assert any(n.startswith("c: p: the derived quantity b isn't defined") for n in notes)  # c relies on the loop

    store = DerivedStore.open(tmp_path / "derived.json")
    for d in (a, b, c):
        store.put(d)
    store.put(Definition("ratio", a.formula, a.variables), replacing="a")  # renamed: b follows
    again = {d.name: d for d in DerivedStore.open(tmp_path / "derived.json").items}
    assert again["b"].variables[0].key == "derived|ratio|" and derived.add_to(_catalog(), again.values()) == []


def test_definitions_are_kept_and_replaced_by_name(tmp_path):
    store = DerivedStore.open(tmp_path / "derived.json")
    a = Definition("a", "x", (Variable("x", "rib", member=(2, 4)),), "m")
    store.put(a)
    store.put(Definition("b", "2*x", (Variable("x", "ws"),)))
    store.put(Definition("a2", "x", (Variable("x", "ws"),)), replacing="a")  # renamed
    again = DerivedStore.open(tmp_path / "derived.json")
    assert [d.name for d in again.items] == ["b", "a2"]
    assert Definition.from_dict(a.to_dict()) == a  # a pair member survives JSON
    again.remove("b")
    assert [d.name for d in DerivedStore.open(tmp_path / "derived.json").items] == ["a2"]
    cat = _catalog()
    notes = derived.add_to(cat, [a, Definition("bad", "x", (Variable("x", "nope"),))])
    assert "derived|a|" in cat.quantities and notes == ["bad: x: nope isn't in this run"]
    derived.add_to(cat, [])
    assert not any(q.table == "derived" for q in cat)  # replaced, not piled up


pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from ttu_tower.viewer import timeline  # noqa: E402
from ttu_tower.viewer.run import RunHandle  # noqa: E402
from ttu_tower.viewer.ui.app import configure_pyqtgraph  # noqa: E402
from ttu_tower.viewer.ui.main_window import MainWindow  # noqa: E402

from viewer_fixtures import FULL_BOOMS  # noqa: E402


def _idle(window):
    return not window.runner._callbacks and window.runner.pool.activeThreadCount() == 0


@pytest.fixture
def window(qtbot, full_run):
    configure_pyqtgraph()
    w = MainWindow(RunHandle.from_dir(full_run.run_dir))
    w.derived_store = w.derived_panel.store = DerivedStore()  # in memory only
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    qtbot.waitUntil(lambda: _idle(w) and w.datas[0] is not None, timeout=60_000)
    yield w
    w.runner.wait(30_000)


def _define(panel, name, formula, variables):
    panel.new()
    panel.name.setText(name)
    panel.formula.setText(formula)
    for row in list(panel.rows):
        panel._remove_row(row)
    for vname, key, variant, member in variables:
        row = panel.add_row(vname)
        row.quantity.setCurrentIndex(row.quantity.findData(key))
        row.sync(variant, member)
    return panel.save()


@pytest.mark.slow
def test_a_derived_quantity_is_defined_in_controls_and_shown_in_a_panel(window, qtbot):
    panel = window.derived_panel
    assert _define(panel, "u_ustar_ratio", "x/y", [("x", "boom_final|ws|mean", "none", None),
                                                 ("y", "boom_final|ustar|", "mrd", None)]), panel.status.text()
    q = window.catalog["derived|u_ustar_ratio|"]
    assert q.kind == "boom" and q.group == "Derived" and panel.list.count() == 1
    window.panels[0].select(q.key, None, tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == q.key, timeout=60_000)
    ws = timeline.load(window.run, window.index, window.catalog["boom_final|ws|mean"], "none", FULL_BOOMS)
    ustar = timeline.load(window.run, window.index, window.catalog["boom_final|ustar|"], "mrd", FULL_BOOMS)
    for b in FULL_BOOMS:
        slots, a, c = timeline.joined(ws.curves[b], ustar.curves[b])
        got = window.datas[0].curves[b]
        assert np.array_equal(got.slots, slots) and np.allclose(got.y, a / c, equal_nan=True)

    b = FULL_BOOMS[-1]  # an edit: each boom's ws over the top boom's u*, times the height
    panel.open(panel.store.items[0])
    panel.formula.setText("x/y*z")
    panel.rows[1].sync("mrd", b)
    assert panel.save(), panel.status.text()
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.extra != q.extra, timeout=60_000)
    low = FULL_BOOMS[0]
    slots, a, c = timeline.joined(ws.curves[low], ustar.curves[b])
    got = window.datas[0].curves[low]
    assert np.array_equal(got.slots, slots) and np.allclose(got.y, a / c * HEIGHTS[low], equal_nan=True)

    panel.delete()
    assert "derived|u_ustar_ratio|" not in window.catalog.quantities and panel.list.count() == 0
    assert "isn't in this run" in window.statusBar().currentMessage()


@pytest.mark.slow
def test_a_formula_takes_g_at_each_booms_height_and_a_variable_hides_a_constant(window, qtbot):
    panel = window.derived_panel
    assert "g: local gravity" in panel.formula.toolTip()
    assert _define(panel, "gws", "x*0 + g/f", [("x", "boom_final|ws|mean", "none", None)]), panel.status.text()
    window.panels[0].select("derived|gws|", None, tuple(FULL_BOOMS))
    qtbot.waitUntil(lambda: _idle(window) and window.datas[0].quantity.key == "derived|gws|", timeout=60_000)
    for b in FULL_BOOMS:
        site = derived.site_values("boom", b)
        y = window.datas[0].curves[b].y
        assert y.size and np.allclose(y[np.isfinite(y)], site["g"] / site["f"])
    assert _define(panel, "fws", "f", [("f", "boom_final|ws|mean", "none", None)]), panel.status.text()
    shown = derived.load(window.run, window.index, window.catalog["derived|fws|"], FULL_BOOMS)
    ws = timeline.load(window.run, window.index, window.catalog["boom_final|ws|mean"], "none", FULL_BOOMS)
    assert np.allclose(shown[FULL_BOOMS[0]][1], ws.curves[FULL_BOOMS[0]].y, equal_nan=True)  # the variable, not f


@pytest.mark.slow
def test_a_derived_quantity_built_on_another_follows_it(window, qtbot):
    panel = window.derived_panel
    assert _define(panel, "ratio", "x/y", [("x", "boom_final|ws|mean", "none", None),
                                           ("y", "boom_final|ustar|", "mrd", None)]), panel.status.text()
    assert _define(panel, "twice", "2*r", [("r", "derived|ratio|", "none", None)]), panel.status.text()
    assert panel.rows[0].quantity.findData("derived|twice|") < 0  # not itself
    ratio = timeline.load(window.run, window.index, window.catalog["derived|ratio|"], "none", FULL_BOOMS)
    twice = timeline.load(window.run, window.index, window.catalog["derived|twice|"], "none", FULL_BOOMS)
    b = FULL_BOOMS[0]
    assert twice.curves[b].y.size and np.allclose(twice.curves[b].y, 2 * ratio.curves[b].y, equal_nan=True)

    panel.open(next(d for d in panel.store.items if d.name == "ratio"))  # ratio on twice: a loop
    panel.rows[0].quantity.setCurrentIndex(panel.rows[0].quantity.findData("derived|twice|"))
    panel.rows[0].sync("none", None)
    assert not panel.save() and "would use itself (ratio → twice → ratio)" in panel.status.text()

    panel.open(next(d for d in panel.store.items if d.name == "ratio"))
    panel.delete()
    assert "twice used it" in panel.status.text() and "derived|twice|" not in window.catalog.quantities


@pytest.mark.slow
def test_duplicating_a_derived_quantity_keeps_its_definition_under_a_new_name(window):
    panel = window.derived_panel
    assert _define(panel, "lb", "x/sqrt(abs(y))", [("x", "boom_final|sigma_w|", "mrd", None),
                                                    ("y", "boom_final|n2|", "none", None)]), panel.status.text()
    panel.duplicate()
    assert panel.editing is None and panel.name.text() == "lb_2" and "copy of lb" in panel.status.text()
    panel.formula.setText("x/sqrt(abs(y))*z")
    assert panel.save(), panel.status.text()
    names = sorted(d.name for d in panel.store.items)
    assert names == ["lb", "lb_2"]
    copy = next(d for d in panel.store.items if d.name == "lb_2")
    assert copy.variables == panel.store.items[0].variables and copy.formula.endswith("*z")
    panel.open(panel.store.items[0])
    panel.duplicate()
    assert panel.name.text() == "lb_3"
