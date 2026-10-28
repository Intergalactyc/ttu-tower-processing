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
