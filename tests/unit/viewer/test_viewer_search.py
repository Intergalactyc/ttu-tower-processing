import os

import pytest

from ttu_tower.viewer import search
from ttu_tower.viewer.catalog import Quantity, _describe

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_VARIABLES = [("vpt", None), ("pt", None), ("vpts", "mean"), ("wvpts", None), ("buoyancy_flux", None),
              ("wpt", None), ("theta_star", None), ("dvpt_dz", None), ("its_corr_vpts", None), ("ustar", None),
              ("sigma_w", None), ("ws", "mean"), ("n2", None), ("phi_h_v", None), ("uw", None), ("t", None)]


def _quantity(variable, stat=None):
    group, label, unit = _describe(variable, stat)
    return Quantity(key=f"boom_final|{variable}|{stat or ''}", table="boom_final", variable=variable, stat=stat,
                    kind="boom", variants=("mrd",), label=label, unit=unit, group=group)


QUANTITIES = [_quantity(v, s) for v, s in _VARIABLES]


def _found(text):
    query = search.Query(text)
    return {q.variable for q in QUANTITIES if query.matches(search.quantity_terms(q))}


def test_long_names_find_symbols():
    assert _found("buoyancy flux") == {"wvpts", "buoyancy_flux"}
    assert _found("virtual potential temperature") >= {"vpt", "vpts", "wvpts", "dvpt_dz", "its_corr_vpts", "phi_h_v"}
    assert _found("friction velocity") == {"ustar"}
    assert _found("brunt vaisala") == _found("Brunt-Väisälä") == {"n2"}


def test_greek_letters_are_spelled_out_and_separators_ignored():
    assert {"vpt", "pt", "vpts", "wvpts", "theta_star", "dvpt_dz"} <= _found("theta")
    assert "uw" not in _found("theta")
    assert {"vpt", "vpts", "wvpts"} <= _found("theta_v") == _found("θv")
    assert _found("sigma w") >= {"sigma_w"} and _found("σw") >= {"sigma_w"}
    assert _found("u*") == _found("ustar") == {"ustar"}
    assert "n2" in _found("N2")


def test_abbreviations_and_empty_query():
    assert {"vpt", "vpts", "wvpts", "dvpt_dz", "phi_h_v"} <= _found("vpt")
    assert "pt" not in _found("vpt")
    assert _found("  ") == {v for v, _ in _VARIABLES}


def test_search_combo_lists_only_matches_and_picks_one(qtbot):
    from PySide6.QtCore import QModelIndex, Qt
    from ttu_tower.viewer.ui.search_combo import SearchCombo

    combo = SearchCombo()
    qtbot.addWidget(combo)
    combo.fill_quantities(QUANTITIES)
    combo.show()
    qtbot.keyClicks(combo.lineEdit(), "buoyancy flux")
    model = combo._completer.completionModel()
    shown = [model.index(r, 0).data(Qt.ItemDataRole.UserRole) for r in range(model.rowCount())]
    assert set(shown) == {q.key for q in QUANTITIES if q.variable in ("wvpts", "buoyancy_flux")}
    combo._completer.activated[QModelIndex].emit(model.index(1, 0))
    assert combo.currentData() == shown[1]
    assert combo.currentText() == combo.itemText(combo.currentIndex())
