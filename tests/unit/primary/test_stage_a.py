import numpy as np
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.constants import BOOMS
from ttu_tower.math.polar import vector_to_bearing
from ttu_tower.primary.stage_a import (
    apply_bounds,
    couple_triplet,
    rotate_to_earth,
    stage_a,
    tilt_correct,
    tilt_matrix,
    to_si,
)


def _qc_cfg():
    raw = {
        "paths": {"raw_dirs": ["/data"]},
        "files": {"bad_records": [[1, 2]]},
    }
    return config_from_dict(raw, default_tag="t").qc


def test_to_si_conversions():
    raw = {
        "u": np.array([2.23694]), "v": np.array([0.0]), "w": np.array([0.0]),
        "ts": np.array([32.0]), "t": np.array([212.0]),
        "rh": np.array([50.0]), "p": np.array([29.9213]),
    }
    si = to_si(raw)
    assert si["u"][0] == pytest.approx(1.0, abs=1e-9)
    assert si["ts"][0] == pytest.approx(273.15, abs=1e-9)
    assert si["t"][0] == pytest.approx(373.15, abs=1e-9)
    assert si["rh"][0] == pytest.approx(0.5, abs=1e-9)
    assert si["p"][0] == pytest.approx(101.325, abs=1e-2)


def test_apply_bounds_removes_out_of_range():
    cfg = _qc_cfg().bounds
    si = {
        "u": np.array([0.0, 100.0]), "v": np.array([0.0, 0.0]), "w": np.array([0.0, 0.0]),
        "ts": np.array([280.0, 400.0]), "t": np.array([280.0, 280.0]),
        "rh": np.array([0.5, 0.5]), "p": np.array([90.0, 90.0]),
    }
    out, removed = apply_bounds(si, cfg)
    assert np.isnan(out["ts"][1])
    assert not np.isnan(out["ts"][0])
    assert removed["ts"].tolist() == [False, True]


def test_apply_bounds_sonic_violation_removes_all_three():
    cfg = _qc_cfg().bounds
    si = {
        "u": np.array([0.0, 1000.0]), "v": np.array([0.0, 1.0]), "w": np.array([0.0, 1.0]),
        "ts": np.array([280.0, 280.0]), "t": np.array([280.0, 280.0]),
        "rh": np.array([0.5, 0.5]), "p": np.array([90.0, 90.0]),
    }
    out, removed = apply_bounds(si, cfg)
    assert np.isnan(out["u"][1]) and np.isnan(out["v"][1]) and np.isnan(out["w"][1])
    assert removed["u"][1] and removed["v"][1] and removed["w"][1]
    assert not np.isnan(out["u"][0])


def test_couple_triplet():
    u = np.array([1.0, np.nan, 3.0])
    v = np.array([1.0, 2.0, np.nan])
    w = np.array([1.0, 2.0, 3.0])
    uc, vc, wc = couple_triplet(u, v, w)
    assert not np.isnan(uc[0]) and not np.isnan(vc[0]) and not np.isnan(wc[0])
    assert np.isnan(uc[1]) and np.isnan(vc[1]) and np.isnan(wc[1])
    assert np.isnan(uc[2]) and np.isnan(vc[2]) and np.isnan(wc[2])


def test_tilt_matrix_orthonormal_every_boom():
    for b in BOOMS:
        r = tilt_matrix(b)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)


def test_tilt_matrix_boom_4_is_identity():
    assert np.allclose(tilt_matrix(4), np.eye(3), atol=1e-12)


def test_tilt_correct_matches_matrix():
    u, v, w = np.array([1.0]), np.array([2.0]), np.array([3.0])
    boom = 7
    un, vn, wn = tilt_correct(u, v, w, boom)
    r = tilt_matrix(boom)
    expected = r @ np.array([1.0, 2.0, 3.0])
    assert np.allclose([un[0], vn[0], wn[0]], expected, atol=1e-12)


def test_rotate_to_earth():
    ue, vn = rotate_to_earth(np.array([3.0]), np.array([5.0]))
    assert ue[0] == 5.0
    assert vn[0] == -3.0


def test_stage_a_full_chain_wind_direction():
    # Raw u is already the north component of the wind's FROM-vector, so a
    # pure north-raw wind (u=1, v=0) at boom 4 (tilt = identity) is a wind
    # FROM the north (FROM-bearing 0): ue=v_raw=0, vn=-u_raw=-1.
    n = 5
    raw = {
        "u": np.full(n, 2.23694), "v": np.zeros(n), "w": np.zeros(n),
        "ts": np.full(n, 32.0), "t": np.full(n, 32.0),
        "rh": np.full(n, 50.0), "p": np.full(n, 29.9213),
    }
    cfg = _qc_cfg()
    out = stage_a(raw, boom=4, cfg=cfg)
    assert np.allclose(out.ue, 0.0, atol=1e-9)
    assert np.allclose(out.vn, -1.0, atol=1e-9)
    _, wd = vector_to_bearing(out.ue, out.vn)
    assert np.allclose(wd, 0.0, atol=1e-7)  # -u_raw points south (vn=-1) -> wind blows toward south -> FROM north (0)


def test_wind_direction_end_to_end_matches_independent_ground_truth():
    # Raw sonic (N, W) FROM-vector samples, non-axis-aligned and non-uniform,
    # through the full stage_a chain at boom 4 (tilt = identity), so this
    # isolates the units/rotation chain the way the old end-to-end test did.
    u_raw_mph = np.array([1.2, 0.8, -0.3, 1.5, -0.6]) * 2.23694  # North component
    v_raw_mph = np.array([0.4, -0.9, 1.1, 0.2, 0.7]) * 2.23694  # West component
    n = u_raw_mph.size
    raw = {
        "u": u_raw_mph, "v": v_raw_mph, "w": np.zeros(n),
        "ts": np.full(n, 32.0), "t": np.full(n, 32.0),
        "rh": np.full(n, 50.0), "p": np.full(n, 29.9213),
    }
    out = stage_a(raw, boom=4, cfg=_qc_cfg())

    # Ground truth, computed independently: raw (u, v) are the FROM-vector's
    # (North, West) projections, so true (East, North) blows-toward =
    # (West_raw, -North_raw), vector-averaged, then TOWARD -> FROM (+180).
    u_raw_si, v_raw_si = u_raw_mph / 2.23694, v_raw_mph / 2.23694
    east, north = v_raw_si, -u_raw_si
    toward_true = np.degrees(np.arctan2(east.mean(), north.mean())) % 360
    from_true = (toward_true + 180) % 360

    _, wd = vector_to_bearing(out.ue.mean(), out.vn.mean())
    assert wd == pytest.approx(from_true, abs=1e-9)
