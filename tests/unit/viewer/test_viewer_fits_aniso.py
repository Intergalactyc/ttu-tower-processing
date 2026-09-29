import numpy as np
import pandas as pd
import pytest

from ttu_tower.io.store import read_table
from ttu_tower.physics.turbulence import _classify_anisotropy_state
from ttu_tower.viewer import anisotropy, curvefits, distfits, slotdata, windrose
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.run import RunHandle
from ttu_tower.viewer.timeline import Curve, joined

from viewer_fixtures import FULL_BOOMS


def test_map_classes_match_the_pipelines_classifier():
    rng = np.random.default_rng(3)
    for _ in range(400):
        l3 = rng.uniform(-1 / 3, 0)
        l2 = rng.uniform(l3, (1 - 3 * l3) / 2 - 1e-9) if rng.random() < 0.9 else l3
        l1 = -l2 - l3
        evals = np.array(sorted([l1, l2, l3], reverse=True))
        x, y = anisotropy.barycentric_xy(evals[1], evals[2])
        assert anisotropy.classify(x, y)[0] == _classify_anisotropy_state(evals, anisotropy.EDGE_K)


def test_map_corners():
    assert anisotropy.barycentric_xy(-1 / 3, -1 / 3) == pytest.approx((0.0, 0.0))  # one component
    assert anisotropy.barycentric_xy(1 / 6, -1 / 3) == pytest.approx((1.0, 0.0))  # two
    assert anisotropy.barycentric_xy(0.0, 0.0) == pytest.approx((0.5, np.sqrt(3) / 2))  # isotropic


@pytest.mark.parametrize("name, truth, make", [
    ("linear", {"a": 1.5, "b": -0.7}, lambda x, p: p["a"] + p["b"] * x),
    ("linear through origin", {"b": 2.0}, lambda x, p: p["b"] * x),
    ("power law", {"a": 3.0, "b": 0.25}, lambda x, p: p["a"] * x ** p["b"]),
    ("logarithmic", {"a": 2.0, "b": 0.5}, lambda x, p: p["a"] + p["b"] * np.log(x)),
    ("exponential", {"A": 2.0, "B": 1.3}, lambda x, p: p["A"] * p["B"] ** x),
    ("double linear", {"A": 2.0, "B": 0.5, "C": 1.0, "D": 0.3},
     lambda x, p: curvefits._double_linear_model(x, p["A"], p["B"], p["C"], p["D"])),
])
def test_curve_fits_recover_exact_parameters(name, truth, make):
    x = np.linspace(0.5, 6.0, 80)
    result = curvefits.fit(name, x, make(x, truth))
    for key, value in truth.items():
        assert result.params[key] == pytest.approx(value, rel=1e-4), key
    assert result.rmse < 1e-6 and result.n == x.size


def test_fit_options_limit_and_bin_before_fitting():
    x = np.linspace(0, 10, 1001)
    y = 2 * x + np.where(x > 9, 100.0, 0.0)  # outliers above the 90th percentile of x
    clean = curvefits.fit("linear", x, y, curvefits.FitOptions(pct_x=(0, 89)))
    assert clean.params["b"] == pytest.approx(2.0)
    binned = curvefits.fit("linear", x, y, curvefits.FitOptions(lim_x=(0, 8), bin_count=10, bin_method="median"))
    assert binned.n == 10 and binned.params["b"] == pytest.approx(2.0, rel=1e-3)


def test_correlation_and_sign_agreement():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    stats = curvefits.correlation(x, 3 * x + 1e-3 * rng.normal(size=500))
    assert stats["N"] == 500 and stats["r"] > 0.999 and stats["ρ"] > 0.999 and stats["κ"] > 0.99
    assert curvefits.cohens_kappa(x, 3 * x) == pytest.approx(1.0) and curvefits.cohens_kappa(x, -x) == pytest.approx(-1.0, abs=1e-3)


def test_distribution_fits_recover_their_parameters():
    rng = np.random.default_rng(2)
    normal = distfits.fit("normal", rng.normal(5.0, 2.0, 20_000))
    assert normal.params["μ"] == pytest.approx(5.0, abs=0.05) and normal.params["σ"] == pytest.approx(2.0, abs=0.05)
    weibull = distfits.fit("Weibull", 8.0 * rng.weibull(2.2, 20_000))
    assert weibull.params["k"] == pytest.approx(2.2, rel=0.03) and weibull.params["λ"] == pytest.approx(8.0, rel=0.02)
    assert weibull.ks < 0.02
    wd = np.degrees(rng.vonmises(np.radians(250.0), 4.0, 20_000)) % 360
    vm = distfits.fit(distfits.CIRCULAR, wd)
    assert vm.params["mean [deg]"] == pytest.approx(250.0, abs=1.0) and vm.params["κ"] == pytest.approx(4.0, rel=0.05)
    grid = np.linspace(0, 360, 3601)
    assert np.trapezoid(vm.pdf(grid), grid) == pytest.approx(1.0, abs=1e-3)
    assert distfits.available(np.array([-1.0, 2.0]), False) == ["normal"]
    assert distfits.fit("gamma", np.array([1.0, 2.0])) is None


def test_joined_keeps_the_common_slots():
    a = Curve(slots=np.array([1, 2, 3, 5]), y=np.array([10.0, 20.0, 30.0, 50.0]))
    b = Curve(slots=np.array([2, 3, 4, 5]), y=np.array([0.2, 0.3, 0.4, 0.5]))
    slots, ya, yb = joined(a, b)
    assert list(slots) == [2, 3, 5] and list(ya) == [20.0, 30.0, 50.0] and list(yb) == [0.2, 0.3, 0.5]


@pytest.mark.slow
def test_slot_states_and_the_band_agree_with_the_stored_classes(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    labels = read_table(run.run_dir / "secondary" / "data" / "boom_labels")
    labels = labels[(labels["variant"].astype(str) == "mrd") & (labels["label"].astype(str) == "aniso_class")]
    checked = 0
    for k in sorted(set(labels["slot"]))[::4]:
        states = anisotropy.slot_states(slotdata.load_across(index, k, FULL_BOOMS), "mrd").set_index("boom")
        for r in labels[labels["slot"] == k].itertuples(index=False):
            if pd.isna(r.value):
                continue
            assert states.loc[r.boom, "state"] == r.value, (k, r.boom)
            checked += 1
    assert checked > 5
    slots, image = anisotropy.class_image(run, index, "mrd", FULL_BOOMS)
    final = read_table(run.run_dir / "tertiary" / "data" / "boom_labels_final")
    final = final[(final["variant"].astype(str) == "mrd") & final["value"].notna()]
    for r in final.itertuples(index=False):
        assert anisotropy.CASES[image[FULL_BOOMS.index(r.boom), r.slot - slots[0]]] == r.value
    assert (image >= 0).sum() == len(final)


def test_rose_bins_by_where_the_wind_comes_from_and_its_speed():
    ws = np.array([1.0, 3.0, 5.0, 20.0, 1.0])
    wd = np.array([350.0, 10.0, 90.0, 180.0, 359.9])
    table = windrose.rose(ws, wd, sectors=4)
    assert table.shape == (4, len(windrose.SPEED_EDGES) - 1)
    assert table.sum() == pytest.approx(100.0)
    assert table[0, 0] == pytest.approx(40.0) and table[0, 1] == pytest.approx(20.0)  # north, both sides of 0
    assert table[1, 2] == pytest.approx(20.0) and table[2, -1] == pytest.approx(20.0)  # east; the open top bin


@pytest.mark.slow
def test_unfiltered_tower_wind_is_the_first_layer_means(full_run):
    from viewer_fixtures import SPIKY_BOOM, SPIKY_SLOT
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    final = windrose.tower_wind(run, index, SPIKY_BOOM, filtered=True)
    every = windrose.tower_wind(run, index, SPIKY_BOOM, filtered=False)
    assert SPIKY_SLOT not in final.slots and SPIKY_SLOT in every.slots
    assert set(final.slots) < set(every.slots)
    l1 = read_table(run.run_dir / "primary" / "data" / "slot_qc")
    l1 = l1[(l1["boom"] == SPIKY_BOOM) & (l1["stat"].astype(str) == "mean_l1")]
    for var, got in (("ws", every.ws), ("wd", every.wd)):
        rows = l1[l1["variable"].astype(str) == var].set_index("slot")["value"]
        np.testing.assert_array_equal(got, rows.reindex(every.slots).to_numpy())
    assert windrose.mesonet_wind(run, index) is None
    assert windrose.parse_bearings("105, 170; 400") == [105.0, 170.0, 40.0]
    with pytest.raises(ValueError):
        windrose.parse_bearings("north")


def test_alpha_ri_fit_recovers_its_branches_and_honours_its_settings():
    alpha0, (a_s, b_s), (a_u, b_u), ri_c = 0.15, (5.0, 0.6), (-2.0, -0.3), 0.25
    x = np.concatenate([np.zeros(5), np.linspace(0.06, 0.24, 60), np.linspace(-2.0, -0.06, 60), np.linspace(0.3, 1, 10)])
    crit = alpha0 * (1 + a_s * ri_c) ** b_s
    y = np.where(x >= ri_c, crit, np.where(x > 0, alpha0 * (1 + a_s * x) ** b_s, alpha0 * (1 + a_u * x) ** b_u))
    y[x == 0] = alpha0
    result = curvefits.fit(curvefits.ALPHA_RI, x, y)
    for key, value in (("alpha0", alpha0), ("a_stable", a_s), ("b_stable", b_s), ("a_unstable", a_u),
                       ("b_unstable", b_u), ("alpha_critical", crit)):
        assert result.params[key] == pytest.approx(value, rel=1e-4), key
    assert result.params["mean_high_ri"] == pytest.approx(crit)
    shifted = curvefits.fit(curvefits.ALPHA_RI, x, y, ri_critical=0.2)
    assert shifted.func(np.array([0.22]))[0] == pytest.approx(shifted.params["alpha_critical"])  # held above Ri_c
    with pytest.raises(ValueError, match="stable"):
        curvefits.fit(curvefits.ALPHA_RI, x[x <= 0], y[x <= 0])
