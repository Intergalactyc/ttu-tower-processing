import numpy as np
import pytest

from ttu_tower.viewer import curvefits, expression


def test_a_formula_is_arithmetic_on_x_and_nothing_else():
    f = expression.compile_expression("y = 0.5*x^2 - 3·x + ln(e) + sqrt(abs(-4))")
    assert f(np.array([0.0, 2.0])) == pytest.approx([3.0, -1.0])
    assert expression.compile_expression("2")(np.zeros(3)).shape == (3,)  # a constant still gives one y per x
    for bad, why in (("2x", "products need *"), ("x.real", "isn't arithmetic"), ("__import__('os')", "functions"),
                     ("foo + 1", "unknown name"), ("exp(x, 2)", "one value"), ("x if x else 1", "arithmetic"),
                     ("'a'", "isn't a number"), ("", "empty"), ("x < 1", "arithmetic")):
        with pytest.raises(ValueError, match=why):
            expression.compile_expression(bad)


def test_pieces_cover_their_ranges_and_the_first_wins_where_they_overlap():
    pieces = [(expression.compile_expression("1"), -np.inf, 0.0), (expression.compile_expression("x"), 0.0, 2.0),
              (expression.compile_expression("10"), 1.0, np.inf)]
    f = expression.piecewise(pieces)
    assert f(np.array([-1.0, 0.5, 1.5, 3.0])) == pytest.approx([1.0, 0.5, 1.5, 10.0])
    assert list(expression.piece_of(np.array([-1.0, 1.5, 3.0]), [(lo, hi) for _, lo, hi in pieces])) == [0, 1, 2]
    assert expression.parse_bound("2*pi", 0.0) == pytest.approx(2 * np.pi)
    assert expression.parse_bound("", -np.inf) == -np.inf
    with pytest.raises(ValueError):
        expression.parse_bound("x", 0.0)


@pytest.mark.parametrize("name", [n for n in curvefits.FITS if n != curvefits.ALPHA_RI])
def test_every_fit_writes_itself_as_formulas_that_draw_the_same_curve(name):
    rng = np.random.default_rng(3)
    x = np.geomspace(0.2, 20.0, 300) if name != "ζ(Ri) (stability parameter)" else np.linspace(-2.0, 0.2, 300)
    y = 1.5 * x ** 0.6 + rng.normal(0.0, 0.05, x.size) if name != "ζ(Ri) (stability parameter)" else (
        np.where(x < 0, 3 * x, 3 * x / (1 + 2 * x)) + rng.normal(0.0, 0.02, x.size))
    res = curvefits.fit(name, x, y)
    pieces = [(expression.compile_expression(f), expression.parse_bound(lo, -np.inf), expression.parse_bound(hi, np.inf))
              for f, lo, hi in curvefits.fit_formula(name, res.params)]
    at = x[::17]
    assert expression.piecewise(pieces)(at) == pytest.approx(res.func(at), rel=1e-4, abs=1e-6)


def test_the_alpha_ri_fit_writes_its_three_pieces():
    rng = np.random.default_rng(4)
    x = np.linspace(-1.0, 0.6, 400)
    y = np.where(x < 0, 0.1 * (1 - 0.5 * x) ** -0.4, 0.1 * (1 + 4 * np.minimum(x, 0.25)) ** 0.5)
    y = y + rng.normal(0.0, 0.002, x.size)
    res = curvefits.fit(curvefits.ALPHA_RI, x, y, ri_critical=0.25)
    formulas = curvefits.fit_formula(curvefits.ALPHA_RI, res.params, {"ri_critical": 0.25})
    assert [(lo, hi) for _, lo, hi in formulas] == [("", "0"), ("0", "0.25"), ("0.25", "")]
    pieces = [(expression.compile_expression(f), expression.parse_bound(lo, -np.inf), expression.parse_bound(hi, np.inf))
              for f, lo, hi in formulas]
    at = np.array([-0.8, -0.2, 0.1, 0.2, 0.4])
    assert expression.piecewise(pieces)(at) == pytest.approx(res.func(at), rel=1e-4)


def test_a_formula_can_use_the_variables_it_is_offered():
    f = expression.compile_expression("x*zA + zB", ("zA", "zB"))
    assert f.names == {"zA", "zB"} and f(np.array([2.0]), zA=3.0, zB=1.0) == pytest.approx([7.0])
    assert expression.compile_expression("2*x", ("zA",)).names == set()
    with pytest.raises(ValueError, match="zA"):  # the message lists what may be used
        expression.compile_expression("zC*x", ("zA",))
    with pytest.raises(ValueError, match="unknown name"):
        expression.compile_expression("zA*x")  # not offered
    pieces = [(expression.compile_expression("zA", ("zA",)), -np.inf, 1.0), (expression.compile_expression("-zA", ("zA",)), 1.0, np.inf)]
    g = expression.piecewise(pieces)
    assert g.names == {"zA"}
    assert g(np.array([0.0, 2.0, 3.0]), zA=np.array([5.0, 6.0, 7.0])) == pytest.approx([5.0, -6.0, -7.0])  # per point
