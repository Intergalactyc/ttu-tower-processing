import numpy as np
import pandas as pd
import pytest

from ttu_tower.post.classify import ClassifyError, IntervalClassifier, stability_classes

_STABILITY_CLASSES = [
    ("strongly unstable", "(-inf,-0.05)"),
    ("unstable", "[-0.05,-0.01)"),
    ("neutral", "[-0.01,0.01)"),
    ("stable", "[0.01,0.1)"),
    ("strongly stable", "[0.1,inf)"),
]


def test_classifies_interior_values():
    clf = IntervalClassifier(_STABILITY_CLASSES)
    result = clf.classify([-1.0, -0.02, 0.0, 0.05, 1.0])
    assert list(result) == ["strongly unstable", "unstable", "neutral", "stable", "strongly stable"]


@pytest.mark.parametrize("value,expected", [
    (-0.05, "unstable"),        # left-closed bound of [-0.05,-0.01) is included here...
    (-0.05000001, "strongly unstable"),  # ...and excluded from the open bound just below it
    (-0.01, "neutral"),         # left-closed bound of [-0.01,0.01)
    (0.01, "stable"),           # left-closed bound of [0.01,0.1)
    (0.1, "strongly stable"),   # left-closed bound of [0.1,inf)
])
def test_classifies_edge_values_by_inclusive_bound(value, expected):
    clf = IntervalClassifier(_STABILITY_CLASSES)
    result = clf.classify([value])
    assert result[0] == expected


def test_nan_classifies_to_none():
    clf = IntervalClassifier(_STABILITY_CLASSES)
    result = clf.classify([np.nan])
    assert result[0] is None


def test_value_matching_no_interval_classifies_to_none():
    clf = IntervalClassifier([("low", "[0,1)"), ("high", "[2,3)")])
    result = clf.classify([1.5])
    assert result[0] is None


def test_first_matching_interval_wins_on_overlap():
    clf = IntervalClassifier([("first", "[0,10)"), ("second", "[5,15)")])
    result = clf.classify([7.0])
    assert result[0] == "first"


def test_closed_interval_both_ends():
    clf = IntervalClassifier([("mid", "[0,1]")])
    result = clf.classify([0.0, 1.0])
    assert list(result) == ["mid", "mid"]


def test_open_interval_both_ends():
    clf = IntervalClassifier([("mid", "(0,1)")])
    result = clf.classify([0.0, 1.0])
    assert list(result) == [None, None]


def test_infinite_bounds_case_insensitive():
    clf = IntervalClassifier([("all", "(-INF,Infinity)")])
    result = clf.classify([-1e300, 0.0, 1e300])
    assert list(result) == ["all", "all", "all"]


@pytest.mark.parametrize("interval", [
    "0,1)",       # missing opening bracket
    "(0,1",       # missing closing bracket
    "(0-1)",      # missing comma
    "(0,1,2)",    # too many commas
    "(a,1)",      # non-numeric bound
    "()",         # empty bounds
])
def test_malformed_interval_raises(interval):
    with pytest.raises(ClassifyError):
        IntervalClassifier([("bad", interval)])


def test_stability_classes_reads_configured_pair():
    pairs = pd.DataFrame([
        {"slot": 1, "boom": 2, "boom2": 4, "variant": "none", "variable": "rib", "value": -0.2},
        {"slot": 2, "boom": 2, "boom2": 4, "variant": "none", "variable": "rib", "value": 0.5},
        {"slot": 1, "boom": 1, "boom2": 3, "variant": "none", "variable": "rib", "value": 99.0},  # wrong pair
    ])
    cfg_post = type("Cfg", (), {"stability": type("S", (), {
        "pair": (2, 4), "classes": _STABILITY_CLASSES,
    })()})()

    result = stability_classes(pairs, cfg_post)

    assert result.loc[1] == "strongly unstable"
    assert result.loc[2] == "strongly stable"
    assert len(result) == 2
