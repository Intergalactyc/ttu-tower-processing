from types import SimpleNamespace

import numpy as np
import pandas as pd

from ttu_tower.post.classify import IntervalClassifier
from ttu_tower.viewer.stability import NAMES, StabilitySpec, classify, from_config

DEFAULT = (("strongly unstable", "(-inf,-0.05)"), ("unstable", "[-0.05,-0.01)"), ("neutral", "[-0.01,0.01)"),
           ("stable", "[0.01,0.1)"), ("strongly stable", "[0.1,inf)"))


def _post(classes, pair=(2, 4)):
    return SimpleNamespace(stability=SimpleNamespace(pair=pair, classes=classes))


def test_the_runs_own_classes_become_a_spec_that_classifies_the_same():
    spec = from_config(_post(DEFAULT))
    assert spec == StabilitySpec("rib", (2, 4), "mrd", (-0.05, -0.01, 0.01, 0.1))
    values = np.array([-1.0, -0.05, -0.02, 0.0, 0.01, 0.099, 0.1, 3.0, np.nan])
    ours = [c if isinstance(c, str) else None for c in classify(pd.Series(values), spec.classes())]
    theirs = IntervalClassifier(list(DEFAULT)).classify(values)
    assert ours == list(theirs)


def test_classes_that_arent_five_contiguous_intervals_stay_the_runs_own():
    assert from_config(_post(DEFAULT[:4])) is None
    gap = list(DEFAULT)
    gap[2] = ("neutral", "[-0.01,0.02)")
    assert from_config(_post(tuple(gap))) is None
    closed = list(DEFAULT)
    closed[1] = ("unstable", "[-0.05,-0.01]")
    closed[2] = ("neutral", "(-0.01,0.01)")
    assert from_config(_post(tuple(closed))) is None


def test_a_spec_checks_itself_and_round_trips():
    assert StabilitySpec(edges=(0.1, 0.0, 0.2, 0.3)).errors()
    assert StabilitySpec(booms=(3, 3)).errors()
    assert not StabilitySpec("zeta", (3, 3), "naive", (-1.0, -0.1, 0.1, 1.0)).errors()
    spec = StabilitySpec("zeta", (8, 8), "naive", (-1.0, -0.1, 0.1, 1.0))
    assert StabilitySpec.from_dict(spec.to_dict()) == spec
    assert [name for name, _ in spec.classes()] == list(NAMES) and spec.describe() == "ζ at b8 (naive)"
    assert StabilitySpec("rib", (4, 2)).describe() == "Ri_b(b2, b4)"
