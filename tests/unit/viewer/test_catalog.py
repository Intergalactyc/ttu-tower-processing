import numpy as np
import pandas as pd
import pytest

from ttu_tower.io.store import read_table
from ttu_tower.viewer.catalog import build_catalog
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.run import RunHandle
from ttu_tower.viewer import timeline

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def opened(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    return run, FragmentIndex(run.run_dir), build_catalog(run)


def test_every_boom_final_quantity_is_listed(full_run, opened):
    _, _, catalog = opened
    boom_final = read_table(full_run.run_dir / "tertiary" / "data" / "boom_final")
    present = {(str(v), None if pd.isna(s) else str(s)) for v, s in
               boom_final[["variable", "stat"]].astype(object).drop_duplicates().itertuples(index=False)}
    listed = {(q.variable, q.stat) for q in catalog if q.table == "boom_final"}
    assert present == listed


def test_every_quantity_loads(full_run, opened):
    run, index, catalog = opened
    empty = []
    for q in catalog:
        variant = q.variants[0]
        members = q.pairs[:1] if q.kind == "pair" else [1, 2]
        data = timeline.load(run, index, q, variant, members)
        if not any(np.isfinite(c.y).any() for c in data.curves.values()):
            empty.append(q.key)
    # legitimately all-missing in a tiny synthetic run: no mesonet, tests that never fire, and a wd-std
    # power law fitted to two booms whose synthetic direction spread doesn't vary with height
    allowed = {k for k in empty if k.startswith(("flags|", "slot_final|")) or "meso" in k} | {"profile|wdgamma"}
    assert set(empty) <= allowed, sorted(set(empty) - allowed)


def test_vocabulary_is_cached_under_the_home_not_the_run(full_run, opened):
    run, _, _ = opened
    before = set(p for p in full_run.run_dir.rglob("*"))
    build_catalog(RunHandle.from_dir(full_run.run_dir))
    assert set(p for p in full_run.run_dir.rglob("*")) == before
