import numpy as np
import pandas as pd

from ttu_tower import schema
from ttu_tower.constants import BOOMS
from ttu_tower.io.store import write_fragment
from ttu_tower.jets.data import load_boom_values, load_context, load_profiles
from ttu_tower.timegrid import slot_to_time

SLOTS = [100, 101, 103, 104]  # slot 102 has no rows at all


def _ws(slot, boom):
    return 5.0 + boom + 0.01 * slot


def _write(run_dir, table, rows, key="bat0000001-0000002"):
    df = pd.DataFrame(rows)
    df["slot_start"] = slot_to_time(df["slot"].to_numpy())
    write_fragment(run_dir / "tertiary" / "data" / table, key, schema.cast(table, df))


def _run(tmp_path):
    rows = []
    for slot in SLOTS:
        for boom in BOOMS:
            ws = np.nan if (slot, boom) == (103, 9) else _ws(slot, boom)
            rows += [
                dict(slot=slot, boom=boom, variant="none", variable="ws", stat="mean", value=ws),
                dict(slot=slot, boom=boom, variant="none", variable="ws", stat="std", value=99.0),
                dict(slot=slot, boom=boom, variant="mrd", variable="ws", stat="mean", value=99.0),
                dict(slot=slot, boom=boom, variant="none", variable="wd", stat="mean", value=10.0 * boom),
                dict(slot=slot, boom=boom, variant="none", variable="vpt", stat=None, value=290.0 + boom),
            ]
    # a second fragment, as tertiary writes one per batch
    _write(tmp_path, "boom_final", [r for r in rows if r["slot"] < 103])
    _write(tmp_path, "boom_final", [r for r in rows if r["slot"] >= 103], key="bat0000003-0000004")
    pairs = [dict(slot=s, boom=a, boom2=b, variant="none", variable="rib", value=0.01 * s + a + b)
             for s in SLOTS for a, b in ((1, 2), (2, 4))]
    _write(tmp_path, "pairs", pairs)
    final = [dict(slot=s, variable=v, value=x) for s in SLOTS
             for v, x in (("night", float(s % 2)), ("sun_elevation", -5.0))]
    _write(tmp_path, "slot_final", final)
    return tmp_path


def test_profiles_are_the_none_variant_slot_means_on_a_complete_grid(tmp_path):
    p = load_profiles(_run(tmp_path))
    assert list(p.slots) == [100, 101, 102, 103, 104] and p.booms == tuple(BOOMS)
    assert np.isnan(p.ws[2]).all()
    assert np.isnan(p.ws[3, BOOMS.index(9)])
    assert p.ws[0, 0] == _ws(100, 1) and p.ws[4, 9] == _ws(104, 10)
    assert p.extra["wd"][1, BOOMS.index(7)] == 70.0


def test_profiles_for_some_booms_only(tmp_path):
    p = load_profiles(_run(tmp_path), booms=(10, 8), with_direction=False)
    assert p.booms == (8, 10) and p.ws.shape == (5, 2) and not p.extra
    assert p.ws[0, 1] == _ws(100, 10)


def test_context_has_the_filter_quantities(tmp_path):
    ctx = load_context(_run(tmp_path), vpt_booms=(2, 10), rib_pairs=((4, 2),))
    assert set(ctx.columns) == {"vpt_b2", "vpt_b10", "rib_b2-b4", "night"}
    assert list(ctx.index) == SLOTS
    assert ctx.loc[101, "vpt_b10"] == 300.0 and ctx.loc[103, "rib_b2-b4"] == 0.01 * 103 + 6
    assert ctx.loc[101, "night"] == 1.0 and ctx.loc[104, "night"] == 0.0


def test_any_boom_quantity_loads_per_slot_and_boom(tmp_path):
    run = _run(tmp_path)
    vpt = load_boom_values(run, "vpt", None, "none", booms=(2, 10))
    assert list(vpt.columns) == [2, 10] and vpt.loc[101, 10] == 300.0
    spread = load_boom_values(run, "ws", "std", "none")
    assert (spread.to_numpy() == 99.0).all() and list(spread.index) == SLOTS
