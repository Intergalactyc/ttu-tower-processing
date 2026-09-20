from pathlib import Path

import numpy as np
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.constants import BOOMS
from ttu_tower.primary.stage_a import stage_a

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "reference" / "stage_a_R01457.npz"


def _qc_cfg():
    raw = {"paths": {"raw_dirs": ["/data"]}, "files": {"bad_records": [[1, 2]]}}
    return config_from_dict(raw, default_tag="t").qc


@pytest.mark.parametrize("boom", BOOMS)
def test_stage_a_matches_old_pipeline_reference(boom):
    d = np.load(FIXTURE_PATH)
    raw = {v: d[f"{v}_{boom}"] for v in ("u", "v", "w", "ts", "t", "rh", "p")}

    out = stage_a(raw, boom=boom, cfg=_qc_cfg())

    assert out.ue == pytest.approx(d[f"out_ue_{boom}"], rel=1e-10, abs=1e-10)
    assert out.vn == pytest.approx(d[f"out_vn_{boom}"], rel=1e-10, abs=1e-10)
    assert out.w == pytest.approx(d[f"out_w_{boom}"], rel=1e-10, abs=1e-10)
    assert out.ts == pytest.approx(d[f"out_ts_{boom}"], rel=1e-10, abs=1e-10)
    assert out.t == pytest.approx(d[f"out_t_{boom}"], rel=1e-10, abs=1e-10)
    assert out.rh == pytest.approx(d[f"out_rh_{boom}"], rel=1e-10, abs=1e-10)
    assert out.p == pytest.approx(d[f"out_p_{boom}"], rel=1e-10, abs=1e-10)
