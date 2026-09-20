from pathlib import Path

import pandas as pd
import pytest

from ttu_tower.config import load_config
from ttu_tower.io.rawfiles import build_file_table
from ttu_tower.timegrid import time_to_half_hour

_TEMPLATE = Path(__file__).resolve().parents[2] / "configs" / "templates" / "oneyear.toml"


@pytest.mark.integration
def test_real_listing_matches_expected_counts(raw_dir):
    cfg = load_config(_TEMPLATE)
    table = build_file_table([str(raw_dir)], cfg.files)

    counts = table["status"].value_counts().to_dict()
    assert len(table) == 16_817
    assert counts.get("bad_record", 0) == 1_686
    assert counts.get("offset", 0) == 133
    assert counts.get("collision", 0) == 0
    assert counts.get("bad_length", 0) == 0
    assert counts.get("unparseable", 0) == 0
    assert counts.get("accepted", 0) == 14_998

    start = pd.Timestamp(cfg.period.start, tz=cfg.output.timezone)
    end = pd.Timestamp(cfg.period.end, tz=cfg.output.timezone)
    h_start, h_end = int(time_to_half_hour(start)), int(time_to_half_hour(end))
    n_half_hours = h_end - h_start
    accepted_in_period = table[
        (table["status"] == "accepted") & (table["half_hour"] >= h_start) & (table["half_hour"] < h_end)
    ]
    assert n_half_hours == 17_520
    assert len(accepted_in_period) == 14_998
