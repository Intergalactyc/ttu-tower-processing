import datetime as dt
from pathlib import Path

import pytest

from ttu_tower.era5 import config as era5_config
from ttu_tower.era5 import download

CONFIG = Path(__file__).parents[3] / "configs" / "era5.toml"


@pytest.fixture
def cfg():
    return era5_config.load(CONFIG)


def test_chunks_never_cross_a_month():
    out = download.chunks(download.days(dt.date(2013, 11, 20), dt.date(2014, 1, 3)), 10)
    spans = [(c[0], c[-1]) for c in out]
    assert spans == [
        (dt.date(2013, 11, 20), dt.date(2013, 11, 29)),
        (dt.date(2013, 11, 30), dt.date(2013, 11, 30)),
        (dt.date(2013, 12, 1), dt.date(2013, 12, 10)),
        (dt.date(2013, 12, 11), dt.date(2013, 12, 20)),
        (dt.date(2013, 12, 21), dt.date(2013, 12, 30)),
        (dt.date(2013, 12, 31), dt.date(2013, 12, 31)),
        (dt.date(2014, 1, 1), dt.date(2014, 1, 3)),
    ]
    assert sum(len(c) for c in out) == 45


def test_chunks_across_months_and_gaps():
    ds = download.days(dt.date(2013, 11, 28), dt.date(2013, 12, 6))
    ds.remove(dt.date(2013, 12, 2))
    spans = [(c[0], c[-1]) for c in download.chunks(ds, 7, within_month=False)]
    assert spans == [
        (dt.date(2013, 11, 28), dt.date(2013, 12, 1)),
        (dt.date(2013, 12, 3), dt.date(2013, 12, 6)),
    ]


def test_full_period_request_counts(cfg, tmp_path):
    sfc = download.surface_requests(cfg, tmp_path, cfg.start, cfg.end)
    pl = download.pressure_requests(cfg, tmp_path, cfg.start, cfg.end)
    ml = download.model_requests(cfg, tmp_path, cfg.start, cfg.end)
    assert len(sfc) == 1 + 13          # invariant + Nov 2013..Oct 2014 + 1 Nov 2014
    assert len(pl) == 11 * 8 + 7 + 1   # 4-day chunks: 8 per 30/31-day month, 7 in Feb, 1 for 1 Nov 2014
    assert len(ml) == 53               # 366 days in 7-day chunks
    assert ml[0].request["date"] == "2013-11-01/to/2013-11-07"
    assert len({r.target for r in sfc + pl + ml}) == len(sfc) + len(pl) + len(ml)


def test_model_skips_days_already_on_disk(cfg, tmp_path):
    for name in ("ml_20131101.grib", "ml_20131102.grib", "ml_20131110-20131116.grib", "ml_20131103.grib.part"):
        (tmp_path / name).touch()
    ml = download.model_requests(cfg, tmp_path, dt.date(2013, 11, 1), dt.date(2013, 11, 20))
    assert [r.request["date"] for r in ml] == [
        "2013-11-03/to/2013-11-09",
        "2013-11-17/to/2013-11-20",
    ]


def test_partial_range_does_not_shadow_full_chunk(cfg, tmp_path):
    full = download.pressure_requests(cfg, tmp_path, dt.date(2013, 11, 1), dt.date(2013, 11, 30))
    one = download.pressure_requests(cfg, tmp_path, dt.date(2013, 11, 1), dt.date(2013, 11, 1))
    assert full[0].target != one[0].target


def test_model_levelist():
    levels = download.model_levelist(75).split("/")
    assert levels[0] == "1" and levels[1] == "75" and levels[-1] == "137"
    assert len(levels) == 1 + 63
