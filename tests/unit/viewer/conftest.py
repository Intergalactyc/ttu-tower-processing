import os
from types import SimpleNamespace

import numpy as np
import pytest

from ttu_tower.config import load_config
from ttu_tower.io.runs import find_home
from ttu_tower.primary.runner import run_primary
from ttu_tower.secondary.runner import run_secondary
from ttu_tower.tertiary.runner import run_tertiary
from ttu_tower.validation.synthetic import write_raw_dataset

from viewer_fixtures import (
    FULL_BOOMS, FULL_HALF_HOURS, PRIMARY_HALF_HOURS, PRIMARY_MISSING, mutate_full, mutate_primary, write_config,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _args(config, **overrides):
    base = dict(config=str(config), nproc=1, test=False, redo_failures=False, force=False,
                allow_non_parquet=False)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(scope="session")
def primary_run(tmp_path_factory):
    """(run_dir, cfg, config_path, raw_dir) of a primary-only synthetic run."""
    base = tmp_path_factory.mktemp("viewer_primary")
    raw_dir = base / "raw"
    write_raw_dataset(raw_dir, PRIMARY_HALF_HOURS, [1], np.random.default_rng(7), missing=PRIMARY_MISSING,
                      mutate=mutate_primary)
    config_path = write_config(base / "viewer_primary.toml", "viewer_primary", raw_dir, PRIMARY_HALF_HOURS, [1], 3)
    cfg = load_config(config_path)
    run_primary(cfg, _args(config_path))
    return SimpleNamespace(run_dir=find_home() / "results" / "viewer_primary", cfg=cfg, config_path=config_path,
                           raw_dir=raw_dir, tag="viewer_primary")


@pytest.fixture(scope="session")
def full_run(tmp_path_factory):
    """(run_dir, cfg, config_path) of a primary+secondary+tertiary synthetic run."""
    base = tmp_path_factory.mktemp("viewer_full")
    raw_dir = base / "raw"
    write_raw_dataset(raw_dir, FULL_HALF_HOURS, FULL_BOOMS, np.random.default_rng(11), missing=[6103],
                      mutate=mutate_full)
    config_path = write_config(base / "viewer_full.toml", "viewer_full", raw_dir, FULL_HALF_HOURS, FULL_BOOMS, 3)
    cfg = load_config(config_path)
    run_primary(cfg, _args(config_path))
    run_secondary(cfg, _args(config_path))
    run_tertiary(cfg, _args(config_path))
    return SimpleNamespace(run_dir=find_home() / "results" / "viewer_full", cfg=cfg, config_path=config_path,
                           raw_dir=raw_dir, tag="viewer_full")
