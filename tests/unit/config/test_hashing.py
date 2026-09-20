from ttu_tower.config import config_from_dict
from ttu_tower.config.hashing import major_minor, primary_hash, secondary_hash, tertiary_hash

_VERSION = "2.0.0"


def _raw(**overrides) -> dict:
    raw = {
        "paths": {"raw_dirs": ["/data/raw"]},
        "files": {"bad_records": [[1, 2]]},
    }
    for section, values in overrides.items():
        raw[section] = values
    return raw


def _cfg(**overrides):
    return config_from_dict(_raw(**overrides), default_tag="t")


def test_major_minor():
    assert major_minor("2.0.0") == "2.0"
    assert major_minor("2.1.3") == "2.1"


def test_hash_stable_under_key_reordering():
    raw_a = {"paths": {"raw_dirs": ["/x"]}, "files": {"bad_records": [[1, 2]]}}
    raw_b = {"files": {"bad_records": [[1, 2]]}, "paths": {"raw_dirs": ["/x"]}}
    cfg_a = config_from_dict(raw_a, default_tag="t")
    cfg_b = config_from_dict(raw_b, default_tag="t")
    assert primary_hash(cfg_a, _VERSION) == primary_hash(cfg_b, _VERSION)


def test_secondary_key_changes_secondary_and_tertiary_but_not_primary():
    cfg_a = _cfg()
    cfg_b = _cfg(secondary={"min_tau_its_ratio": 5.0})

    assert primary_hash(cfg_a, _VERSION) == primary_hash(cfg_b, _VERSION)
    assert secondary_hash(cfg_a, _VERSION) != secondary_hash(cfg_b, _VERSION)
    assert tertiary_hash(cfg_a, _VERSION) != tertiary_hash(cfg_b, _VERSION)


def test_nproc_changes_no_hash():
    cfg_a = _cfg(primary={"nproc": 0})
    cfg_b = _cfg(primary={"nproc": 8})

    assert primary_hash(cfg_a, _VERSION) == primary_hash(cfg_b, _VERSION)
    assert secondary_hash(cfg_a, _VERSION) == secondary_hash(cfg_b, _VERSION)
    assert tertiary_hash(cfg_a, _VERSION) == tertiary_hash(cfg_b, _VERSION)


def test_tertiary_key_changes_only_tertiary():
    cfg_a = _cfg()
    cfg_b = _cfg(tertiary={"min_coverage": 0.5})

    assert primary_hash(cfg_a, _VERSION) == primary_hash(cfg_b, _VERSION)
    assert secondary_hash(cfg_a, _VERSION) == secondary_hash(cfg_b, _VERSION)
    assert tertiary_hash(cfg_a, _VERSION) != tertiary_hash(cfg_b, _VERSION)


def test_primary_relevant_key_changes_all_three_hashes():
    cfg_a = _cfg()
    cfg_b = _cfg(qc={"min_coverage": 0.5})

    assert primary_hash(cfg_a, _VERSION) != primary_hash(cfg_b, _VERSION)
    assert secondary_hash(cfg_a, _VERSION) != secondary_hash(cfg_b, _VERSION)
    assert tertiary_hash(cfg_a, _VERSION) != tertiary_hash(cfg_b, _VERSION)
