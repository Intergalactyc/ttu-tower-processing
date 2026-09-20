"""Per-stage config hashes: SHA-256 (first 16 hex chars) of canonical JSON of
the sections each stage depends on, plus the package's major.minor version.
"""
import hashlib
import json

from ttu_tower.config.model import Config, to_jsonable


def major_minor(version: str) -> str:
    """'2.0.0' -> '2.0'."""
    parts = version.split(".")
    return ".".join(parts[:2])


def _hash_payload(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def primary_hash(cfg: Config, version: str) -> str:
    primary_dict = to_jsonable(cfg.primary)
    primary_dict.pop("nproc", None)
    payload = {
        "files": to_jsonable(cfg.files),
        "qc": to_jsonable(cfg.qc),
        "mrd": to_jsonable(cfg.mrd),
        "ladder": to_jsonable(cfg.ladder),
        "primary": primary_dict,
        "output": to_jsonable(cfg.output),
        "paths.raw_dirs": to_jsonable(cfg.paths.raw_dirs),
        "period": to_jsonable(cfg.period),
        "version": major_minor(version),
    }
    return _hash_payload(payload)


def secondary_hash(cfg: Config, version: str) -> str:
    payload = {
        "secondary": to_jsonable(cfg.secondary),
        "primary_hash": primary_hash(cfg, version),
    }
    return _hash_payload(payload)


def tertiary_hash(cfg: Config, version: str) -> str:
    payload = {
        "tertiary": to_jsonable(cfg.tertiary),
        "paths.mesonet_dir": cfg.paths.mesonet_dir,
        "secondary_hash": secondary_hash(cfg, version),
    }
    return _hash_payload(payload)
