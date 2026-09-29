"""Finding slots: a pandas query over one row per (slot, boom), with a column
for every catalog quantity it names (read only when named) plus the slot's
time, height, stability class and tertiary filtering.
"""
import ast
import difflib
import io
import keyword
import re
import tokenize
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower.constants import HEIGHTS
from ttu_tower.viewer import timeline
from ttu_tower.viewer.timeaxis import slot_to_unix

_PREFIX = {"coverage": "coverage_", "flags": "flag_", "slot_qc": "qc_", "mrd_frame": "frame_"}
_TAU_FINAL = {"tau_s": "tau_s", "source": "tau_source", "source_status": "tau_status"}
FILTER_GROUPS = ("momentum", "heat", "ts", "t", "rh", "p", "vpt")
BUILTINS = {
    "slot": "slot index (10-min slots since the epoch)",
    "boom": "boom number",
    "height": "boom height [m]",
    "time": "slot start, wall-clock time in the run's zone (compare with strings: time > '2014-01-05')",
    "hour": "hour of day (run's zone), 0-23",
    "month": "month, 1-12",
    "doy": "day of year, 1-366",
    "stability": "stability class name (Ri_b of the configured pair), e.g. 'strongly unstable'",
    "filtered": "True where tertiary filtered any quantity group of this boom-slot (chosen variant or the means)",
    **{f"filtered_{g}": f"True where tertiary filtered the {g} group of this boom-slot" for g in FILTER_GROUPS},
}
PRESETS = (
    ("slots a test flagged, 30 at random", "flag_spike_w > 0 and boom == 9", 30),
    ("strongly unstable with a capped τ", "stability == 'strongly unstable' and tau_status == 'capped'", 0),
    ("a file but no usable data", "status == 'no_data'", 0),
    ("filtered momentum on one boom", "filtered_momentum and boom == 9", 0),
    ("strong wind at night", "ws_mean > 12 and night > 0.5", 0),
    ("low momentum coverage", "coverage_momentum_usable < 0.5", 0),
)


# pandas.eval's own functions
_ALLOWED = {"True", "False", "None", "abs", "sin", "cos", "tan", "exp", "log", "log10", "log1p", "expm1", "sqrt",
            "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "arctan2", "arcsinh", "arccosh", "arctanh"}


class FindError(ValueError):
    pass


@dataclass(frozen=True)
class Name:
    name: str
    description: str
    key: str | None = None  # catalog key; None for a built-in
    pair: tuple[int, int] | None = None


def _clean(text: str) -> str:
    text = re.sub(r"\W", "_", text).strip("_")
    return f"_{text}" if text[:1].isdigit() or keyword.iskeyword(text) else text


def names(catalog) -> dict[str, Name]:
    """Every name a query can use: built-ins, then one per catalog quantity
    (one per pair for pair quantities), e.g. ws_mean, ustar, rib_2_4, tau_status,
    flag_spike_w, coverage_momentum_usable, heat_tau_s.
    """
    out = {n: Name(n, d) for n, d in BUILTINS.items()}
    for q in catalog:
        parts = [p for p in q.key.split("|")[1:] if p]
        if q.table == "tau_final":
            base = _TAU_FINAL.get(parts[-1], f"tau_{parts[-1]}")
        else:
            base = _PREFIX.get(q.table, "") + "_".join(parts)
        base = _clean(base)
        if q.kind == "pair":
            for pair in q.pairs:
                n = f"{base}_{pair[0]}_{pair[1]}"
                out.setdefault(n, Name(n, f"{q.title}, b{pair[0]}–b{pair[1]}", q.key, pair))
            continue
        n = base if base not in out else f"{base}_{q.table}"
        what = "per slot" if q.kind == "slot" else "per boom"
        out[n] = Name(n, f"{q.title} ({what}; {q.table})", q.key)
    return out


def referenced(expr: str, known: dict) -> list[str]:
    """The known names an expression uses, in order of first use."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(expr).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        raise FindError(f"can't read the query: {exc}") from None
    seen = []
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string in known and tok.string not in seen:
            seen.append(tok.string)
    return seen


def unknown(expr: str, known: dict) -> list[str]:
    """Names in the expression that aren't columns (nor Python keywords or string contents)."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(expr).readline))
    except (tokenize.TokenError, IndentationError):
        return []
    out = []
    for i, t in enumerate(tokens):
        after_dot = i > 0 and tokens[i - 1].type == tokenize.OP and tokens[i - 1].string == "."  # a method
        if (t.type == tokenize.NAME and t.string not in known and t.string not in _ALLOWED and not after_dot
                and not keyword.iskeyword(t.string)):
            out.append(t.string)
    return out


def pinned_booms(expr: str, booms) -> list:
    """The booms a query can match: those it pins with a top-level `boom == N`
    or `boom in [...]` joined by `and` (so only their data is read), else all.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval").body
    except SyntaxError:
        return list(booms)
    terms = tree.values if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And) else [tree]
    allowed = set(booms)
    for t in terms:
        if not (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.left, ast.Name)
                and t.left.id == "boom"):
            continue
        try:
            value = ast.literal_eval(t.comparators[0])
        except ValueError:
            continue
        if isinstance(t.ops[0], ast.Eq) and isinstance(value, int):
            allowed &= {value}
        elif isinstance(t.ops[0], ast.In) and isinstance(value, (list, tuple, set)):
            allowed &= set(value)
    return [b for b in booms if b in allowed]


def build_frame(run, index, catalog, wanted: list[str], variant: str, booms, slot_range,
                stability, tz: str) -> pd.DataFrame:
    """One row per (slot, boom) over the period (or `slot_range`), with the
    `wanted` columns. Runs on a worker thread.
    """
    known = names(catalog)
    a, b = run.period
    if slot_range is not None:
        a, b = max(a, slot_range[0]), min(b, slot_range[1])
    slots = np.arange(a, max(a, b), dtype=np.int64)
    booms = list(booms)
    n = slots.size
    frame = pd.DataFrame({"slot": np.tile(slots, len(booms)), "boom": np.repeat(booms, n)})
    for name in wanted:
        if name in frame.columns:  # slot and boom
            continue
        entry = known[name]
        if entry.key is None:
            frame[name] = _builtin(name, run, index, slots, booms, variant, stability, tz, frame)
            continue
        q = catalog[entry.key]
        v = variant if variant in q.variants else q.variants[0]
        if q.kind == "boom":
            data = timeline.load(run, index, q, v, tuple(booms))
            values = np.concatenate([timeline.values_at(data.curves[bm], slots) if bm in data.curves
                                     else np.full(n, np.nan) for bm in booms])
        else:
            member = entry.pair if q.kind == "pair" else None
            data = timeline.load(run, index, q, v, (member,))
            curve = data.curves.get(member)
            values = np.tile(timeline.values_at(curve, slots) if curve is not None else np.full(n, np.nan),
                             len(booms))
        if q.categorical and data.categories:
            codes = np.where(np.isfinite(values), values, -1).astype(np.int64)
            frame[name] = pd.Categorical.from_codes(codes, categories=list(data.categories))
        else:
            frame[name] = values
    return frame


def _builtin(name, run, index, slots, booms, variant, stability, tz, frame) -> np.ndarray | pd.Series:
    n = slots.size
    if name == "height":
        return frame["boom"].map(HEIGHTS).to_numpy(dtype=float)
    if name in ("time", "hour", "month", "doy"):
        t = pd.to_datetime(slot_to_unix(slots), unit="s", utc=True).tz_convert(tz).tz_localize(None)
        part = {"time": t, "hour": t.hour, "month": t.month, "doy": t.dayofyear}[name]
        return np.tile(np.asarray(part), len(booms))
    if name == "stability":
        if stability is None:
            return pd.Categorical([None] * (n * len(booms)))
        curve, classes = stability
        codes = timeline.values_at(curve, slots)
        codes = np.where(np.isfinite(codes), codes, -1).astype(np.int64)
        return pd.Categorical.from_codes(np.tile(codes, len(booms)), categories=list(classes))
    if name.startswith("filtered"):
        group = name[len("filtered_"):] if name != "filtered" else None
        return _filtered(run, index, slots, booms, variant, group)
    raise FindError(f"unknown built-in {name}")


def _filtered(run, index, slots, booms, variant, group) -> np.ndarray:
    out = np.zeros(slots.size * len(booms), dtype=bool)
    if "tertiary" not in run.stages or slots.size == 0:
        return out
    f = pa_dataset.field
    expr = f("variant").isin([variant, "none"]) & (f("slot") >= int(slots[0])) & (f("slot") <= int(slots[-1]))
    if group is not None:
        expr = expr & (f("group") == group)
    log = index.read("filter_log", columns=["slot", "boom"], filter=expr).drop_duplicates()
    row_of = {bm: i for i, bm in enumerate(booms)}
    rows = log["boom"].map(row_of)
    ok = rows.notna().to_numpy()
    positions = rows[ok].to_numpy(dtype=np.int64) * slots.size + (log["slot"].to_numpy()[ok] - slots[0])
    out[positions] = True
    return out


def run_query(frame: pd.DataFrame, expr: str, known: dict, sample: int = 0, seed: int | None = None) -> pd.DataFrame:
    """The matching rows (a random `sample` of them if > 0), by time then boom."""
    if not expr.strip():
        raise FindError("write a condition, e.g. ws_mean > 12 and boom == 9")
    try:
        mask = frame.eval(expr, engine="python")
    except Exception as exc:
        missing = unknown(expr, known)
        if missing:
            close = difflib.get_close_matches(missing[0], list(known), n=3)
            hint = f" (did you mean {', '.join(close)}?)" if close else " (see Names…)"
            raise FindError(f"'{missing[0]}' isn't a column{hint}") from None
        raise FindError(f"{type(exc).__name__}: {exc}") from None
    if not isinstance(mask, pd.Series) or mask.dtype != bool:
        raise FindError("the query must be a condition (true or false for each slot and boom)")
    result = frame[mask.to_numpy()]
    if sample and len(result) > sample:
        result = result.sample(sample, random_state=seed)
    return result.sort_values(["slot", "boom"], kind="stable").reset_index(drop=True)


def find(run, index, catalog, expr: str, variant: str, booms, slot_range, stability, tz: str,
         sample: int = 0, seed: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Runs on a worker thread: (matches, the columns the query named)."""
    known = names(catalog)
    wanted = referenced(expr, known)
    missing = unknown(expr, known)
    if missing:
        close = difflib.get_close_matches(missing[0], list(known), n=3)
        hint = f" (did you mean {', '.join(close)}?)" if close else " (see Names…)"
        raise FindError(f"'{missing[0]}' isn't a column{hint}")
    frame = build_frame(run, index, catalog, wanted, variant, pinned_booms(expr, booms), slot_range, stability, tz)
    return run_query(frame, expr, known, sample, seed), wanted
