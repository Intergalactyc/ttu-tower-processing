"""Quantities derived by formula from others, e.g. ws/ustar, defined in the
viewer and kept in the ttu-tower home for every run.

A definition names its variables: each one of the catalog's quantities (with
its own variant), at "each boom" (the derived quantity's own member) or at one
fixed member. The result is per boom if an "each boom" variable is per boom
(a slot quantity's value then holds at every boom, and `z` is the boom's
height), per boom pair if one is per pair, else per slot.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from functools import reduce
from pathlib import Path

import numpy as np

from ttu_tower.constants import HEIGHTS
from ttu_tower.io.runs import locate_home
from ttu_tower.viewer import expression
from ttu_tower.viewer.catalog import Catalog, Quantity

GROUP = "Derived"
TABLE = "derived"
HEIGHT = "z"  # a per-boom result's height, m
_VERSION = 1


@dataclass(frozen=True)
class Variable:
    name: str
    key: str  # the catalog quantity's key
    variant: str = "none"
    member: object = None  # None: each member of the result; else a boom, or a (boom, boom2) pair


@dataclass(frozen=True)
class Definition:
    name: str
    formula: str
    variables: tuple[Variable, ...]
    unit: str = ""

    @property
    def key(self) -> str:
        return f"{TABLE}|{self.name}|"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["variables"] = [dict(asdict(v), member=list(v.member) if isinstance(v.member, tuple) else v.member)
                            for v in self.variables]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Definition":
        variables = tuple(Variable(v["name"], v["key"], v.get("variant", "none"),
                                   tuple(v["member"]) if isinstance(v.get("member"), list) else v.get("member"))
                          for v in d.get("variables", []))
        return cls(d["name"], d["formula"], variables, d.get("unit", ""))


@dataclass(frozen=True)
class Resolved:
    """A definition with its variables' quantities from one run's catalog."""
    definition: Definition
    quantities: tuple[Quantity, ...]


def reserved() -> set[str]:
    return {HEIGHT, *expression.FUNCTIONS, *expression.CONSTANTS}


def result_kind(quantities, variables) -> str:
    """'boom', 'pair' or 'slot' (ValueError where "each boom" variables mix booms and pairs)."""
    kinds = {q.kind for q, v in zip(quantities, variables) if v.member is None and q.kind in ("boom", "pair")}
    if len(kinds) > 1:
        raise ValueError("the variables at each member mix per-boom and per-pair quantities")
    return kinds.pop() if kinds else "slot"


def resolve(d: Definition, catalog: Catalog) -> Quantity:
    """The derived quantity on this catalog (ValueError, with the reason, where it can't be)."""
    if not d.name.isidentifier():
        raise ValueError(f"the name '{d.name}' must be letters, digits and _ (not starting with a digit)")
    if not d.variables:
        raise ValueError("no variables")
    names = [v.name for v in d.variables]
    for n in names:
        if not n.isidentifier() or n in reserved():
            raise ValueError(f"'{n}' can't name a variable (letters, digits and _; not {', '.join(sorted(reserved()))})")
    if len(set(names)) != len(names):
        raise ValueError("two variables share a name")
    quantities = []
    for v in d.variables:
        q = catalog.quantities.get(v.key)
        if q is None:
            raise ValueError(f"{v.name}: {v.key} isn't in this run")
        if q.categorical or q.table == TABLE:
            raise ValueError(f"{v.name}: {q.title} can't be used (categories, or another derived quantity)")
        if v.variant not in q.variants:
            raise ValueError(f"{v.name}: {q.title} has no {v.variant} variant")
        quantities.append(q)
    kind = result_kind(quantities, d.variables)
    expression.compile_expression(d.formula, (*names, *((HEIGHT,) if kind == "boom" else ())), x=False)
    pairs = ()
    if kind == "pair":
        sets = [set(q.pairs) for q, v in zip(quantities, d.variables) if v.member is None and q.kind == "pair"]
        pairs = tuple(sorted(set.intersection(*sets)))
    resolved = Resolved(d, tuple(quantities))
    return Quantity(key=d.key, table=TABLE, variable=d.name, stat=None, kind=kind, variants=("none",),
                    label=d.name, unit=d.unit, group=GROUP, pairs=pairs, extra=(("definition", resolved),))


def definition_of(q: Quantity) -> Resolved:
    return dict(q.extra)["definition"]


def add_to(catalog: Catalog, definitions) -> list[str]:
    """Add every definition this run can resolve (replacing earlier derived ones); why the others can't be."""
    for key in [k for k, q in catalog.quantities.items() if q.table == TABLE]:
        del catalog.quantities[key]
    notes = []
    for d in definitions:
        try:
            catalog.add(resolve(d, catalog))
        except ValueError as exc:
            notes.append(f"{d.name}: {exc}")
    return notes


def load(run, index, q: Quantity, members) -> dict:
    """{member: (slots, values)}: the formula over the slots every variable has."""
    from ttu_tower.viewer import timeline  # timeline loads derived quantities through here

    resolved = definition_of(q)
    d = resolved.definition
    func = expression.compile_expression(d.formula, (*(v.name for v in d.variables),
                                                     *((HEIGHT,) if q.kind == "boom" else ())), x=False)
    out = {}
    for m in members:
        curves = []
        for v, vq in zip(d.variables, resolved.quantities):
            at = None if vq.kind == "slot" else (m if v.member is None else v.member)
            data = timeline.load(run, index, vq, v.variant, None if at is None else [at])
            curves.append(data.curves.get(at))
        if any(c is None or c.slots.size == 0 for c in curves):
            out[m] = (np.empty(0, dtype=np.int64), np.empty(0))
            continue
        slots = reduce(lambda a, b: np.intersect1d(a, b, assume_unique=True), (c.slots for c in curves))
        values = {v.name: c.y[np.searchsorted(c.slots, slots)] for v, c in zip(d.variables, curves)}
        if q.kind == "boom":
            values[HEIGHT] = HEIGHTS[m]
        y = func(**values) if slots.size else np.empty(0)
        out[m] = (slots, np.where(np.isfinite(y), y, np.nan))
    return out


# --- kept in the ttu-tower home ------------------------------------------------------------------

def default_path() -> Path | None:
    home = locate_home()
    return home / "viewer" / "derived.json" if home is not None else None


@dataclass
class DerivedStore:
    """Every change is written through; `path` None keeps them in memory only."""
    path: Path | None = None
    items: list[Definition] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path | None = None, load: bool = True) -> "DerivedStore":
        store = cls(Path(path) if path is not None else (default_path() if load else None))
        if store.path is not None and store.path.exists():
            raw = json.loads(store.path.read_text(encoding="utf-8"))
            store.items = [Definition.from_dict(d) for d in raw.get("derived", [])]
        return store

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": _VERSION, "derived": [d.to_dict() for d in self.items]}, indent=1,
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def put(self, d: Definition, replacing: str | None = None) -> None:
        """Add `d`, replacing the one named `replacing` (renamed) or of its own name."""
        gone = {d.name, replacing}
        self.items = [x for x in self.items if x.name not in gone] + [d]
        self.save()

    def remove(self, name: str) -> None:
        self.items = [x for x in self.items if x.name != name]
        self.save()
