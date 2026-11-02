"""Quantities derived by formula from others, e.g. ws/ustar, defined in the
viewer and kept in the ttu-tower home for every run.

A definition names its variables: each one of the catalog's quantities (with
its own variant), at "each boom" (the derived quantity's own member) or at one
fixed member. The result is per boom if an "each boom" variable is per boom
(a slot quantity's value then holds at every boom, and `z` is the boom's
height), per boom pair if one is per pair, else per slot. A formula may also
use the site's constants (`SITE`): g, local gravity as the pipeline takes it,
at the result's height; f, the Coriolis parameter; kappa, cp and Rd. A
variable of the same name hides one. A variable may be another derived
quantity, so long as none ends up using itself.
"""
import json
import os
from dataclasses import asdict, dataclass, field, replace
from functools import reduce
from pathlib import Path

import numpy as np

from ttu_tower.constants import HEIGHTS, SITE_ELEVATION, SITE_LATITUDE
from ttu_tower.io.runs import locate_home
from ttu_tower.physics.constants import CP, KAPPA, R
from ttu_tower.physics.earth import coriolis, local_gravity
from ttu_tower.viewer import expression
from ttu_tower.viewer.catalog import Catalog, Quantity

GROUP = "Derived"
TABLE = "derived"
HEIGHT = "z"  # a per-boom result's height, m
SITE = {"g": "local gravity, m/s² (at the boom's height; a pair's mean height; the ground for a per-slot quantity)",
        "f": "Coriolis parameter at the tower, 1/s",
        "kappa": "von Kármán constant",
        "cp": "specific heat of dry air at constant pressure, J/(kg K)",
        "Rd": "gas constant of dry air, J/(kg K)"}
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


def site_values(kind: str, member) -> dict[str, float]:
    """The site's constants for a result's `member` ("boom": a boom; "pair": a (boom, boom2) pair)."""
    height = (HEIGHTS[member] if kind == "boom" else
              (HEIGHTS[member[0]] + HEIGHTS[member[1]]) / 2 if kind == "pair" else 0.0)
    return {"g": float(local_gravity(SITE_LATITUDE, SITE_ELEVATION + height)), "f": float(coriolis(SITE_LATITUDE)),
            "kappa": KAPPA, "cp": CP, "Rd": R}


def _names(variables, kind: str) -> tuple[str, ...]:
    """What a formula may use: the variables, the site's constants, and z for a per-boom result."""
    return (*variables, *SITE, *((HEIGHT,) if kind == "boom" else ()))


def reserved() -> set[str]:
    return {HEIGHT, *expression.FUNCTIONS, *expression.CONSTANTS}  # min, max: a variable may still be called so


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
        if q is None and v.key.startswith(f"{TABLE}|"):
            raise ValueError(f"{v.name}: the derived quantity {derived_name(v.key)} isn't defined, or can't be on "
                             f"this run")
        if q is None:
            raise ValueError(f"{v.name}: {v.key} isn't in this run")
        if q.categorical:
            raise ValueError(f"{v.name}: {q.title} can't be used (categories)")
        if v.variant not in q.variants:
            raise ValueError(f"{v.name}: {q.title} has no {v.variant} variant")
        quantities.append(q)
    kind = result_kind(quantities, d.variables)
    expression.compile_expression(d.formula, _names(names, kind), x=False)
    pairs = ()
    if kind == "pair":
        sets = [set(q.pairs) for q, v in zip(quantities, d.variables) if v.member is None and q.kind == "pair"]
        pairs = tuple(sorted(set.intersection(*sets)))
    resolved = Resolved(d, tuple(quantities))
    return Quantity(key=d.key, table=TABLE, variable=d.name, stat=None, kind=kind, variants=("none",),
                    label=d.name, unit=d.unit, group=GROUP, pairs=pairs, extra=(("definition", resolved),))


def definition_of(q: Quantity) -> Resolved:
    return dict(q.extra)["definition"]


def derived_name(key: str) -> str:
    return key.split("|")[1]


def uses(d: Definition) -> set[str]:
    """The derived quantities `d`'s variables are."""
    return {derived_name(v.key) for v in d.variables if v.key.startswith(f"{TABLE}|")}


def in_order(definitions) -> tuple[list[Definition], dict[str, list[str]]]:
    """The definitions, each after the derived quantities it uses; and {name: its loop} for those
    that use themselves (directly or through others).
    """
    by_name = {d.name: d for d in definitions}
    order, done, loops = [], set(), {}

    def visit(d, path):
        if d.name in done:
            return
        if d.name in path:
            cycle = path[path.index(d.name):]
            for i, n in enumerate(cycle):  # each told from itself: b → a → b
                loops.setdefault(n, cycle[i:] + cycle[:i] + [n])
            return
        for n in sorted(uses(d)):
            if n in by_name:
                visit(by_name[n], path + [d.name])
        done.add(d.name)
        order.append(d)

    for d in definitions:
        visit(d, [])
    return order, loops


def add_to(catalog: Catalog, definitions) -> list[str]:
    """Add every definition this run can resolve (replacing earlier derived ones), each after those it
    uses; why the others can't be.
    """
    for key in [k for k, q in catalog.quantities.items() if q.table == TABLE]:
        del catalog.quantities[key]
    order, loops = in_order(definitions)
    notes = []
    for d in order:
        try:
            if d.name in loops:
                raise ValueError(f"it uses itself ({' → '.join(loops[d.name])})")
            catalog.add(resolve(d, catalog))
        except ValueError as exc:
            notes.append(f"{d.name}: {exc}")
    return notes


def load(run, index, q: Quantity, members) -> dict:
    """{member: (slots, values)}: the formula over the slots every variable has."""
    from ttu_tower.viewer import timeline  # timeline loads derived quantities through here

    resolved = definition_of(q)
    d = resolved.definition
    func = expression.compile_expression(d.formula, _names([v.name for v in d.variables], q.kind), x=False)
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
        values = site_values(q.kind, m) | {v.name: c.y[np.searchsorted(c.slots, slots)]
                                           for v, c in zip(d.variables, curves)}  # a variable hides a constant
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
        if replacing is not None and replacing != d.name:  # what used it follows the new name
            old = f"{TABLE}|{replacing}|"
            self.items = [replace(x, variables=tuple(replace(v, key=d.key) if v.key == old else v
                                                     for v in x.variables)) for x in self.items]
        self.save()

    def remove(self, name: str) -> None:
        self.items = [x for x in self.items if x.name != name]
        self.save()
