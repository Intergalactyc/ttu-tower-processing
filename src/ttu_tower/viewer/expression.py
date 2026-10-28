"""Formulas typed by hand, y in terms of x, evaluated with numpy: numbers, x,
+ - * / ^ (or **), brackets, a few functions, pi, e, and any further variables
the caller names (e.g. boom heights). Parsed to a syntax tree and checked node
by node, so nothing but arithmetic ever runs.
"""
import ast
from typing import Callable

import numpy as np

FUNCTIONS = {"exp": np.exp, "ln": np.log, "log": np.log, "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs,
             "tanh": np.tanh, "sin": np.sin, "cos": np.cos, "tan": np.tan, "arctan": np.arctan, "sinh": np.sinh,
             "cosh": np.cosh}
CONSTANTS = {"pi": np.pi, "e": np.e}
_BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
_UNARY = {ast.UAdd: np.positive, ast.USub: np.negative}


def _clean(text: str) -> str:
    text = text.strip()
    if text.lower().startswith("y") and "=" in text[:3]:
        text = text.split("=", 1)[1]
    for a, b in (("^", "**"), ("·", "*"), ("×", "*"), ("−", "-"), ("–", "-")):
        text = text.replace(a, b)
    return text.strip()  # a leading space is an indentation error to the parser


def _check(node, variables=()) -> None:
    if isinstance(node, ast.Expression):
        _check(node.body, variables)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY:
            raise ValueError(f"'{ast.unparse(node)}': only + - * / ^")
        _check(node.left, variables)
        _check(node.right, variables)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY:
            raise ValueError(f"'{ast.unparse(node)}': only + and - before a value")
        _check(node.operand, variables)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"'{ast.unparse(node)}' isn't a number")
    elif isinstance(node, ast.Name):
        if node.id not in CONSTANTS and node.id not in variables:
            raise ValueError(f"unknown name '{node.id}' ({', '.join((*variables, *CONSTANTS))}, or a function: "
                             f"{', '.join(FUNCTIONS)})")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise ValueError(f"'{ast.unparse(node.func)}' isn't one of the functions: {', '.join(FUNCTIONS)}")
        if node.keywords or len(node.args) != 1:
            raise ValueError(f"{node.func.id}() takes one value")
        _check(node.args[0], variables)
    else:
        raise ValueError(f"'{ast.unparse(node)}' isn't arithmetic")


def _evaluate(node, names: dict):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, names)
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](_evaluate(node.left, names), _evaluate(node.right, names))
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_evaluate(node.operand, names))
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return names[node.id] if node.id in names else CONSTANTS[node.id]
    return FUNCTIONS[node.func.id](_evaluate(node.args[0], names))


def compile_expression(text: str, variables=(), x: bool = True) -> Callable[..., np.ndarray]:
    """y(x, **values) for a formula such as "0.18*x^0.5" (or "y = ..."), in x and any of `variables`
    (each a number, or an array matching x); ValueError, with the reason, otherwise. The function's
    `names` are the variables the formula uses. Without `x`, y(**values) in the variables alone.
    """
    source = _clean(text)
    if not source:
        raise ValueError("empty")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        raise ValueError("can't read it (products need *, e.g. 2*x)") from None
    _check(tree, (("x",) if x else ()) + tuple(variables))

    def func(x, **values):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(_evaluate(tree, {**values, "x": x}), dtype=float), x.shape).copy()

    def of_values(**values):
        shape = np.broadcast_shapes(*(np.shape(v) for v in values.values()))
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(_evaluate(tree, values), dtype=float), shape).copy()

    made = func if x else of_values
    made.names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in variables}
    return made


def parse_bound(text: str, default: float) -> float:
    text = _clean(text)
    if not text:
        return default
    value = compile_expression(text)(np.full(1, np.nan))[0]  # a number, or arithmetic on numbers and constants
    if np.isnan(value):
        raise ValueError(f"'{text}' isn't a number")
    return float(value)


def piece_of(x: np.ndarray, ranges) -> np.ndarray:
    """Which of [(lo, hi)] covers each x (the first that does), -1 for none."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, -1)
    for i, (lo, hi) in enumerate(ranges):
        out[(out < 0) & (x >= lo) & (x < hi)] = i
    return out


def piecewise(pieces) -> Callable[..., np.ndarray]:
    """[(func, lo, hi)]: each func over lo <= x < hi; where pieces overlap, the first one listed.
    Further values (numbers, or arrays matching x) pass to every piece.
    """
    def func(x, **values):
        x = np.asarray(x, dtype=float)
        y = np.full(x.shape, np.nan)
        free = np.ones(x.shape, dtype=bool)
        for f, lo, hi in pieces:
            on = free & (x >= lo) & (x < hi)
            if on.any():
                y[on] = f(x[on], **{k: v[on] if np.ndim(v) else v for k, v in values.items()})
                free &= ~on
        return y

    func.names = set().union(*(getattr(f, "names", set()) for f, _, _ in pieces))
    return func
