"""Role templates and micro-templates for HOF (Higher-Order Function) patterns."""

from __future__ import annotations

from dataclasses import dataclass

# ---------- Role-based callable constant pools ----------
# Based on corpus analysis: curate constants by their semantic role

_SELECTOR_CALLABLES = [
    "center",
    "ulcorner",
    "urcorner",
    "llcorner",
    "lrcorner",
    "color",
    "size",
    "width",
    "height",
    "first",
    "last",
]

_PREDICATE_CALLABLES = [
    "hline",
    "vline",
    "even",
    "positive",
    "equality",
    "greater",
]

_GEOMETRY_CALLABLES = [
    "neighbors",
    "dneighbors",
    "frontiers",
    "corners",
    "outbox",
    "box",
    "shoot",
    "shift",
    "delta",
]

_SCORER_ENDING_CALLABLES = ["size", "width", "height", "color", "equality", "greater"]


def _is_selector_callable(name: str) -> bool:
    return name in _SELECTOR_CALLABLES


def _is_predicate_callable(name: str) -> bool:
    return name in _PREDICATE_CALLABLES


def _is_geometry_callable(name: str) -> bool:
    return name in _GEOMETRY_CALLABLES


def _is_scorer_ending(name: str) -> bool:
    return name in _SCORER_ENDING_CALLABLES


_KEYLIKE_PARAM_NAMES = {"compfunc", "condition", "key", "predicate", "proj", "fn"}

# desired callable arities per (fn_name, param_index)
# (if param names differ, index is safest)
_DESIRED_ARITY: dict[str, dict[int, int]] = {
    # returns unary: outer(inner(x))
    "compose": {0: 1, 1: 1},
    # h(g(f(x)))
    "chain": {0: 1, 1: 1, 2: 1},
    # outer(a(x), b(x))  => outer must be binary, a & b unary
    "fork": {0: 2, 1: 1, 2: 1},
    # power(function, n) => function is unary
    "power": {0: 1},
    # also help the consumers:
    "apply": {0: 1},
    "mapply": {0: 1},
    "rapply": {},  # its first arg is a container of callables
    "papply": {0: 2},
    "mpapply": {0: 2},
    "prapply": {0: 2},
    "order": {1: 1},
    "valmax": {1: 1},
    "valmin": {1: 1},
    "sfilter": {1: 1},
    "argmax": {1: 1},
    "argmin": {1: 1},
    "matcher": {0: 1},
}


def get_desired_arity(fn_name: str, p_index: int) -> int | None:
    """Get desired arity for a specific function parameter."""
    return _DESIRED_ARITY.get(fn_name, {}).get(p_index)


# ---------- Micro-templates for top HOF idioms ----------
# These are common patterns found in the corpus that should be sampled as units


@dataclass(frozen=True, slots=True)
class _MicroTemplate:
    """A 2-step template: producer function -> consumer function."""

    producer: str  # e.g., "rbind", "fork", "compose"
    consumer: str  # e.g., "apply", "mapply", "argmax"


_HOF_MICRO_TEMPLATES = [
    # rbind/lbind + apply/mapply
    _MicroTemplate("rbind", "apply"),
    _MicroTemplate("rbind", "mapply"),
    _MicroTemplate("lbind", "apply"),
    _MicroTemplate("lbind", "mapply"),
    # fork + mapply
    _MicroTemplate("fork", "mapply"),
    # compose/chain + argmax/sfilter/matcher
    _MicroTemplate("compose", "argmax"),
    _MicroTemplate("compose", "argmin"),
    _MicroTemplate("compose", "sfilter"),
    _MicroTemplate("compose", "matcher"),
    _MicroTemplate("chain", "argmax"),
    _MicroTemplate("chain", "argmin"),
    _MicroTemplate("chain", "sfilter"),
    # branch + immediate apply
    _MicroTemplate("branch", "apply"),
]


def get_hof_micro_templates() -> list[_MicroTemplate]:
    """Get all HOF micro-templates."""
    return _HOF_MICRO_TEMPLATES


# Common source/sink pairs from corpus
_SOURCE_SINK_PAIRS: dict[str, list[str]] = {
    # If we just produced these sources, prefer these sinks next
    "ofcolor": ["mapply", "sfilter", "extract", "order", "fill", "paint"],
    "objects": ["mapply", "sfilter", "extract", "order", "argmax", "argmin"],
    "toindices": ["fill", "paint", "underfill", "cover", "mapply"],
    "fgpartition": ["mapply", "sfilter", "extract", "order", "argmax"],
    "normalize": ["fill", "paint", "cover"],
    "frontiers": ["fill", "paint", "sfilter", "extract"],
    "corners": ["fill", "paint", "sfilter", "extract"],
    "outbox": ["fill", "paint", "sfilter"],
    "shoot": ["connect", "fill", "paint"],
    "gravitate": ["connect", "shift", "fill"],
    "center": ["shoot", "shift", "connect"],
    # tuple/product producers bias toward pair applicators
    "pair": ["papply", "prapply", "mpapply", "connect", "shift"],
    "product": ["papply", "prapply", "mpapply", "connect", "fill"],
    "interval": ["papply", "prapply"],
}


def get_source_sink_pairs() -> dict[str, list[str]]:
    """Get source/sink pair mappings."""
    return _SOURCE_SINK_PAIRS


# ---------- Role templates (soft bias) ----------
# Roles: 'var_callable', 'const_callable', 'var_non_callable', 'var_tuple'

_ROLE_TEMPLATES: dict[str, list[tuple[str, ...]]] = {
    "compose": [("var_callable", "var_callable")],
    "chain": [("var_callable", "var_callable", "var_callable")],
    "power": [("var_callable", "var_non_callable")],  # second is int exponent
    "lbind": [("const_callable", "var_non_callable")],
    "rbind": [("const_callable", "var_non_callable")],
    "fork": [("var_callable", "var_callable", "var_callable")],
    "apply": [("const_callable", "var_non_callable")],
    "order": [("var_non_callable", "const_callable")],
    "papply": [("const_callable", "var_tuple", "var_tuple")],
    "mpapply": [("const_callable", "var_tuple", "var_tuple")],
    "mapply": [("const_callable", "var_non_callable")],
}


def get_role_templates() -> dict[str, list[tuple[str, ...]]]:
    """Get role templates for functions."""
    return _ROLE_TEMPLATES


def get_role_hint(fn_name: str, p_index: int) -> str | None:
    """Determine the semantic role hint for a callable parameter based on function and parameter index."""
    # Scorer functions (argmax, argmin, sfilter, order, valmax, valmin) - second param
    if fn_name in {"argmax", "argmin", "sfilter", "order", "valmax", "valmin"} and p_index == 1:
        return "scorer"
    # Predicate functions (mfilter, extract when second param) - second param
    if fn_name in {"mfilter"} and p_index == 1:
        return "predicate"
    # Compose/chain/fork - prefer scorers at the end for common idioms
    if fn_name in {"compose", "chain"} and p_index == 0:
        return "scorer"  # outer function is often a scorer
    # For mapply/apply on indices/objects, prefer geometry
    # (This is contextual, we handle it elsewhere)
    return None


# Export all role-related utilities
__all__ = [
    "_is_selector_callable",
    "_is_predicate_callable",
    "_is_geometry_callable",
    "_is_scorer_ending",
    "get_desired_arity",
    "get_hof_micro_templates",
    "get_source_sink_pairs",
    "get_role_templates",
    "get_role_hint",
    "_MicroTemplate",
]
