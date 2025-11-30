"""Role templates and micro-templates for HOF (Higher-Order Function) patterns."""

from __future__ import annotations

from dataclasses import dataclass

# ---------- Role-based callable constant pools ----------
# Based on corpus analysis of solvers.py: curate constants by their semantic role
# Usage counts extracted from actual solver patterns

# Priority ordering based on usage frequency in solvers.py:
# - Higher frequency = earlier in list = more likely to be sampled first

_SELECTOR_CALLABLES = [
    # High usage (50+)
    "first",  # 76 uses
    "color",  # 278 uses - very common for recoloring
    "size",  # 23 uses
    # Medium usage (10-50)
    "width",  # 31 uses
    "height",  # 27 uses
    "center",  # 16 uses
    "last",  # 17 uses
    # Lower usage
    "ulcorner",
    "urcorner",
    "llcorner",
    "lrcorner",
]

_PREDICATE_CALLABLES = [
    # High usage predicates
    "equality",  # 26 uses - common in fork patterns
    "greater",  # used for comparisons
    "hline",  # line detection
    "vline",  # line detection
    "even",
    "positive",
]

_GEOMETRY_CALLABLES = [
    # High usage (50+)
    "shift",  # 64 uses - very common
    # Medium usage (10-50)
    "neighbors",  # 17 uses
    "box",  # 17 uses
    "connect",  # 16 uses
    "delta",  # used for differences
    # Lower usage
    "dneighbors",
    "frontiers",  # 3 uses
    "corners",  # 3 uses
    "outbox",  # 4 uses
    "shoot",
]

_SCORER_ENDING_CALLABLES = [
    # Ordered by usage in argmax/argmin/sfilter patterns
    "size",  # very common scorer
    "width",
    "height",
    "color",
    "equality",
    "greater",
]

# Callables commonly used as combinator arguments (for fork/chain/compose)
# These are unary functions that transform objects/cells
# Ordered by actual usage frequency in solvers.py
_UNARY_TRANSFORM_CALLABLES = [
    # Very high usage (50+)
    "color",  # 278 uses - extremely common
    "identity",  # 70 uses - often in fork branches
    "first",  # 76 uses
    # High usage (20-50)
    "vmirror",  # 32 uses
    "hmirror",  # 28 uses
    "width",  # 31 uses
    "height",  # 27 uses
    "rot90",  # 24 uses
    "size",  # 23 uses
    "normalize",  # 21 uses
    # Medium usage (10-20)
    "box",  # 17 uses
    "center",  # 16 uses
    "last",  # 17 uses
    "dmirror",  # 15 uses
    "rot180",  # 14 uses
    # Lower usage (still useful)
    "rot270",
    "flip",
    "cmirror",
    "backdrop",  # 4 uses
    "toindices",  # 9 uses
    "ulcorner",
    "urcorner",
    "llcorner",
    "lrcorner",
    "vfrontier",
    "hfrontier",
    "centerofmass",
]

# Binary functions commonly used as outer function in fork(outer, a, b)
# Ordered by actual usage frequency in solvers.py
_BINARY_COMBINER_CALLABLES = [
    # Very high usage (40+)
    "difference",  # 45 uses - very common
    "combine",  # 39 uses
    "equality",  # 26 uses - common in fork predicates
    # Medium usage (10-40)
    "recolor",  # 19 uses
    "intersection",  # 17 uses
    "connect",  # 16 uses
    # Also used but less common
    "fill",
    "paint",
    "multiply",
    "add",
    "subtract",
    "greater",
    "either",
    "both",
]


def _is_selector_callable(name: str) -> bool:
    return name in _SELECTOR_CALLABLES


def _is_predicate_callable(name: str) -> bool:
    return name in _PREDICATE_CALLABLES


def _is_geometry_callable(name: str) -> bool:
    return name in _GEOMETRY_CALLABLES


def _is_scorer_ending(name: str) -> bool:
    return name in _SCORER_ENDING_CALLABLES


def _is_unary_transform(name: str) -> bool:
    """Check if a callable is a common unary transform (for fork/chain/compose branches)."""
    return name in _UNARY_TRANSFORM_CALLABLES


def _is_binary_combiner(name: str) -> bool:
    """Check if a callable is a common binary combiner (for fork outer function)."""
    return name in _BINARY_COMBINER_CALLABLES


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
    # rbind/lbind can feed into combinators (producing complex callables)
    _MicroTemplate("rbind", "compose"),
    _MicroTemplate("lbind", "compose"),
    _MicroTemplate("rbind", "fork"),
    _MicroTemplate("lbind", "fork"),
    # fork patterns (very common in solvers.py)
    _MicroTemplate("fork", "mapply"),  # fork(recolor, color, backdrop) + mapply
    _MicroTemplate("fork", "apply"),
    _MicroTemplate("fork", "extract"),  # extract(objects, fork(...))
    # compose patterns
    _MicroTemplate("compose", "mapply"),
    _MicroTemplate("compose", "apply"),
    _MicroTemplate("compose", "argmax"),
    _MicroTemplate("compose", "argmin"),
    _MicroTemplate("compose", "sfilter"),
    _MicroTemplate("compose", "matcher"),
    _MicroTemplate("compose", "order"),
    # chain patterns
    _MicroTemplate("chain", "mapply"),
    _MicroTemplate("chain", "apply"),
    _MicroTemplate("chain", "argmax"),
    _MicroTemplate("chain", "argmin"),
    _MicroTemplate("chain", "sfilter"),
    _MicroTemplate("chain", "matcher"),
    # power patterns
    _MicroTemplate("power", "apply"),
    _MicroTemplate("power", "mapply"),
    # branch + immediate apply
    _MicroTemplate("branch", "apply"),
    _MicroTemplate("branch", "mapply"),
    # Extractor patterns for two-patch functions (manhattan, adjacent, gravitate, etc.)
    # After extracting individual objects, these functions become usable
    _MicroTemplate("first", "manhattan"),
    _MicroTemplate("first", "adjacent"),
    _MicroTemplate("first", "gravitate"),
    _MicroTemplate("last", "manhattan"),
    _MicroTemplate("last", "adjacent"),
    _MicroTemplate("last", "gravitate"),
    _MicroTemplate("argmax", "manhattan"),
    _MicroTemplate("argmax", "adjacent"),
    _MicroTemplate("argmax", "gravitate"),
    _MicroTemplate("argmin", "manhattan"),
    _MicroTemplate("argmin", "adjacent"),
    _MicroTemplate("argmin", "gravitate"),
]


def get_hof_micro_templates() -> list[_MicroTemplate]:
    """Get all HOF micro-templates."""
    return _HOF_MICRO_TEMPLATES


# Common source/sink pairs from corpus
# Based on actual usage patterns in solvers.py (counts shown)
_SOURCE_SINK_PAIRS: dict[str, list[str]] = {
    # High-priority sources -> most common sinks first
    # objects (237 uses) -> often followed by these operations
    "objects": ["mapply", "first", "argmax", "sfilter", "argmin", "extract", "order", "last"],
    # ofcolor (175 uses) -> common operations on color sets
    "ofcolor": ["mapply", "fill", "sfilter", "paint", "underfill", "extract", "order"],
    # partition (26 uses) and fgpartition (16 uses)
    "partition": ["argmax", "argmin", "sfilter", "first", "mapply", "extract", "order"],
    "fgpartition": ["mapply", "sfilter", "argmax", "first", "last", "extract", "order"],
    # Medium usage sources
    "toindices": ["fill", "paint", "underfill", "cover", "mapply"],
    "normalize": ["fill", "paint", "cover", "shift"],
    "frontiers": ["fill", "paint", "sfilter", "extract"],
    "corners": ["fill", "paint", "sfilter", "extract"],
    "outbox": ["fill", "paint", "sfilter"],
    "shoot": ["connect", "fill", "paint"],
    "gravitate": ["shift", "connect", "fill"],
    "center": ["shift", "shoot", "connect"],
    # tuple/product producers bias toward pair applicators
    "pair": ["papply", "prapply", "mpapply", "connect", "shift"],
    "product": ["papply", "prapply", "mpapply", "connect", "fill"],
    "interval": ["papply", "prapply"],
    # Combinator functions (very high usage!) produce callables that should be applied
    # fork (277), compose (268), chain (126) - most critical patterns
    "fork": ["mapply", "apply", "sfilter", "argmax", "argmin", "matcher", "order"],
    "compose": ["mapply", "apply", "sfilter", "argmax", "argmin", "matcher", "order"],
    "chain": ["mapply", "apply", "sfilter", "argmax", "argmin", "matcher"],
    # rbind (209) / lbind (240) produce callables - very common binding patterns
    "lbind": ["mapply", "apply", "compose", "fork", "chain", "power"],
    "rbind": ["mapply", "apply", "compose", "fork", "chain", "power"],
    # branch (79 uses) produces callable
    "branch": ["apply", "mapply"],
    # matcher (48 uses) produces predicates for filtering
    "matcher": ["sfilter", "extract", "mfilter"],
    # Extractors (first/last/argmax/argmin) produce individual patches -> enable two-patch functions
    "first": ["shift", "connect", "gravitate", "position", "manhattan", "adjacent", "normalize"],
    "last": ["shift", "connect", "gravitate", "position", "manhattan", "adjacent"],
    "argmax": ["shift", "connect", "normalize", "gravitate", "position", "manhattan", "adjacent"],
    "argmin": ["shift", "connect", "normalize", "gravitate", "position", "manhattan", "adjacent"],
    "extract": ["shift", "connect", "normalize", "gravitate", "manhattan", "adjacent"],
    # sfilter (73 uses) often produces filtered sets for further operations
    "sfilter": ["mapply", "first", "argmax", "merge", "fill", "paint"],
    # merge (71 uses) often followed by fill/paint operations
    "merge": ["fill", "paint", "underfill", "cover"],
    # difference (45 uses) often followed by fill
    "difference": ["fill", "paint", "underfill"],
}


def get_source_sink_pairs() -> dict[str, list[str]]:
    """Get source/sink pair mappings."""
    return _SOURCE_SINK_PAIRS


# ---------- Role templates (soft bias) ----------
# Roles: 'var_callable', 'const_callable', 'var_non_callable', 'var_tuple'

_ROLE_TEMPLATES: dict[str, list[tuple[str, ...]]] = {
    # Combinator functions: allow BOTH var_callable AND const_callable (DSL functions directly)
    # This matches solver patterns like fork(equality, dmirror, identity)
    "compose": [
        ("const_callable", "const_callable"),  # compose(size, color) - common pattern
        ("var_callable", "var_callable"),  # compose with prior computed callables
        ("const_callable", "var_callable"),  # mixed
        ("var_callable", "const_callable"),
    ],
    "chain": [
        ("const_callable", "const_callable", "const_callable"),  # chain(h, g, f) with DSL fns
        ("var_callable", "var_callable", "var_callable"),
        ("const_callable", "var_callable", "var_callable"),
    ],
    "fork": [
        ("const_callable", "const_callable", "const_callable"),  # fork(equality, dmirror, identity)
        ("var_callable", "const_callable", "const_callable"),  # fork with computed outer
        ("const_callable", "var_callable", "var_callable"),  # fork with computed branches
        ("var_callable", "var_callable", "var_callable"),
    ],
    "power": [("var_callable", "var_non_callable"), ("const_callable", "var_non_callable")],
    "lbind": [("const_callable", "var_non_callable")],
    "rbind": [("const_callable", "var_non_callable")],
    "apply": [("const_callable", "var_non_callable"), ("var_callable", "var_non_callable")],
    "order": [("var_non_callable", "const_callable")],
    "papply": [("const_callable", "var_tuple", "var_tuple")],
    "mpapply": [("const_callable", "var_tuple", "var_tuple")],
    "mapply": [("const_callable", "var_non_callable"), ("var_callable", "var_non_callable")],
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
    # Compose/chain - outer function (p_index=0) can be scorer or transform
    if fn_name in {"compose", "chain"} and p_index == 0:
        return "scorer"  # outer function is often a scorer
    # Compose/chain - inner functions are typically unary transforms
    if fn_name == "compose" and p_index == 1:
        return "unary_transform"
    if fn_name == "chain" and p_index in {1, 2}:
        return "unary_transform"
    # Fork: outer is binary combiner, branches are unary transforms
    if fn_name == "fork":
        if p_index == 0:
            return "binary_combiner"  # fork(outer, a, b) - outer must be binary
        else:
            return "unary_transform"  # a and b are unary
    # For mapply/apply on indices/objects, prefer geometry
    # (This is contextual, we handle it elsewhere)
    return None


# Export all role-related utilities
__all__ = [
    "_is_selector_callable",
    "_is_predicate_callable",
    "_is_geometry_callable",
    "_is_scorer_ending",
    "_is_unary_transform",
    "_is_binary_combiner",
    "get_desired_arity",
    "get_hof_micro_templates",
    "get_source_sink_pairs",
    "get_role_templates",
    "get_role_hint",
    "_MicroTemplate",
]
