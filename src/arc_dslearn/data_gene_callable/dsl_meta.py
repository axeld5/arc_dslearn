"""DSL introspection: constants, functions, and metadata collection."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from typing import Callable as TypingCallable

import src.arc_dslearn.arc_dsl.constants as C
import src.arc_dslearn.arc_dsl.dsl as dsl
from src.arc_dslearn.data_gene_callable.types_utils import (
    _accepts_grid,
    _expand_atoms_cached,
    _is_callable_anno,
    _produces_grid,
)

# ---------- Constants ----------

_CONSTANTS: dict[str, Any] = {
    # bools
    "F": C.F,
    "T": C.T,
    # ints
    "ZERO": C.ZERO,
    "ONE": C.ONE,
    "TWO": C.TWO,
    "THREE": C.THREE,
    "FOUR": C.FOUR,
    "FIVE": C.FIVE,
    "SIX": C.SIX,
    "SEVEN": C.SEVEN,
    "EIGHT": C.EIGHT,
    "NINE": C.NINE,
    "TEN": C.TEN,
    "NEG_ONE": C.NEG_ONE,
    "NEG_TWO": C.NEG_TWO,
    # tuples
    "DOWN": C.DOWN,
    "RIGHT": C.RIGHT,
    "UP": C.UP,
    "LEFT": C.LEFT,
    "ORIGIN": C.ORIGIN,
    "UNITY": C.UNITY,
    "NEG_UNITY": C.NEG_UNITY,
    "UP_RIGHT": C.UP_RIGHT,
    "DOWN_LEFT": C.DOWN_LEFT,
    "ZERO_BY_TWO": C.ZERO_BY_TWO,
    "TWO_BY_ZERO": C.TWO_BY_ZERO,
    "TWO_BY_TWO": C.TWO_BY_TWO,
    "THREE_BY_THREE": C.THREE_BY_THREE,
}


def _public_dsl_functions() -> list[tuple[str, TypingCallable[..., Any]]]:
    """Cache public DSL callables once."""
    return [(n, f) for n, f in inspect.getmembers(dsl, inspect.isfunction) if not n.startswith("_")]


_DSL_FUNCS: list[tuple[str, TypingCallable[..., Any]]] = _public_dsl_functions()
_CALLABLE_CONSTS: list[tuple[str, TypingCallable[..., Any]]] = _DSL_FUNCS[:]


def dsl_functions_summary() -> str:
    """Return a one-line bullet list of DSL functions."""
    return "\n".join(f"- {name}" for name, _ in _DSL_FUNCS)


DSL_FUNCTIONS_BLOCK = dsl_functions_summary()


# ---------- Constant classification ----------


def _classify_constant_atoms(val: Any) -> set[str]:
    """Return which atom kinds a constant value can satisfy."""
    if isinstance(val, bool):
        return {"Boolean"}
    if isinstance(val, int):
        return {"Integer"}
    if isinstance(val, tuple) and len(val) == 2 and all(isinstance(x, int) for x in val):
        return {"IntegerTuple"}
    return set()


# Build atom → [(name, value)] lookup for constants we can use
_CONST_BY_ATOM: dict[str, list[tuple[str, Any]]] = {}
for cname, cval in _CONSTANTS.items():
    for a in _classify_constant_atoms(cval):
        _CONST_BY_ATOM.setdefault(a, []).append((cname, cval))


def _pick_constant_for_annotation(anno: Any, rng) -> tuple[str, Any] | None:
    """Pick a (name, value) constant suitable for the given annotation (non-Grid)."""
    A = set(_expand_atoms_cached(anno))
    A.discard("Grid")  # do not feed Grid constants for non-flow parameters
    # Try in a deterministic order but randomized within each atom group
    for atom in ("Boolean", "Integer", "IntegerTuple"):
        if atom in A and atom in _CONST_BY_ATOM and _CONST_BY_ATOM[atom]:
            return rng.choice(_CONST_BY_ATOM[atom])
    return None


# ---------- Callable arity detection ----------


@lru_cache(maxsize=4096)
def _callable_arity(fn) -> int | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    req = 0
    for p in sig.parameters.values():
        if (
            p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect._empty
        ):
            req += 1
    return req


_UNARY_CALLABLE_CONSTS, _MULTIARG_CALLABLE_CONSTS = [], []
for cname, cval in _CALLABLE_CONSTS:
    ar = _callable_arity(cval)
    if ar == 1:
        _UNARY_CALLABLE_CONSTS.append((cname, cval))
    elif ar and ar > 1:
        _MULTIARG_CALLABLE_CONSTS.append((cname, cval))


# ---------- Function metadata ----------


@dataclass(frozen=True, slots=True)
class _ParamSpec:
    name: str
    anno: Any
    required: bool  # True if no default


@dataclass(frozen=True, slots=True)
class _FnMeta:
    name: str
    func: TypingCallable[..., Any]
    params: tuple[_ParamSpec, ...]  # ordered positional-or-kw params
    flow_positions: tuple[int, ...]  # indices whose annotation accepts Grid (may be empty)
    produces_grid: bool
    return_atoms: frozenset[str]  # expanded atoms of return type
    accepts_grid: bool  # any param can take a Grid
    returns_callable: bool  # return is a Callable


def _collect_fn_meta() -> list[_FnMeta]:
    metas: list[_FnMeta] = []
    for name, func in _DSL_FUNCS:
        sig = inspect.signature(func)
        raw_params = [
            p for p in sig.parameters.values() if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        if not raw_params:
            continue
        # All params must have annotations for us to reason about them
        if any(p.annotation is inspect._empty for p in raw_params):
            continue
        ret = sig.return_annotation
        if ret is inspect._empty:
            continue

        params: list[_ParamSpec] = []
        for p in raw_params:
            params.append(
                _ParamSpec(
                    name=p.name,
                    anno=p.annotation,
                    required=(p.default is inspect._empty),
                )
            )

        flow_idxs = tuple(i for i, sp in enumerate(params) if _accepts_grid(sp.anno))
        returns_callable = _is_callable_anno(ret)

        metas.append(
            _FnMeta(
                name=name,
                func=func,
                params=tuple(params),
                flow_positions=flow_idxs,
                produces_grid=_produces_grid(ret),
                return_atoms=_expand_atoms_cached(ret),
                accepts_grid=bool(flow_idxs),
                returns_callable=returns_callable,
            )
        )
    return metas


_FN_META: list[_FnMeta] = _collect_fn_meta()
_ALL_IDX: list[int] = list(range(len(_FN_META)))

# Precompute simple cost model to downweight heavy operators during sampling
_HEAVY_NAMES = {
    "objects",
    "frontiers",
    "occurrences",
    "compress",
    "upscale",
    "downscale",
    "hsplit",
    "vsplit",
}
_COST: dict[int, int] = {i: (3 if _FN_META[i].name in _HEAVY_NAMES else 1) for i in _ALL_IDX}


def _weighted_shuffle(ids: list[int], rng) -> list[int]:
    """Sample without replacement using weights inversely proportional to cost."""
    if not ids:
        return ids
    weights = [1.0 / max(1, _COST[i]) for i in ids]
    # Efraimidis–Spirakis
    keys = [rng.random() ** (1.0 / w) for w in weights]
    return [i for _, i in sorted(zip(keys, ids, strict=False), reverse=True)]


# ---------- Exports ----------


def get_constants() -> dict[str, Any]:
    """Get all DSL constants."""
    return _CONSTANTS


def get_const_by_atom() -> dict[str, list[tuple[str, Any]]]:
    """Get constants organized by atom type."""
    return _CONST_BY_ATOM


def get_callable_consts() -> list[tuple[str, TypingCallable[..., Any]]]:
    """Get all callable constants."""
    return _CALLABLE_CONSTS


def get_unary_callable_consts() -> list[tuple[str, TypingCallable[..., Any]]]:
    """Get unary callable constants."""
    return _UNARY_CALLABLE_CONSTS


def get_fn_meta() -> list[_FnMeta]:
    """Get function metadata."""
    return _FN_META


def get_all_idx() -> list[int]:
    """Get all function indices."""
    return _ALL_IDX


def get_heavy_names() -> set[str]:
    """Get set of heavy operation names."""
    return _HEAVY_NAMES


def _is_heavy_meta(mi: int) -> bool:
    return _FN_META[mi].name in _HEAVY_NAMES


def _meta_wants_callable(mi: int) -> bool:
    return any(_is_callable_anno(sp.anno) for sp in _FN_META[mi].params)


def _produces_tuple_type(meta: _FnMeta) -> bool:
    """Check if a function produces tuple-like output (IntegerTuple, pairs, products)."""
    return "IntegerTuple" in meta.return_atoms


def _is_pair_applicator(mi: int) -> bool:
    """Check if this function applies operations to pairs/tuples."""
    return _FN_META[mi].name in {"papply", "prapply", "mpapply"}
