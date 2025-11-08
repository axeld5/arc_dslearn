"""Type helpers for analyzing DSL function annotations."""

from __future__ import annotations

import collections.abc as cabc
from functools import lru_cache
from typing import Any, Tuple, Union, get_args, get_origin
from typing import Callable as TypingCallable
from typing import Container as TypingContainer

import src.arc_dslearn.arc_dsl.arc_types as T

# ---------- Type helpers (alias/union aware) ----------


def _is_grid_structural(anno_or_val: Any) -> bool:
    """Tuple[Tuple[int]] structural recognition (besides T.Grid/'Grid'). Accepts types or values."""
    # If value instance of grid-like
    if isinstance(anno_or_val, tuple) and anno_or_val and isinstance(anno_or_val[0], tuple):
        # assume grid of ints
        return True
    anno = anno_or_val
    if anno in (T.Grid, "Grid"):
        return True
    origin = get_origin(anno)
    if origin in {tuple, Tuple}:
        args = get_args(anno)
        if len(args) == 1 and get_origin(args[0]) in {tuple, Tuple}:
            inner = get_args(args[0])
            return len(inner) == 1 and inner[0] in {int, T.Integer, "int", "Integer"}
    return False


def _name(anno: Any) -> str:
    return getattr(anno, "__name__", str(anno))


_ATOMS = {
    "Boolean",
    "Integer",
    "IntegerTuple",
    "IntegerSet",
    "Grid",
    "Cell",
    "Object",
    "Objects",
    "Indices",
    "IndicesSet",
    "Patch",
    "Element",
    "Piece",
}


def _expand_atoms(anno: Any) -> frozenset[str]:
    """Expand aliases and Unions to a small set of 'atomic' names we route on. Handles: structural Grid, named aliases, and typing.Union."""
    # Containers
    if anno in (cabc.Container, TypingContainer, "Container", "ContainerContainer"):
        return frozenset({"Container"})  # soft tag
    origin = get_origin(anno)
    if origin in (cabc.Container, TypingContainer):
        return frozenset({"Container"})

    # Structural Grid first
    if _is_grid_structural(anno):
        return frozenset({"Grid"})

    # typing.Union[...] support
    if get_origin(anno) is Union:
        atoms: set[str] = set()
        for a in get_args(anno):
            atoms |= set(_expand_atoms(a))
        return frozenset(atoms)

    # Known aliases/leaves
    n = _name(anno)
    if n in _ATOMS:
        # Expand composite aliases to their leaves
        if n == "Patch":  # Patch = Union[Object, Indices]
            return frozenset({"Object", "Indices"})
        if n == "Element":  # Element = Union[Object, Grid]
            return frozenset({"Object", "Grid"})
        if n == "Piece":  # Piece = Union[Grid, Patch] = {Grid, Object, Indices}
            return frozenset({"Grid", "Object", "Indices"})
        return frozenset({n})

    # Identity checks for typed aliases from T
    if anno in (T.Boolean, bool):
        return frozenset({"Boolean"})
    if anno in (T.Integer, int):
        return frozenset({"Integer"})
    if anno in (T.IntegerTuple,):
        return frozenset({"IntegerTuple"})
    if anno in (T.IntegerSet,):
        return frozenset({"IntegerSet"})
    if anno in (T.Grid,):
        return frozenset({"Grid"})
    if anno in (T.Cell,):
        return frozenset({"Cell"})
    if anno in (T.Object,):
        return frozenset({"Object"})
    if anno in (T.Objects,):
        return frozenset({"Objects"})
    if anno in (T.Indices,):
        return frozenset({"Indices"})
    if anno in (T.IndicesSet,):
        return frozenset({"IndicesSet"})
    if anno in (T.Patch,):
        return frozenset({"Object", "Indices"})
    if anno in (T.Element,):
        return frozenset({"Object", "Grid"})
    if anno in (T.Piece,):
        return frozenset({"Grid", "Object", "Indices"})

    # Fallback: unknown
    return frozenset()


@lru_cache(maxsize=32768)
def _expand_atoms_cached(anno: Any) -> frozenset[str]:
    """Wrap around _expand_atoms to avoid repeating work."""
    return _expand_atoms(anno)


def _param_atoms(anno: Any) -> frozenset[str]:
    return _expand_atoms_cached(anno)


def _accepts_grid(anno: Any) -> bool:
    """Return True if a parameter annotated with `anno` can accept a Grid."""
    A = _expand_atoms_cached(anno)
    return bool(A) and "Grid" in A


def _produces_grid(anno: Any) -> bool:
    """Return True if a return annotated with `anno` can be a Grid."""
    A = _expand_atoms_cached(anno)
    return bool(A) and "Grid" in A


def overlaps(a: Any, b: Any) -> bool:
    """Return True if types share at least one possible runtime atom (for chaining)."""
    A, B = _expand_atoms_cached(a), _expand_atoms_cached(b)
    return bool(A and B and (A & B))


# ---------- Callable detection ----------


def _is_callable_anno(anno: Any) -> bool:
    """Return True if annotation is typing.Callable *or* collections.abc.Callable, or Union containing it."""
    if anno is TypingCallable or anno is cabc.Callable:
        return True
    origin = get_origin(anno)
    if origin in (TypingCallable, cabc.Callable):
        return True
    if origin is Union:
        return any(_is_callable_anno(a) for a in get_args(anno))
    return isinstance(anno, str) and anno.lower() == "callable"
