"""Random value generators and variant generators for DSL function training data."""

from __future__ import annotations

import collections.abc as cabc
import random
from typing import (
    Any,
    Callable,
    Container,
    FrozenSet,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import src.arc_dslearn.arc_dsl.arc_types as T  # Grid, Object, …


# ---------------------------------------------------------------------------
# Random ARC-ish primitives
# ---------------------------------------------------------------------------
def rand_color() -> int:
    """Generate a random color between 0 and 9."""
    return random.randint(0, 9)


def rand_grid(max_side: int = 5) -> T.Grid:
    """Generate a random grid."""
    h, w = random.randint(1, max_side), random.randint(1, max_side)
    return tuple(tuple(rand_color() for _ in range(w)) for _ in range(h))  # type: ignore[return-value]


def rand_object(max_side: int = 5) -> T.Object:
    """Generate a random object."""
    g = rand_grid(max_side)
    triples = {
        (pix, (r, c))
        for r, row in enumerate(g)
        for c, pix in enumerate(row)
        if random.random() < 0.35
    }
    # Ensure at least one element to avoid empty collections
    if not triples:
        # Add at least one random element
        r, c = random.randint(0, len(g) - 1), random.randint(0, len(g[0]) - 1)
        triples.add((g[r][c], (r, c)))
    return frozenset(triples)


def rand_indices(max_side: int = 5) -> T.Indices:
    """Generate a random set of indices."""
    h, w = random.randint(1, max_side), random.randint(1, max_side)
    coords = {
        (random.randint(0, h - 1), random.randint(0, w - 1))
        for _ in range(random.randint(1, h * w))
    }
    # Ensure at least one coordinate to avoid empty collections
    if not coords:
        coords.add((random.randint(0, h - 1), random.randint(0, w - 1)))
    return frozenset(coords)


def rand_int_tuple() -> Tuple[int, int]:
    """Generate a random tuple of integers."""
    return (random.randint(-3, 3), random.randint(-3, 3))


def rand_int_frozenset() -> FrozenSet[int]:
    """Generate a random frozenset of integers."""
    size = random.randint(1, 5)  # Ensure at least size 1
    result: set[int] = set()
    while len(result) < size:
        result.add(random.randint(0, 9))
        # Prevent infinite loop if trying to get more than 10 unique values
        if len(result) >= 10:
            break
    return frozenset(result)


def rand_container(depth: int = 1) -> Any:
    """Generate a very general container generator: depth=1 → tuple of ints, depth=2 → tuple of tuple of ints."""
    if depth == 1:
        size = random.randint(1, 5)  # Ensure at least size 1
        return tuple(random.randint(0, 9) for _ in range(size))
    size = random.randint(1, 4)  # Ensure at least size 1
    return tuple(rand_container(depth - 1) for _ in range(size))


# ---------------------------------------------------------------------------
# Value-generator dispatch table
# ---------------------------------------------------------------------------
def variant_generators(name: str, anno: Any, func_name: str = "") -> list[Callable[[], Any]]:
    """Return a *non-empty* list of zero-arg callables that generate admissible sample values for the given annotation."""
    # --- ARC aliases ---------------------------------------------------------
    if anno in {T.Patch, "Patch"}:
        # Functions that need non-empty collections
        if func_name in {"bordering", "center", "inbox", "outbox", "corners"}:
            return [lambda: rand_object(max_side=3), lambda: rand_indices(max_side=3)]
        # Functions that need count method - use tuples instead of frozensets
        elif func_name in {"mostcommon", "leastcommon"}:
            return [
                lambda: tuple(rand_color() for _ in range(random.randint(3, 8))),
                lambda: tuple(rand_color() for _ in range(random.randint(4, 10))),
            ]
        return [rand_object, rand_indices]
    if anno in {T.Object, "Object"}:
        # Functions that need non-empty objects
        if func_name in {
            "bordering",
            "center",
            "inbox",
            "outbox",
            "corners",
            "mostcommon",
            "leastcommon",
        }:
            return [lambda: rand_object(max_side=3)]
        return [rand_object]
    if anno in {T.Objects, "Objects"}:
        # Generate a frozenset of objects
        return [
            lambda: frozenset([rand_object(max_side=3) for _ in range(random.randint(1, 3))]),
            lambda: frozenset([rand_object(max_side=2) for _ in range(random.randint(2, 4))]),
        ]
    if anno in {T.Indices, "Indices"}:
        # Functions that need non-empty indices
        if func_name in {
            "bordering",
            "center",
            "inbox",
            "outbox",
            "corners",
            "mostcommon",
            "leastcommon",
        }:
            return [lambda: rand_indices(max_side=3)]
        return [rand_indices]
    if anno in {T.IndicesSet, "IndicesSet"}:
        # Generate a frozenset of indices
        return [
            lambda: frozenset([rand_indices(max_side=3) for _ in range(random.randint(1, 3))]),
            lambda: frozenset([rand_indices(max_side=2) for _ in range(random.randint(2, 4))]),
        ]
    if anno in {T.Grid, "Grid"} or "grid" in name.lower():
        return [lambda: rand_grid(max_side=2), rand_grid]
    if anno in {T.Element, "Element"}:
        return [rand_grid, rand_object]
    if anno in {T.Piece, "Piece"}:
        return [rand_grid, rand_object]
    if anno in {T.IntegerSet, "IntegerSet"}:
        return [rand_int_frozenset]
    if anno in {bool, "bool", T.Boolean}:
        return [lambda: True, lambda: False]
    if anno in {int, "int", T.Integer}:
        # Special case for functions that perform division or modulo operations
        # to avoid division by zero errors
        division_funcs = {"divide", "downscale", "hsplit", "vsplit"}
        if (
            func_name in division_funcs
            or func_name == "interval"
            and ("step" in name.lower() or name.lower() in ["n", "delta", "increment"])
        ):
            return [lambda: random.randint(1, 9), lambda: random.randint(1, 9)]
        else:
            return [lambda: 0, lambda: random.randint(1, 4)]
    if anno in {T.IntegerTuple, "IntegerTuple"} or any(
        k in name.lower() for k in ["coord", "delta", "offset", "direction"]
    ):
        # Always generate exactly 2-element tuples for IntegerTuple
        return [
            lambda: (random.randint(-3, 3), random.randint(-3, 3)),
            lambda: (random.randint(-2, 2), random.randint(-2, 2)),
        ]
    if anno in {T.ContainerContainer, "ContainerContainer"}:
        # Generate container of containers
        return [
            lambda: (rand_container(), rand_container()),
            lambda: (rand_object(), rand_indices(), rand_int_frozenset()),
            lambda: frozenset([rand_object(), rand_indices()]),
        ]
    if anno is Callable or get_origin(anno) is Callable or "Callable" in str(anno):
        # This should not be reached since we skip functions with callable parameters
        pass

    # --- typing generics -----------------------------------------------------
    origin = get_origin(anno)
    args = get_args(anno)

    # --- handle Union ---------------------------------------------------------
    if origin is Union:  # ①
        gens = []
        for a in args:
            gens.extend(variant_generators(name, a, func_name))  # recurse
        return gens or [rand_color]

    # --- plain Container / Any ------------------------------------------------
    if anno in {cabc.Container, Container} or str(anno) == "Container":
        # Functions that need count method - use tuples instead of frozensets
        if func_name in {"mostcommon", "leastcommon"}:
            return [
                lambda: tuple(rand_color() for _ in range(random.randint(3, 8))),
                lambda: tuple(rand_color() for _ in range(random.randint(4, 10))),
                lambda: tuple(rand_color() for _ in range(random.randint(2, 6))),
            ]
        # Functions that need len() on items - use container of containers
        elif func_name == "sizefilter":
            return [
                lambda: (rand_object(), rand_indices(), rand_object()),
                lambda: (rand_container(2), rand_container(2), rand_grid()),
                lambda: frozenset([rand_object(), rand_indices(), rand_grid()]),
            ]
        return [rand_container, rand_object, rand_indices, rand_int_frozenset]

    if anno is Any:  # ③
        return [rand_color]

    # --- generics -------------------------------------------------------------
    if origin in {tuple, Tuple}:  # ④
        return [
            lambda: (random.randint(-3, 3), random.randint(-3, 3)),
            lambda: tuple(rand_color() for _ in range(3)),
        ]

    if origin in {frozenset, FrozenSet}:  # ⑤
        inner = args[0] if args else int
        if inner in {int, "int", T.Integer}:
            return [rand_int_frozenset]

    if origin in {cabc.Container, Container}:  # ⑥
        # Container[...]   → shallow tuple of ints
        return [rand_container]
    if origin is FrozenSet:
        # FrozenSet[int] or similar
        inner = args[0] if args else int
        if inner in {int, "int", T.Integer}:
            return [rand_int_frozenset]
    if origin is Tuple:
        # Tuple[int, int]  or Tuple[…]
        return [
            lambda: (random.randint(-3, 3), random.randint(-3, 3)),
            lambda: tuple(rand_color() for _ in range(3)),
        ]

    # --- name-based fall-backs ----------------------------------------------
    lname = name.lower()
    if any(k in lname for k in ["obj", "patch", "piece"]):
        return [rand_object]
    if "mask" in lname or "matrix" in lname:
        return [lambda: tuple(tuple(int(bool(rand_color())) for _ in row) for row in rand_grid())]
    print(f"name: {name}")
    return [rand_color]
