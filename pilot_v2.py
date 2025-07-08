from __future__ import annotations

import inspect
import json
import random
import collections.abc as cabc
from pprint import pformat
from typing import (
    Any, Callable, FrozenSet, Tuple, Sequence,
    get_origin, get_args, Container as TypingContainer,
    Union,
)

# ---------------------------------------------------------------------------
# robust, version-independent `get_type_hints`
# ---------------------------------------------------------------------------
try:
    from typing import get_type_hints          # Python ≥3.10
except ImportError:                            # 3.8 / 3.9 + typing-extensions
    from typing_extensions import get_type_hints  # type: ignore

# your repo ------------------------------------------------------------------
import arc_dsl.dsl as dsl
import arc_dsl.arc_types as T        # Grid, Object, …

# ---------------------------------------------------------------------------
# 0. random ARC-ish primitives
# ---------------------------------------------------------------------------
def rand_color() -> int:
    return random.randint(0, 9)


def rand_grid(max_side: int = 5) -> T.Grid:
    h, w = random.randint(1, max_side), random.randint(1, max_side)
    return tuple(tuple(rand_color() for _ in range(w)) for _ in range(h))


def rand_object(max_side: int = 5) -> T.Object:
    g = rand_grid(max_side)
    triples = {
        (pix, (r, c))
        for r, row in enumerate(g)
        for c, pix in enumerate(row)
        if random.random() < 0.35
    }
    return frozenset(triples)


def rand_indices(max_side: int = 5) -> T.Indices:
    h, w = random.randint(1, max_side), random.randint(1, max_side)
    coords = {
        (random.randint(0, h - 1), random.randint(0, w - 1))
        for _ in range(random.randint(1, h * w))
    }
    return frozenset(coords)


def rand_int_tuple() -> Tuple[int, int]:
    return (random.randint(-3, 3), random.randint(-3, 3))


def rand_int_frozenset() -> FrozenSet[int]:
    return frozenset(random.randint(0, 9) for _ in range(random.randint(1, 5)))


def rand_container(depth: int = 1) -> Any:
    """
    A very general container generator:
        depth=1 → tuple of ints
        depth=2 → tuple of tuple of ints
    """
    if depth == 1:
        return tuple(random.randint(0, 9) for _ in range(random.randint(1, 5)))
    return tuple(
        rand_container(depth - 1)
        for _ in range(random.randint(1, 4))
    )


def random_unary_dsl() -> Callable:
    """Return a random DSL function that itself takes exactly one arg."""
    unary = [
        f for _, f in inspect.getmembers(dsl, inspect.isfunction)
        if len(inspect.signature(f).parameters) == 1
           and not f.__name__.startswith('_')
    ]
    return random.choice(unary) if unary else (lambda x: x)

# ---------------------------------------------------------------------------
# 1. value-generator dispatch table
# ---------------------------------------------------------------------------
def variant_generators(name: str, anno: Any) -> list[Callable[[], Any]]:
    """
    Return a *non-empty* list of zero-arg callables that generate admissible
    sample values for the given annotation.
    """
    # --- ARC aliases ---------------------------------------------------------
    if anno in {T.Patch, 'Patch'}:
        return [rand_object, rand_indices]
    if anno in {T.Object, 'Object'}:
        return [rand_object]
    if anno in {T.Indices, 'Indices'}:
        return [rand_indices]
    if anno in {T.Grid, 'Grid'} or 'grid' in name.lower():
        return [lambda: rand_grid(max_side=2), rand_grid]
    if anno in {T.Element, 'Element'}:
        return [rand_grid, rand_object]
    if anno in {T.Piece, 'Piece'}:
        return [rand_grid, rand_object]
    if anno in {T.IntegerSet, 'IntegerSet'}:
        return [rand_int_frozenset]
    if anno in {bool, 'bool', T.Boolean}:
        return [lambda: True, lambda: False]
    if anno in {int, 'int', T.Integer}:
        return [lambda: 0, lambda: random.randint(1, 4)]
    if anno is Callable or get_origin(anno) is Callable or 'Callable' in str(anno):
        # For param-type Callables we still allow them (they aren't returns)
        return [lambda: (lambda x: x), random_unary_dsl]

    # --- typing generics -----------------------------------------------------
    origin = get_origin(anno)
    args = get_args(anno)

    # --- handle Union ---------------------------------------------------------
    if origin is Union:                                    # ①
        gens = []
        for a in args:
            gens.extend(variant_generators(name, a))       # recurse
        return gens or [rand_color]

    # --- plain Container / Any ------------------------------------------------
    if anno in {cabc.Container, TypingContainer}:          # ②
        return [rand_container]

    if anno is Any:                                       # ③
        return [rand_color]

    # --- generics -------------------------------------------------------------
    if origin in {tuple, Tuple}:                           # ④
        return [rand_int_tuple,
                lambda: tuple(rand_color() for _ in range(3))]

    if origin in {frozenset, FrozenSet}:                   # ⑤
        inner = args[0] if args else int
        if inner in {int, 'int', T.Integer}:
            return [rand_int_frozenset]

    if origin in {cabc.Container, TypingContainer}:        # ⑥
        return [rand_container]

    if origin is TypingContainer:
        # Container[...]   → shallow tuple of ints
        return [rand_container]
    if origin is FrozenSet:
        # FrozenSet[int] or similar
        inner = args[0] if args else int
        if inner in {int, 'int', T.Integer}:
            return [rand_int_frozenset]
    if origin is Tuple:
        # Tuple[int, int]  or Tuple[…]
        return [rand_int_tuple, lambda: tuple(rand_color() for _ in range(3))]

    # --- name-based fall-backs ----------------------------------------------
    lname = name.lower()
    if any(k in lname for k in ['obj', 'patch', 'piece']):
        return [rand_object]
    if 'mask' in lname or 'matrix' in lname:
        return [lambda: tuple(tuple(int(bool(rand_color())) for _ in row)
                              for row in rand_grid())]
    if any(k in lname for k in ['coord', 'delta', 'offset']):
        return [rand_int_tuple]

    
    return [rand_color]

# ---------------------------------------------------------------------------
# 2. build one prompt block
# ---------------------------------------------------------------------------
def make_block(func: Callable,
               min_shots: int = 2,
               max_shots: int = 5) -> dict:
    sig = inspect.signature(func)

    param_variants = {
        name: variant_generators(name, param.annotation)
        for name, param in sig.parameters.items()
        if param.default is inspect._empty
           and param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    }

    # assure *every* parameter got at least one generator
    if any(len(v) == 0 for v in param_variants.values()):
        raise RuntimeError("missing generators")

    needed = max(len(v) for v in param_variants.values())
    n_shots = max(needed, random.randint(min_shots, max_shots))

    shots = []
    for i in range(n_shots):
        kwargs = {
            n: (gens[i] if i < len(gens) else random.choice(gens))()
            for n, gens in param_variants.items()
        }
        try:
            out = func(**kwargs)
        except Exception:
            # retry with fresh random values
            kwargs = {n: random.choice(g)() for n, g in param_variants.items()}
            out = func(**kwargs)
        shots.append({"inputs": kwargs, "output": out})

    # -------------------- few-shot prompt text ------------------------------
    demo_lines = []
    for k, s in enumerate(shots, 1):
        demo_lines.append(f"# Example {k}")
        for arg_name, val in s["inputs"].items():
            demo_lines.append(f"{arg_name.upper()} = {pformat(val)}")
        demo_lines.append(f"# Desired → {pformat(s['output'])}")
        demo_lines.append("")

    input_prompt = (
        f"You are given {len(shots)} input/output examples of "
        f"`{func.__name__}` in action.\n\n"
        + "\n".join(demo_lines)
        + "### Now implement solve(I):"
    )

    output_prompt = (
        "def solve(I):\n"
        f"    return {func.__name__}(I)"
    )

    return {
        "name": func.__name__,
        "shots": shots,
        "input_prompt": input_prompt,
        "output_prompt": output_prompt,
    }

# ---------------------------------------------------------------------------
# 3. iterate over DSL, keep *one-arg* & *non-Callable return*
# ---------------------------------------------------------------------------
def main() -> Sequence[dict]:
    blocks = []
    for name, func in inspect.getmembers(dsl, inspect.isfunction):
        if name.startswith('_'):
            continue

        sig = inspect.signature(func)
        #if len([
        #    p for p in sig.parameters.values()
        #    if p.default is inspect._empty
        #       and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        #]) != 1:
        #    continue  # not unary in the strict sense

        # skip functions that *return* a Callable
        ret_anno = sig.return_annotation
        if ret_anno is not inspect._empty and (
            ret_anno is Callable or get_origin(ret_anno) is Callable
        ):
            continue

        try:
            blocks.append(make_block(func))
        except Exception as err:
            # silenced: comment out the next line for verbose debugging
            # print(f"[warn] skipped {name}: {err}")
            pass
    return blocks


if __name__ == "__main__":
    with open("train_set.json", "w") as fp:
        json.dump(main(), fp, indent=2, default=lambda o: pformat(o))
    print("✓ Wrote train_set.json")