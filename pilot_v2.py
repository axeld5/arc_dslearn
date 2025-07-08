from __future__ import annotations

import inspect
import json
import random
from pprint import pformat
from typing import Any, Callable, List, Sequence, get_origin, get_args

# ---------------------------------------------------------------------------
# 0. safe, version-independent `get_type_hints`
# ---------------------------------------------------------------------------
try:                      # Py ≥ 3.10
    from typing import get_type_hints        # noqa: F401
except ImportError:       # Py 3.8/3.9 + typing-extensions
    from typing_extensions import get_type_hints  # type: ignore

# third-party (your repo) ----------------------------------------------------
import arc_dsl.dsl as dsl
import arc_dsl.arc_types as T

# ---------------------------------------------------------------------------
# 1.   helpers to fabricate random ARC-style values
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


def random_unary_dsl() -> Callable:
    """Return a random DSL primitive that takes exactly one positional arg."""
    unary = [
        f for _, f in inspect.getmembers(dsl, inspect.isfunction)
        if len(inspect.signature(f).parameters) == 1
           and not f.__name__.startswith('_')
    ]
    return random.choice(unary) if unary else (lambda x: x)

# ---------------------------------------------------------------------------
# 2.   choose sample generators for each parameter
# ---------------------------------------------------------------------------
def variant_generators(name: str, anno: Any) -> List[Callable[[], Any]]:
    """
    Return *all* distinct 0-arg callables that can create admissible values
    for `anno`.  (Heuristics only – good enough for prompts.)
    """
    # explicit ARC aliases ----------------------------------------------------
    if anno in {T.Patch, 'Patch'}:
        return [rand_object, rand_indices]

    if anno in {T.Object, 'Object'}:
        return [rand_object]

    if anno in {T.Indices, 'Indices'}:
        return [rand_indices]

    if anno in {T.Grid, 'Grid'} or 'grid' in name.lower():
        return [
            lambda: rand_grid(max_side=2),   # very small
            rand_grid                       # arbitrary
        ]

    if anno in {bool, 'bool', T.Boolean}:
        return [lambda: True, lambda: False]

    # Callable? – supply a simple lambda or a real DSL primitive -------------
    if anno is Callable or get_origin(anno) is Callable or 'Callable' in str(anno):
        return [lambda: (lambda x: x), random_unary_dsl]

    # ints --------------------------------------------------------------------
    if anno is int or 'int' in str(anno):
        return [lambda: 0, lambda: random.randint(1, 4)]

    # name‐based fall-backs ---------------------------------------------------
    lname = name.lower()
    if any(k in lname for k in ['obj', 'patch', 'piece']):
        return [rand_object]
    if 'mask' in lname or 'matrix' in lname:
        return [lambda: tuple(tuple(int(bool(rand_color())) for _ in row)
                              for row in rand_grid())]
    if any(k in lname for k in ['coord', 'delta', 'offset']):
        return [lambda: (random.randint(-2, 2), random.randint(-2, 2))]

    # absolute last resort ----------------------------------------------------
    return [rand_color]

# ---------------------------------------------------------------------------
# 3.   build one prompt block for *one* DSL primitive
# ---------------------------------------------------------------------------
def make_block(func: Callable,
               min_shots: int = 2,
               max_shots: int = 5) -> dict:
    sig = inspect.signature(func)

    # collect generators only for *required* parameters ----------------------
    param_variants = {
        name: variant_generators(name, param.annotation)
        for name, param in sig.parameters.items()
        if param.default is inspect._empty
           and param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    }

    needed = max((len(v) for v in param_variants.values()), default=1)
    n_shots = max(needed, random.randint(min_shots, max_shots))

    shots = []
    for i in range(n_shots):
        kwargs = {
            name: (variants[i] if i < len(variants) else random.choice(variants))()
            for name, variants in param_variants.items()
        }
        try:
            out = func(**kwargs)
        except Exception:
            # retry once with fresh random values
            kwargs = {n: random.choice(v)() for n, v in param_variants.items()}
            out = func(**kwargs)
        shots.append({"inputs": kwargs, "output": out})

    # pretty prompt -----------------------------------------------------------
    demo_lines = []
    for k, shot in enumerate(shots, 1):
        demo_lines.append(f"# Example {k}")
        for arg_name, val in shot["inputs"].items():
            demo_lines.append(f"{arg_name.upper()} = {pformat(val)}")
        demo_lines.append(f"# Desired → {pformat(shot['output'])}")
        demo_lines.append("")

    input_prompt = (
        f"You are given {len(shots)} input/output examples.\n"
        f"Write solve(I) so that it applies `{func.__name__}` to its input.\n\n"
        + "\n".join(demo_lines)
        + "### Now write solve(I) below"
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
# 4.   iterate over every *unary* primitive, dump JSON
# ---------------------------------------------------------------------------
def main() -> Sequence[dict]:
    blocks = []
    for name, func in inspect.getmembers(dsl, inspect.isfunction):
        if name.startswith('_'):
            continue
        if len(inspect.signature(func).parameters) != 1:
            continue           # keep the strict “one-liner” constraint
        try:
            blocks.append(make_block(func))
        except Exception as err:
            print(f"[warn] skipped {name}: {err}", flush=True)
    return blocks


if __name__ == "__main__":
    with open("train_set.json", "w") as fp:
        json.dump(main(), fp, indent=2, default=lambda o: pformat(o))
    print("✓ Wrote train_set.json")