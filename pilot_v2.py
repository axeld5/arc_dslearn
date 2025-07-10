from __future__ import annotations

import inspect
import json
import random
import collections.abc as cabc
from pprint import pformat
from typing import (
    Any, Callable, FrozenSet, Tuple, Sequence,
    get_origin, get_args, Container,
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
    # Ensure at least one element to avoid empty collections
    if not triples:
        # Add at least one random element
        r, c = random.randint(0, len(g)-1), random.randint(0, len(g[0])-1)
        triples.add((g[r][c], (r, c)))
    return frozenset(triples)


def rand_indices(max_side: int = 5) -> T.Indices:
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
    return (random.randint(-3, 3), random.randint(-3, 3))


def rand_int_frozenset() -> FrozenSet[int]:
    size = random.randint(1, 5)  # Ensure at least size 1
    result = set()
    while len(result) < size:
        result.add(random.randint(0, 9))
        # Prevent infinite loop if trying to get more than 10 unique values
        if len(result) >= 10:
            break
    return frozenset(result)


def rand_container(depth: int = 1) -> Any:
    """
    A very general container generator:
        depth=1 → tuple of ints
        depth=2 → tuple of tuple of ints
    """
    if depth == 1:
        size = random.randint(1, 5)  # Ensure at least size 1
        return tuple(random.randint(0, 9) for _ in range(size))
    size = random.randint(1, 4)  # Ensure at least size 1
    return tuple(
        rand_container(depth - 1)
        for _ in range(size)
    )

# ---------------------------------------------------------------------------
# 1. value-generator dispatch table
# ---------------------------------------------------------------------------
def variant_generators(name: str, anno: Any, func_name: str = "") -> list[Callable[[], Any]]:
    """
    Return a *non-empty* list of zero-arg callables that generate admissible
    sample values for the given annotation.
    """
    # --- ARC aliases ---------------------------------------------------------
    if anno in {T.Patch, 'Patch'}:
        # Functions that need non-empty collections
        if func_name in {'bordering', 'center', 'inbox', 'outbox', 'corners'}:
            return [lambda: rand_object(max_side=3), lambda: rand_indices(max_side=3)]
        # Functions that need count method - use tuples instead of frozensets
        elif func_name in {'mostcommon', 'leastcommon'}:
            return [lambda: tuple(rand_color() for _ in range(random.randint(3, 8))),
                    lambda: tuple(rand_color() for _ in range(random.randint(4, 10)))]
        return [rand_object, rand_indices]
    if anno in {T.Object, 'Object'}:
        # Functions that need non-empty objects
        if func_name in {'bordering', 'center', 'inbox', 'outbox', 'corners', 'mostcommon', 'leastcommon'}:
            return [lambda: rand_object(max_side=3)]
        return [rand_object]
    if anno in {T.Objects, 'Objects'}:
        # Generate a frozenset of objects
        return [lambda: frozenset([rand_object(max_side=3) for _ in range(random.randint(1, 3))]),
                lambda: frozenset([rand_object(max_side=2) for _ in range(random.randint(2, 4))])]
    if anno in {T.Indices, 'Indices'}:
        # Functions that need non-empty indices
        if func_name in {'bordering', 'center', 'inbox', 'outbox', 'corners', 'mostcommon', 'leastcommon'}:
            return [lambda: rand_indices(max_side=3)]
        return [rand_indices]
    if anno in {T.IndicesSet, 'IndicesSet'}:
        # Generate a frozenset of indices
        return [lambda: frozenset([rand_indices(max_side=3) for _ in range(random.randint(1, 3))]),
                lambda: frozenset([rand_indices(max_side=2) for _ in range(random.randint(2, 4))])]
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
        # Special case for functions that perform division or modulo operations
        # to avoid division by zero errors
        division_funcs = {'divide', 'downscale', 'hsplit', 'vsplit'}
        if func_name in division_funcs:
            return [lambda: random.randint(1, 9), lambda: random.randint(1, 9)]
        # Special case for interval function - step parameter must never be 0
        elif func_name == 'interval' and ('step' in name.lower() or name.lower() in ['n', 'delta', 'increment']):
            return [lambda: random.randint(1, 9), lambda: random.randint(1, 9)]
        else:
            return [lambda: 0, lambda: random.randint(1, 4)]
    if anno in {T.IntegerTuple, 'IntegerTuple'} or any(k in name.lower() for k in ['coord', 'delta', 'offset', 'direction']):
        # Always generate exactly 2-element tuples for IntegerTuple
        return [lambda: (random.randint(-3, 3), random.randint(-3, 3)),
                lambda: (random.randint(-2, 2), random.randint(-2, 2))]
    if anno in {T.ContainerContainer, 'ContainerContainer'}:
        # Generate container of containers
        return [lambda: (rand_container(), rand_container()),
                lambda: (rand_object(), rand_indices(), rand_int_frozenset()),
                lambda: frozenset([rand_object(), rand_indices()])]
    if anno is Callable or get_origin(anno) is Callable or 'Callable' in str(anno):
        # This should not be reached since we skip functions with callable parameters
        pass

    # --- typing generics -----------------------------------------------------
    origin = get_origin(anno)
    args = get_args(anno)

    # --- handle Union ---------------------------------------------------------
    if origin is Union:                                    # ①
        gens = []
        for a in args:
            gens.extend(variant_generators(name, a, func_name))       # recurse
        return gens or [rand_color]

    # --- plain Container / Any ------------------------------------------------
    if anno in {cabc.Container, Container} or str(anno) == 'Container':
        # Functions that need count method - use tuples instead of frozensets
        if func_name in {'mostcommon', 'leastcommon'}:
            return [lambda: tuple(rand_color() for _ in range(random.randint(3, 8))),
                    lambda: tuple(rand_color() for _ in range(random.randint(4, 10))),
                    lambda: tuple(rand_color() for _ in range(random.randint(2, 6)))]
        # Functions that need len() on items - use container of containers
        elif func_name == 'sizefilter':
            return [lambda: (rand_object(), rand_indices(), rand_object()),
                    lambda: (rand_container(2), rand_container(2), rand_grid()),
                    lambda: frozenset([rand_object(), rand_indices(), rand_grid()])]
        return [rand_container, rand_object, rand_indices, rand_int_frozenset]

    if anno is Any:                                       # ③
        return [rand_color]

    # --- generics -------------------------------------------------------------
    if origin in {tuple, Tuple}:                           # ④
        return [lambda: (random.randint(-3, 3), random.randint(-3, 3)),
                lambda: tuple(rand_color() for _ in range(3))]

    if origin in {frozenset, FrozenSet}:                   # ⑤
        inner = args[0] if args else int
        if inner in {int, 'int', T.Integer}:
            return [rand_int_frozenset]

    if origin in {cabc.Container, Container}:        # ⑥
        # Container[...]   → shallow tuple of ints
        return [rand_container]
    if origin is FrozenSet:
        # FrozenSet[int] or similar
        inner = args[0] if args else int
        if inner in {int, 'int', T.Integer}:
            return [rand_int_frozenset]
    if origin is Tuple:
        # Tuple[int, int]  or Tuple[…]
        return [lambda: (random.randint(-3, 3), random.randint(-3, 3)),
                lambda: tuple(rand_color() for _ in range(3))]

    # --- name-based fall-backs ----------------------------------------------
    lname = name.lower()
    if any(k in lname for k in ['obj', 'patch', 'piece']):
        return [rand_object]
    if 'mask' in lname or 'matrix' in lname:
        return [lambda: tuple(tuple(int(bool(rand_color())) for _ in row)
                              for row in rand_grid())]
    print(f"name: {name}")
    return [rand_color]

# ---------------------------------------------------------------------------
# 2. build one prompt block
# ---------------------------------------------------------------------------
def dsl_functions_summary() -> str:
    """Return a readable bullet‑list of DSL functions (name, signature, return,
    first docstring line).  Runs **once** at import time."""
    lines: list[str] = []
    for name, func in inspect.getmembers(dsl, inspect.isfunction):
        if name.startswith("_"):
            continue  # skip private helpers
        sig = inspect.signature(func)
        ret = sig.return_annotation if sig.return_annotation is not inspect._empty else "Any"
        doc = (func.__doc__ or "").strip().splitlines()[0]
        lines.append(f"- {name}{sig} -> {ret}: {doc}")
    return "\n".join(lines)

DSL_FUNCTIONS_BLOCK = dsl_functions_summary()

def make_block(func: Callable,
               min_shots: int = 2,
               max_shots: int = 5) -> dict:
    sig = inspect.signature(func)

    param_variants = {
        name: variant_generators(name, param.annotation, func.__name__)
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
    system_prompt = (
        "Given the functions of the DSL below, you must implement a new "
        "function called `solve` that **only** composes these DSL primitives.\n"
        "Constants, if any, must come from the DSL’s public constants list.\n\n"
        "DSL reference:\n" + DSL_FUNCTIONS_BLOCK
    )

    # few‑shot part the model sees *before* the task
    examples_txt: list[str] = []
    for k, s in enumerate(shots, 1):
        examples_txt.append(f"# Example {k}")
        for arg_name, val in s["inputs"].items():
            examples_txt.append(f"{arg_name.upper()} = {pformat(val)}")
        examples_txt.append(f"# Desired → {pformat(s['output'])}\n")

    user_prompt = (
        "\n".join(examples_txt) +
        "### Task\n"
        "Write python **code only** following *exactly* this template:```python\n"
        "def solve(I):\n"
        "    # line_1\n"
        "    x1 = dsl_function(... )\n"
        "    # line_2\n"
        "    x2 = dsl_function(... )\n"
        "    # ...\n"
        "    O  = xN\n"
        "    return O\n```\n"
        "Do not import anything.  Do not use non‑DSL names.  Return the final "
        "value in *O*."
    )
    
    assistant_prompt = (
        "```python\n"
        f"def solve(I):\n    O = {func.__name__}(I)\n    return O\n```")

    return {
        "name": func.__name__,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "assistant_prompt": assistant_prompt,
        "shots": shots,
    }

# ---------------------------------------------------------------------------
# 3. iterate over DSL, keep *one-arg* & *non-Callable return*
# ---------------------------------------------------------------------------
def main() -> Sequence[dict]:
    blocks = []
    for _ in range(10):
        for name, func in inspect.getmembers(dsl, inspect.isfunction):
            if name.startswith('_'):
                continue

            sig = inspect.signature(func)

            # skip functions that *return* a Callable
            ret_anno = sig.return_annotation
            if ret_anno is not inspect._empty and (
                ret_anno is Callable or get_origin(ret_anno) is Callable
            ):
                continue

            # skip functions that *take* a Callable as parameter
            has_callable_param = any(
                param.annotation is Callable or get_origin(param.annotation) is Callable
                for param in sig.parameters.values()
                if param.annotation is not inspect._empty
            )
            if has_callable_param:
                print(f"[info] skipped {name}: has callable parameter")
                continue

            try:
                blocks.append(make_block(func))
            except Exception as err:
                # silenced: comment out the next line for verbose debugging
                print(f"[warn] skipped {name}: {err}")
                pass
    return blocks


if __name__ == "__main__":
    with open("train_set.json", "w") as fp:
        json.dump(main(), fp, indent=2, default=lambda o: pformat(o))
    print("✓ Wrote train_set.json")