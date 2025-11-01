"""Block generation logic for DSL function training data (refactored)."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Tuple, get_args, get_origin

import src.arc_dslearn.arc_dsl.arc_types as T
import src.arc_dslearn.arc_dsl.dsl as dsl
from src.arc_dslearn.data_gene_grid.data_processing import compact_format, to_jsonable
from src.arc_dslearn.data_gene_grid.generators import rand_grid

# ---------- Introspection & caching ----------


def _public_dsl_functions() -> list[tuple[str, Callable[..., Any]]]:
    """Cache public DSL callables once."""
    return [(n, f) for n, f in inspect.getmembers(dsl, inspect.isfunction) if not n.startswith("_")]


_DSL_FUNCS: list[tuple[str, Callable[..., Any]]] = _public_dsl_functions()


def dsl_functions_summary() -> str:
    """Return aone-line bullet list of DSL functions."""
    return "\n".join(f"- {name}" for name, _ in _DSL_FUNCS)


DSL_FUNCTIONS_BLOCK = dsl_functions_summary()

# ---------- Type helpers ----------

_ARC_EQUIV: dict[Any, set[str]] = {
    T.Grid: {"Grid"},
    T.Object: {"Object"},
    T.Objects: {"Objects"},
    T.Indices: {"Indices"},
    T.IndicesSet: {"IndicesSet"},
    T.Patch: {"Patch"},
    T.Element: {"Element"},
    T.Piece: {"Piece"},
    T.Integer: {"Integer", "int"},
    T.Boolean: {"Boolean", "bool"},
    T.IntegerTuple: {"IntegerTuple"},
    T.IntegerSet: {"IntegerSet"},
}


def _name_set(anno: Any) -> set[str]:
    return {getattr(anno, "__name__", str(anno)), str(anno)}


def _is_grid(anno: Any) -> bool:
    """Return True if annotation denotes a Grid or Grid-like tuple-of-tuples[int]."""
    if anno in (T.Grid, "Grid"):
        return True
    origin = get_origin(anno)
    if origin in {tuple, Tuple}:
        args = get_args(anno)
        if len(args) == 1 and get_origin(args[0]) in {tuple, Tuple}:
            inner = get_args(args[0])
            return len(inner) == 1 and inner[0] in {int, T.Integer, "int", "Integer"}
    return False


def _same_type(a: Any, b: Any) -> bool:
    if a == b:
        return True
    if _is_grid(a) and _is_grid(b):
        return True
    # alias equivalence by names
    names_a, names_b = _name_set(a), _name_set(b)
    for canon, aliases in _ARC_EQUIV.items():
        if names_a & aliases and (b is canon or names_b & aliases):
            return True
    return False


# ---------- Function filtering & signatures ----------


def should_skip_function(func: Callable[..., Any]) -> bool:
    """Skip functions that use or return Callables (non-unary composition primitives)."""
    sig = inspect.signature(func)
    ret = sig.return_annotation
    if ret is not inspect._empty and (ret is Callable or get_origin(ret) is Callable):
        return True
    for p in sig.parameters.values():
        if p.annotation is not inspect._empty and (
            p.annotation is Callable or get_origin(p.annotation) is Callable
        ):
            return True
    return False


def _unary_funcs_with_types() -> list[tuple[str, Callable[..., Any], Any, Any]]:
    """(name, func, in_type, out_type) for unary functions with annotations."""
    out: list[tuple[str, Callable[..., Any], Any, Any]] = []
    for name, func in _DSL_FUNCS:
        if should_skip_function(func):
            continue
        sig = inspect.signature(func)
        params = [
            p for p in sig.parameters.values() if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        if len(params) != 1:
            continue
        p = params[0]
        if p.annotation is inspect._empty or sig.return_annotation is inspect._empty:
            continue
        out.append((name, func, p.annotation, sig.return_annotation))
    return out


_UNARY_FUNCS = _unary_funcs_with_types()

# ---------- Path search (Grid → Grid) ----------


def _find_paths_grid_to_grid(max_len: int = 5, seed: int | None = None):
    """Yield lists of (name, func, in_type, out_type) from Grid back to Grid, 2..max_len steps."""
    rng = __import__("random").Random(seed)
    funcs = _UNARY_FUNCS
    starters = [(n, f, tin, tout) for (n, f, tin, tout) in funcs if _is_grid(tin)]
    paths: list[list[tuple[str, Callable[..., Any], Any, Any]]] = []

    def dfs(path: list[tuple[str, Callable[..., Any], Any, Any]]):
        # Accept 2..max_len where last out_type is Grid
        if 2 <= len(path) <= max_len and _is_grid(path[-1][3]):
            paths.append(path[:])
        if len(path) >= max_len:
            return
        last_out = path[-1][3]
        used_funcs = {id(p[1]) for p in path}  # prevent trivial cycles by same func
        for n, f, tin, tout in funcs:
            if id(f) in used_funcs:
                continue
            if _same_type(tin, last_out):
                path.append((n, f, tin, tout))
                dfs(path)
                path.pop()

    for s in starters:
        dfs([s])

    rng.shuffle(paths)
    return paths


# ---------- Block builder ----------


def make_multiline_block(
    max_lines: int = 5,
    seed: int | None = None,
    n_shots: int = 3,
) -> dict[str, Any]:
    """Compose a chain of 2..max_lines unary DSL functions with Grid input & output. Return a training block with few-shot I→O pairs and a code template."""
    rng = __import__("random").Random(seed)

    # 1) choose path
    paths = _find_paths_grid_to_grid(max_len=max_lines, seed=seed)
    if not paths:
        # fallback to single-step Grid→Grid if available
        single = [
            (n, f, tin, tout)
            for (n, f, tin, tout) in _UNARY_FUNCS
            if _is_grid(tin) and _is_grid(tout)
        ]
        if not single:
            raise RuntimeError("No unary Grid→Grid paths (≤5) found among DSL functions.")
        path = [single[0]]
    else:
        path = rng.choice(paths)

    # 2) render assistant code body
    body_lines = []
    prev = "I"
    for i, (fname, _f, _tin, _tout) in enumerate(path, start=1):
        xi = f"x{i}"
        body_lines.append(f"    {xi} = {fname}({prev})")
        prev = xi
    body_lines.append(f"    O = {prev}")
    body = "\n".join(body_lines)

    assistant_prompt = f"""```python
def solve(I):
{body}
    return O
```"""

    # 3) system + user prompts
    system_prompt = (
        "Given the functions of the DSL below, implement `solve` that **only** composes these DSL primitives.\n"
        "Constants, if any, must come from the DSL's public constants list.\n\n"
        "DSL reference:\n" + DSL_FUNCTIONS_BLOCK
    )
    user_template = (
        "You will receive examples where I (input) is a Grid and the desired output O is a Grid.\n"
        "Write python **code only** following *exactly* this template:\n"
        "```python\n"
        "def solve(I):\n"
        "    # line_1\n"
        "    x1 = dsl_function(I)\n"
        "    # line_2\n"
        "    x2 = dsl_function(x1)\n"
        "    # ... up to 5 lines total\n"
        "    O  = xN\n"
        "    return O\n"
        "```\n"
        "Do not import anything. Do not use non-DSL names. Return the final value in O."
    )

    # 4) few-shot examples via the chosen path
    shots = []
    for i in range(n_shots):
        # make rand_grid deterministic relative to overall seed without modifying global state
        # (if rand_grid supports seed, prefer that; otherwise rely on global RNG)
        if seed is not None:
            __import__("random").seed(seed + i * 1000)
        input_value = rand_grid()
        cur = input_value
        for _fname, f, _tin, _tout in path:
            cur = f(cur)
        output_value = cur
        shots.append({
            "inputs": {"I": to_jsonable(input_value)},
            "output": to_jsonable(output_value),
        })

    examples = "\n".join(
        f"# Example {j + 1}\nI = {compact_format(s['inputs']['I'])}\n# Desired → {compact_format(s['output']['O'])}\n"
        for j, s in enumerate(shots)
    )

    return {
        "name": "⟂".join([p[0] for p in path]),
        "system_prompt": system_prompt,
        "user_prompt": examples + "\n### Task\n" + user_template,
        "assistant_prompt": assistant_prompt,
        "shots": shots,
        "lines": len(path),
    }
