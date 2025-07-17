"""Block generation logic for DSL function training data."""

from __future__ import annotations

import inspect
import random
from pprint import pformat
from typing import Any, Callable, get_origin

import src.arc_dslearn.arc_dsl.dsl as dsl

from .data_processing import to_jsonable
from .generators import variant_generators


def dsl_functions_summary() -> str:
    """Return a readable bullet-list of DSL functions (name, signature, return, first docstring line).  Runs **once** at import time."""
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


def make_block(func: Callable[..., Any], min_shots: int = 2, max_shots: int = 5) -> dict[str, Any]:
    """Generate a block of code for a DSL function."""
    sig = inspect.signature(func)

    param_variants = {
        name: variant_generators(name, param.annotation, func.__name__)
        for name, param in sig.parameters.items()
        if param.default is inspect._empty and param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
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
        shots.append({
            "inputs": to_jsonable(kwargs),
            "output": to_jsonable(out),
        })

    # -------------------- few-shot prompt text ------------------------------
    system_prompt = (
        "Given the functions of the DSL below, you must implement a new "
        "function called `solve` that **only** composes these DSL primitives.\n"
        "Constants, if any, must come from the DSL's public constants list.\n\n"
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
        "\n".join(examples_txt) + "### Task\n"
        "Write python **code only** following *exactly* this template:```python\n"
        "def solve(I):\n"
        "    # line_1\n"
        "    x1 = dsl_function(... )\n"
        "    # line_2\n"
        "    x2 = dsl_function(... )\n"
        "    # ...\n"
        "    O  = xN\n"
        "    return O\n```\n"
        "Do not import anything.  Do not use non-DSL names.  Return the final "
        "value in *O*."
    )

    assistant_prompt = f"```python\ndef solve(I):\n    O = {func.__name__}(I)\n    return O\n```"

    return {
        "name": func.__name__,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "assistant_prompt": assistant_prompt,
        "shots": shots,
    }


def should_skip_function(func: Callable[..., Any]) -> bool:
    """Check if a function should be skipped during block generation."""
    sig = inspect.signature(func)

    # skip functions that *return* a Callable
    ret_anno = sig.return_annotation
    if ret_anno is not inspect._empty and (
        ret_anno is Callable or get_origin(ret_anno) is Callable
    ):
        return True

    # skip functions that *take* a Callable as parameter
    has_callable_param = any(
        param.annotation is Callable or get_origin(param.annotation) is Callable
        for param in sig.parameters.values()
        if param.annotation is not inspect._empty
    )
    return has_callable_param
