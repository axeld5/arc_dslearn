"""Reward function for the RL script."""

import ast
import builtins
import contextlib
import inspect
import json
import re
import threading
import types
from typing import Any, Callable, Dict, Generator, List

import src.arc_dslearn.arc_dsl.dsl as dsl
from src.arc_dslearn.utils import from_jsonable


def equality_key(x: Any) -> Any:
    """Return a deterministic representation that ignores order for containers that are *logically* unordered."""
    if isinstance(x, (int, float, bool, str)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return tuple(equality_key(v) for v in x)
    if isinstance(x, dict):
        return tuple((k, equality_key(v)) for k, v in x.items())
    if isinstance(x, (set, frozenset)):
        return frozenset(equality_key(v) for v in x)
    return x


def equivalent(a: Any, b: Any, shot_inputs: Dict[str, Any] | None = None) -> bool:
    """Return True if values should be considered equal using ARC-DSL."""
    if equality_key(a) == equality_key(b):
        return True

    # special-case: colour ties ------------------------------------------
    if (
        isinstance(a, int)
        and isinstance(b, int)
        and isinstance(shot_inputs, dict)
        and "obj" in shot_inputs
    ):
        obj = from_jsonable(shot_inputs["obj"])
        if isinstance(obj, frozenset):  # sanity guard
            colors = {pix for pix, _ in obj}
            if a in colors and b in colors:
                freq = {c: dsl.colorcount(obj, c) for c in colors}
                maxcnt = max(freq.values())
                max_colors = {c for c, f in freq.items() if f == maxcnt}  # NEW
                return a in max_colors and b in max_colors  # NEW
    return False


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    # Match ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no markdown, return as-is (might be plain Python)
    return text.strip()


def create_smart_wrappers(mod: types.ModuleType) -> None:
    """Attach smart wrappers for every public DSL primitive.

    A call like   foo(I)   is translated to   foo(I['a'], I['b'])   (etc.),
    and each extracted value is converted with `from_jsonable` before the
    real DSL function is invoked.
    """

    def make_wrapper(dsl_func: Callable[..., Any]) -> Callable[..., Any]:
        try:
            sig = inspect.signature(dsl_func)
        except (TypeError, ValueError):  # C built-ins (bool, int, …)
            return dsl_func  # → leave them untouched

        param_names = [
            p.name
            for p in sig.parameters.values()
            if p.default is p.empty and p.kind is p.POSITIONAL_OR_KEYWORD
        ]
        arity = len(param_names)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                bundle = args[0]

                # ––– 1. perfect match on parameter names –––––––––––––––––
                if set(param_names) <= bundle.keys():
                    vals = [from_jsonable(bundle[n]) for n in param_names]

                # ––– 2. fallback: same cardinality –––––––––––––––––––––––
                elif len(bundle) == arity:
                    vals = [from_jsonable(v) for v in bundle.values()]

                # ––– 3. single-parameter convenience –––––––––––––––––––––
                elif arity == 1:
                    # common synonyms
                    for k in ("patch", "obj", "object", "piece", "grid", "container", "a"):
                        if k in bundle:
                            vals = [from_jsonable(bundle[k])]
                            break
                    else:
                        # first (or only) value
                        vals = [from_jsonable(next(iter(bundle.values())))]

                # ––– 4. we cannot guess ––––––––––––––––––––––––––––––––––
                else:
                    return dsl_func(*args, **kwargs)

                return dsl_func(*vals)
            return dsl_func(*args, **kwargs)

        return wrapper

    for name, func in dsl.__dict__.items():
        if callable(func) and not name.startswith("_"):
            setattr(mod, name, make_wrapper(func))


SOLVE_RE = re.compile(r"\bdef\s+solve\s*\(\s*I\s*\)\s*:")
IMPORT_RE = re.compile(r"^\s*import\s+", re.M)


class TimeoutError(Exception):
    """Custom timeout error for cross-platform compatibility."""

    pass


@contextlib.contextmanager
def time_limit(seconds: int) -> Generator[None, None, None]:
    """Cross-platform time limit for code execution using threading."""
    timeout_occurred = threading.Event()

    def timeout_handler() -> None:
        timeout_occurred.set()

    # Start the timer
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()

    try:
        yield
        # Check if timeout occurred during execution
        if timeout_occurred.is_set():
            raise TimeoutError("Code execution timed out")
    finally:
        # Always cancel the timer to prevent it from firing later
        timer.cancel()


def safe_exec(code: str, input_data: Any = None) -> types.ModuleType:
    """Execute the code with the input data."""
    mod = types.ModuleType("submission")
    mod.dsl = dsl  # type: ignore
    mod.__builtins__ = {n: getattr(builtins, n) for n in ("range", "len", "enumerate", "zip")}  # type: ignore

    # expose raw DSL names first …
    for name, func in dsl.__dict__.items():
        if callable(func) and not name.startswith("_"):
            setattr(mod, name, func)

    # … then replace them with the smart versions
    create_smart_wrappers(mod)

    # make the *entire* input bundle available both as a global 'I'
    # and as the sole argument that we'll pass to solve()
    processed = from_jsonable(input_data)
    mod.I = processed  # type: ignore

    with time_limit(2):
        exec(code, mod.__dict__)
    return mod


def reward_function(
    completions: List[str], shots: List[List[Dict[str, Any]]], **kwargs: Any
) -> List[float]:
    """Reward = 0.1 (format) + 0.1 (DSL only) + 0.8 (all shots pass)."""
    rewards: List[float] = []
    for code_raw, shot_list in zip(completions, shots, strict=False):
        # Extract Python code from markdown
        code = extract_python_code(code_raw)

        shot_list = from_jsonable(shot_list)
        r = 0.0

        # (1) Solve function present
        r += 0.1 if SOLVE_RE.search(code) else -1.0

        # (2) No bad imports, only DSL names
        try:
            tree = ast.parse(code)
            bad_imports = bool(IMPORT_RE.search(code))
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            unknown = (
                names - {"I", "O"} - {f"x{i}" for i in range(1, 100)} - set(dsl.__dict__.keys())
            )
            if not bad_imports and not unknown:
                r += 0.1
        except SyntaxError:
            pass

        # (3) Functional correctness on all provided shots
        all_passed = True
        for shot in shot_list:
            try:
                # Parse the input - it comes as a JSON string
                input_data = shot["inputs"]
                if isinstance(input_data, str):
                    input_data = json.loads(input_data)

                # Execute with input data context
                mod = safe_exec(code, input_data)
                if mod and callable(getattr(mod, "solve", None)):
                    result = mod.solve(input_data)

                    # Parse expected output
                    expected = shot["output"]
                    if isinstance(expected, str):
                        expected = json.loads(expected)

                    expected = from_jsonable(expected)

                    if not equivalent(result, expected, input_data):
                        all_passed = False
                        break
                else:
                    all_passed = False
                    break
            except Exception:
                all_passed = False
                break

        if all_passed:
            r += 0.8

        rewards.append(r)
    return rewards
