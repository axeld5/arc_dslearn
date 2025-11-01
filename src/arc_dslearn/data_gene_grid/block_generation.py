"""Block generation logic for DSL function training data (refactored, fast sampler)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Tuple, Union, get_args, get_origin

import src.arc_dslearn.arc_dsl.arc_types as T
import src.arc_dslearn.arc_dsl.constants as C
import src.arc_dslearn.arc_dsl.dsl as dsl
from src.arc_dslearn.data_gene_grid.data_processing import compact_format, to_jsonable
from src.arc_dslearn.data_gene_grid.generators import rand_grid

# ---------- Introspection & caching ----------

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


def _public_dsl_functions() -> list[tuple[str, Callable[..., Any]]]:
    """Cache public DSL callables once."""
    return [(n, f) for n, f in inspect.getmembers(dsl, inspect.isfunction) if not n.startswith("_")]


_DSL_FUNCS: list[tuple[str, Callable[..., Any]]] = _public_dsl_functions()


def dsl_functions_summary() -> str:
    """Return a one-line bullet list of DSL functions."""
    return "\n".join(f"- {name}" for name, _ in _DSL_FUNCS)


DSL_FUNCTIONS_BLOCK = dsl_functions_summary()

# ---------- Type helpers (alias/union aware) ----------


def _is_grid_structural(anno: Any) -> bool:
    """Tuple[Tuple[int]] structural recognition (besides T.Grid/'Grid')."""
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


@lru_cache(maxsize=None)
def _expand_atoms_cached(anno: Any) -> frozenset[str]:
    """Wrap around _expand_atoms to avoid repeating work."""
    return _expand_atoms(anno)


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


# ---------- Constant matching ----------


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


@dataclass(frozen=True)
class _FnMeta:
    name: str
    func: Callable[..., Any]
    param_annos: list[Any]  # annotations per positional-or-keyword param
    flow_positions: tuple[
        int, ...
    ]  # indices whose annotation accepts Grid (possible "wire-through")
    produces_grid: bool  # return type can be Grid


def _collect_fn_meta() -> list[_FnMeta]:
    metas: list[_FnMeta] = []
    for name, func in _DSL_FUNCS:
        if should_skip_function(func):
            continue
        sig = inspect.signature(func)
        params = [
            p for p in sig.parameters.values() if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        if not params:
            continue
        # All params must have annotations for us to reason about them
        if any(p.annotation is inspect._empty for p in params):
            continue
        ret = sig.return_annotation
        if ret is inspect._empty:
            continue

        annos = [p.annotation for p in params]
        flow_idxs = tuple(i for i, a in enumerate(annos) if _accepts_grid(a))
        produces = _produces_grid(ret)

        # Discard functions that have no place to wire the flowing Grid
        if not flow_idxs:
            continue

        metas.append(
            _FnMeta(
                name=name,
                func=func,
                param_annos=annos,
                flow_positions=flow_idxs,
                produces_grid=produces,
            )
        )
    return metas


_FN_META: list[_FnMeta] = _collect_fn_meta()
_ALL_IDX: list[int] = list(range(len(_FN_META)))

# ---------- Random path sampling (fast) ----------


def _biased_length(rng, min_len: int, max_len: int) -> int:
    """Favor longer paths softly."""
    lengths = list(range(min_len, max_len + 1))
    weights = list(range(1, len(lengths) + 1))
    return rng.choices(lengths, weights=weights, k=1)[0]


@dataclass(frozen=True)
class _StepPlan:
    idx: int  # index into _FN_META
    flow_pos: int  # which param receives the previous Grid
    const_names: tuple[str, ...]  # length == num params; "" marks the flow position
    const_vals: tuple[Any, ...]  # same length; None marks the flow position


def _try_make_step_plan(meta: _FnMeta, rng) -> _StepPlan | None:
    """Choose a flow position and fill other params with constants. Fail fast if impossible."""
    # Choose a random flow position among the acceptable ones
    flow_pos = rng.choice(meta.flow_positions)
    const_names: list[str] = []
    const_vals: list[Any] = []

    for i, anno in enumerate(meta.param_annos):
        if i == flow_pos:
            const_names.append("")  # placeholder, will be replaced with prev var
            const_vals.append(None)
            continue
        picked = _pick_constant_for_annotation(anno, rng)
        if picked is None:
            # Cannot satisfy parameter with our constants; give up on this meta for now
            return None
        cname, cval = picked
        const_names.append(cname)
        const_vals.append(cval)

    return _StepPlan(
        idx=_FN_META.index(meta),
        flow_pos=flow_pos,
        const_names=tuple(const_names),
        const_vals=tuple(const_vals),
    )


def _sample_path_plans(
    rng,
    max_len: int,
    min_len: int = 3,
    path_budget: int = 512,
    middle_pool_size: int = 64,
) -> list[list[_StepPlan]]:
    """Sample path plans (with constants chosen) whose LAST function can return a Grid."""
    if not _ALL_IDX:
        return []

    paths: list[list[_StepPlan]] = []
    seen: set[tuple[int, ...]] = set()

    for _ in range(path_budget):
        L = _biased_length(rng, min_len=min_len, max_len=max_len)

        # pick last from metas that can produce a Grid
        enders = [i for i in _ALL_IDX if _FN_META[i].produces_grid]
        if not enders:
            break
        last_idx = rng.choice(enders)

        # pick first + middles from anywhere; avoid trivial repeats
        pool = [i for i in _ALL_IDX if i != last_idx]
        if not pool:
            continue
        if len(pool) > middle_pool_size:
            rng.shuffle(pool)
            pool = pool[:middle_pool_size]
        rng.shuffle(pool)

        # choose L-1 items for first..middle, then append last
        pre = pool[: max(L - 1, 1)]
        idxs = pre + [last_idx]
        key = tuple(id(_FN_META[i].func) for i in idxs)
        if key in seen:
            continue
        seen.add(key)

        # Build a concrete plan (flow position + constants for each step)
        plan: list[_StepPlan] = []
        ok = True
        for i in idxs:
            meta = _FN_META[i]
            step = _try_make_step_plan(meta, rng)
            if step is None:
                ok = False
                break
            plan.append(step)
        if not ok:
            continue

        # avoid all-same-function-name paths (dull)
        names = {_FN_META[s.idx].name for s in plan}
        if len(names) == 1:
            continue

        paths.append(plan)

    return paths


# ---------- Block builder ----------


def make_multiline_block(
    max_lines: int = 5,
    seed: int | None = None,
    n_shots: int = 3,
    max_path_retries: int = 10,
    max_shot_retries: int = 50,
) -> dict[str, Any]:
    """Compose a chain of 2..max_lines DSL functions (multi-arg allowed) with Grid input and Grid output. Only the last function must be able to produce a Grid. Non-flow params are filled from DSL constants."""
    rng = __import__("random").Random(seed)

    # 1) Sample candidate path plans (with constants chosen per param)
    plans = _sample_path_plans(
        rng,
        max_len=max_lines,
        min_len=3,
        path_budget=1024,
        middle_pool_size=64,
    )
    if not plans:
        # fallback: single-step that produces Grid and can be satisfied with constants
        singles = []
        for meta in _FN_META:
            if not meta.produces_grid:
                continue
            step = _try_make_step_plan(meta, rng)
            if step:
                singles.append([step])
        if not singles:
            raise RuntimeError(
                "No satisfiable DSL plans found (need constants for non-grid params)."
            )
        plans = singles

    # Try multiple plans until we can generate all shots successfully
    for attempt in range(min(max_path_retries, len(plans))):
        plan = plans[attempt]

        # 2) Render assistant code body
        body_lines = []
        prev = "I"
        for i, step in enumerate(plan, start=1):
            meta = _FN_META[step.idx]
            xi = f"x{i}"
            # Build call string in the param order
            arg_exprs = []
            for j, cname in enumerate(step.const_names):
                if j == step.flow_pos:
                    arg_exprs.append(prev)
                else:
                    arg_exprs.append(cname)
            call = f"{meta.name}({', '.join(arg_exprs)})"
            body_lines.append(f"    {xi} = {call}")
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
            "    x1 = dsl_function(I, CONST1, CONST2)\n"
            "    # line_2\n"
            "    x2 = dsl_function(x1, CONSTa, CONSTb)\n"
            "    # ... up to 5 lines total\n"
            "    O  = xN\n"
            "    return O\n"
            "```\n"
            "Do not import anything. Do not use non-DSL names. Return the final value in O."
        )

        # 4) Few-shot examples via the chosen plan
        shots = []
        shot_attempts = 0
        seed_bump = 0

        while len(shots) < n_shots and shot_attempts < max_shot_retries:
            try:
                if seed is not None:
                    __import__("random").seed((seed or 0) + seed_bump * 1000)

                input_value = rand_grid()
                cur = input_value

                # Execute each step with real values
                for step in plan:
                    meta = _FN_META[step.idx]
                    args: list[Any] = []
                    for j, val in enumerate(step.const_vals):
                        if j == step.flow_pos:
                            args.append(cur)
                        else:
                            args.append(val)
                    cur = meta.func(*args)

                output_value = cur
                shots.append({
                    "inputs": {"I": to_jsonable(input_value)},
                    "output": to_jsonable(output_value),
                })
                seed_bump += 1

            except Exception:
                # Try a different random input
                shot_attempts += 1
                seed_bump += 1
                if shot_attempts < max_shot_retries:
                    continue
                else:
                    # Give up on this plan, try another
                    break

        if len(shots) == n_shots:
            examples = "\n".join(
                f"# Example {j + 1}\nI = {compact_format(s['inputs']['I'])}\n# Desired → {compact_format(s['output'])}\n"
                for j, s in enumerate(shots)
            )
            return {
                "name": "|".join(_FN_META[s.idx].name for s in plan),
                "system_prompt": system_prompt,
                "user_prompt": examples + "\n### Task\n" + user_template,
                "assistant_prompt": assistant_prompt,
                "shots": shots,
                "lines": len(plan),
            }

    raise RuntimeError("Failed to generate block with sampled multi-arg plans.")
