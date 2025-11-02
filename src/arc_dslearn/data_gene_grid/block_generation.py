"""Block generation logic for DSL function training data (multi-arg vars + cleanup + fallback)."""

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
class _ParamSpec:
    name: str
    anno: Any
    required: bool  # True if no default


@dataclass(frozen=True)
class _FnMeta:
    name: str
    func: Callable[..., Any]
    params: tuple[_ParamSpec, ...]  # ordered positional-or-kw params
    flow_positions: tuple[int, ...]  # indices whose annotation accepts Grid
    produces_grid: bool
    return_atoms: frozenset[str]  # expanded atoms of return type


def _collect_fn_meta() -> list[_FnMeta]:
    metas: list[_FnMeta] = []
    for name, func in _DSL_FUNCS:
        if should_skip_function(func):
            continue
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
        if not flow_idxs:
            continue

        metas.append(
            _FnMeta(
                name=name,
                func=func,
                params=tuple(params),
                flow_positions=flow_idxs,
                produces_grid=_produces_grid(ret),
                return_atoms=_expand_atoms_cached(ret),
            )
        )
    return metas


_FN_META: list[_FnMeta] = _collect_fn_meta()
_ALL_IDX: list[int] = list(range(len(_FN_META)))

# ---------- Random path sampling (multi-arg) ----------


@dataclass(frozen=True)
class _Var:
    name: str
    atoms: frozenset[str]


@dataclass(frozen=True)
class _ParamSrc:
    # Exactly one of (var_name) or (const_name,const_val) is set, unless 'omit' is True
    var_name: str | None
    const_name: str | None
    const_val: Any | None
    omit: bool = False  # True => don't pass this parameter at all


def _render_call(meta: _FnMeta, srcs: tuple[_ParamSrc, ...]) -> str:
    """Render a python call, using keywords if any parameter is omitted or any earlier param is omitted."""
    # If we omit any param that's not at the end, we must use keywords for *all* that we pass.
    any_omitted = any(s.omit for s in srcs)
    # If using keywords, pass ONLY the non-omitted ones as name=value in declared order.
    if any_omitted:
        parts = []
        for spec, s in zip(meta.params, srcs, strict=False):
            if s.omit:
                continue
            val = s.var_name if s.var_name is not None else s.const_name
            parts.append(f"{spec.name}={val}")
        return f"{meta.name}({', '.join(parts)})"
    else:
        # positional-only prefix
        parts = []
        for s in srcs:
            val = s.var_name if s.var_name is not None else s.const_name
            parts.append(val)
        return f"{meta.name}({', '.join(parts)})"


@dataclass(frozen=True)
class _StepPlan:
    idx: int  # index into _FN_META
    out_var: str  # variable defined by this step (e.g., "x3")
    param_srcs: tuple[_ParamSrc, ...]  # one per passed positional parameter (var or const)


def _biased_length(rng, min_len: int, max_len: int) -> int:
    """Favor longer paths softly."""
    lengths = list(range(min_len, max_len + 1))
    weights = list(range(1, len(lengths) + 1))
    return rng.choices(lengths, weights=weights, k=1)[0]


@dataclass(frozen=True)
class _Binding:
    # What the code will show:
    src: _ParamSrc
    # What we used in the dry-run to validate the step:
    value: Any


def _try_bind_and_run(
    meta: _FnMeta,
    env_vals: dict[str, Any],
    rng,
    max_trials: int = 32,
) -> tuple[tuple[_ParamSrc, ...], Any] | None:
    """Try to build a valid call to `meta.func` using current env (I, x1, ...), constants, and optional param omission. Execute it; on success return (param_srcs_for_rendering, return_value)."""
    params = meta.params

    # Precompute candidate sources per parameter (variables + constants + "omit" for optionals)
    cand_lists: list[list[_Binding]] = []
    var_names = list(env_vals.keys())  # e.g., ["I", "x1", "x2", ...]

    for sp in params:
        cands: list[_Binding] = []

        # variables: any previously produced value
        for vn in var_names:
            cands.append(
                _Binding(
                    src=_ParamSrc(var_name=vn, const_name=None, const_val=None), value=env_vals[vn]
                )
            )

        # constants: only if we have a matching name/value (we don't type-check here;
        # correctness will be validated by actually calling the function)
        for atom_list in _CONST_BY_ATOM.values():
            for cname, cval in atom_list:
                cands.append(
                    _Binding(
                        src=_ParamSrc(var_name=None, const_name=cname, const_val=cval), value=cval
                    )
                )

        # omission for optionals
        if not sp.required:
            cands.append(
                _Binding(
                    src=_ParamSrc(var_name=None, const_name=None, const_val=None, omit=True),
                    value=None,
                )
            )

        cand_lists.append(cands)

    # Try randomized attempts
    for _ in range(max_trials):
        # Heuristic: bias toward variables to keep the graph connected
        picks: list[_Binding] = []
        used_var = False
        for cands in cand_lists:
            # prefer vars 60% if available
            var_cands = [b for b in cands if b.src.var_name is not None]
            b = rng.choice(var_cands) if var_cands and rng.random() < 0.6 else rng.choice(cands)
            if b.src.var_name is not None:
                used_var = True
            picks.append(b)

        # Make sure required params are not omitted
        for b, sp in zip(picks, params, strict=False):
            if sp.required and b.src.omit:
                # resample that slot to non-omit if possible
                non_omit = [
                    c
                    for c in cand_lists[len(picks) - len(params) + params.index(sp)]
                    if not c.src.omit
                ]
                b = rng.choice(non_omit) if non_omit else None
                if b is None:
                    break

        if any(b is None for b in picks):
            continue

        # Ensure at least one variable is used (connectivity)
        if not used_var and any(b.src.var_name is not None for cl in cand_lists for b in cl):
            # force one slot to be a var if possible
            var_slots = [
                i
                for i, cl in enumerate(cand_lists)
                if any(bb.src.var_name is not None for bb in cl)
            ]
            if var_slots:
                i = rng.choice(var_slots)
                var_choices = [bb for bb in cand_lists[i] if bb.src.var_name is not None]
                if var_choices:
                    picks[i] = rng.choice(var_choices)

        # Decide whether to render as keywords (if any omit in the middle)
        any_omit = any(b.src.omit for b in picks)
        # Build args for execution
        if any_omit:
            # keyword call: only non-omitted as name=value
            kwargs = {}
            try:
                for sp, b in zip(params, picks, strict=False):
                    if b.src.omit:
                        continue
                    kwargs[sp.name] = b.value
                ret = meta.func(**kwargs)
                param_srcs = tuple(b.src for b in picks)
                return param_srcs, ret
            except Exception:
                continue
        else:
            # positional call
            args = [b.value for b in picks]
            try:
                ret = meta.func(*args)
                param_srcs = tuple(b.src for b in picks)
                return param_srcs, ret
            except Exception:
                continue

    return None


def _sample_path_plans(
    rng,
    max_len: int,
    min_len: int = 3,
    path_budget: int = 128,
    middle_pool_size: int = 64,  # unused; kept for signature compatibility
) -> list[list[_StepPlan]]:
    """Build plans by *executing steps during sampling* on a single dry-run input. The last chosen function must return a Grid (verified by return typing OR by runtime check)."""
    paths: list[list[_StepPlan]] = []
    if not _FN_META:
        return paths

    for _ in range(path_budget):
        # fresh dry-run environment
        dry_env: dict[str, Any] = {}
        dry_env["I"] = rand_grid()

        # choose target length
        L = _biased_length(rng, min_len=min_len, max_len=max_len)
        plan: list[_StepPlan] = []

        # step 1..L-1: any function that we can run successfully
        ok = True
        for step_no in range(1, L):
            # try a bounded number of metas to avoid O(n*m) blowup
            metas_order = list(range(len(_FN_META)))
            rng.shuffle(metas_order)

            picked: tuple[int, tuple[_ParamSrc, ...], Any] | None = None
            for mi in metas_order:
                meta = _FN_META[mi]
                # try to bind and run
                res = _try_bind_and_run(meta, dry_env, rng)
                if res is None:
                    continue
                param_srcs, ret = res
                picked = (mi, param_srcs, ret)
                break

            if picked is None:
                ok = False
                break

            mi, param_srcs, ret_val = picked
            out_name = f"x{step_no}"
            plan.append(_StepPlan(idx=mi, out_var=out_name, param_srcs=param_srcs))
            dry_env[out_name] = ret_val

        if not ok:
            continue

        # final step: must return Grid. Prefer metas annotated as producing Grid; but accept
        # runtime-validated returns that are structurally a Grid (tuple-of-tuples-of-ints).
        metas_order = [i for i, m in enumerate(_FN_META) if m.produces_grid] or list(
            range(len(_FN_META))
        )
        rng.shuffle(metas_order)

        picked_last: tuple[int, tuple[_ParamSrc, ...], Any] | None = None
        for mi in metas_order:
            meta = _FN_META[mi]
            res = _try_bind_and_run(meta, dry_env, rng)
            if res is None:
                continue
            param_srcs, ret = res

            # Check runtime return looks like Grid if meta isn't annotated as such
            if (
                not meta.produces_grid
                and not _is_grid_structural(type(ret))
                and not (isinstance(ret, tuple) and ret and all(isinstance(r, tuple) for r in ret))
            ):
                continue

            picked_last = (mi, param_srcs, ret)
            break

        if picked_last is None:
            continue

        mi, param_srcs, ret_val = picked_last
        out_name = f"x{L}"
        plan.append(_StepPlan(idx=mi, out_var=out_name, param_srcs=param_srcs))
        dry_env[out_name] = ret_val

        # avoid all-same-function paths
        names = {_FN_META[s.idx].name for s in plan}
        if len(names) == 1:
            continue

        paths.append(plan)

    return paths


# ---------- Plan cleanup: dead-code elimination + renumber ----------


def _live_vars(plan: list[_StepPlan]) -> set[str]:
    """Compute variables that are needed to obtain the final output (the last step's out_var)."""
    needed: set[str] = set()
    if not plan:
        return needed
    needed.add(plan[-1].out_var)
    for step in reversed(plan):
        if step.out_var in needed:
            for src in step.param_srcs:
                if src.var_name is not None:
                    needed.add(src.var_name)
    # Keep input I only if referenced later
    if "I" in needed:
        return needed
    # Add I if any step references I directly
    for step in plan:
        for src in step.param_srcs:
            if src.var_name == "I":
                needed.add("I")
                break
    return needed


def _prune_and_relabel(plan: list[_StepPlan]) -> list[_StepPlan]:
    """Remove steps whose outputs are unused and relabel xk consecutively; update all references."""
    if not plan:
        return plan

    needed = _live_vars(plan)

    # Filter to only live steps (those whose out_var is needed)
    live_steps = [s for s in plan if s.out_var in needed]
    if not live_steps:
        return plan  # should not happen, but keep original just in case

    # Build renaming map for xk, in order of appearance
    new_names: dict[str, str] = {}
    next_idx = 1
    for s in live_steps:
        old = s.out_var
        if old not in new_names:
            new_names[old] = f"x{next_idx}"
            next_idx += 1

    # Rewrite steps with new names and updated parameter references
    rewritten: list[_StepPlan] = []
    for s in live_steps:
        new_out = new_names[s.out_var]
        new_params: list[_ParamSrc] = []
        for p in s.param_srcs:
            if p.var_name is not None:
                nn = "I" if p.var_name == "I" else new_names.get(p.var_name, p.var_name)
                new_params.append(_ParamSrc(var_name=nn, const_name=None, const_val=None))
            else:
                new_params.append(p)
        rewritten.append(_StepPlan(idx=s.idx, out_var=new_out, param_srcs=tuple(new_params)))

    return rewritten


# ---------- Legacy single-flow fallback (ensures progress) ----------


def _legacy_try_make_step_plan(meta: _FnMeta, rng) -> _StepPlan | None:
    """Legacy: choose one flow position (Grid) and fill the rest with constants only, over the required positional prefix."""
    grid_positions = [i for i, sp in enumerate(meta.params) if _accepts_grid(sp.anno)]
    if not grid_positions:
        return None
    flow_pos = rng.choice(grid_positions)

    last_required = max((i for i, sp in enumerate(meta.params) if sp.required), default=-1)
    upto = max(last_required, flow_pos)  # must include the flow pos too

    param_srcs: list[_ParamSrc] = []
    for i, sp in enumerate(meta.params[: upto + 1]):
        if i == flow_pos:
            param_srcs.append(_ParamSrc(var_name="I", const_name=None, const_val=None))
        else:
            picked = _pick_constant_for_annotation(sp.anno, rng)
            if picked is None:
                return None
            cname, cval = picked
            param_srcs.append(_ParamSrc(var_name=None, const_name=cname, const_val=cval))

    return _StepPlan(idx=_FN_META.index(meta), out_var="x1", param_srcs=tuple(param_srcs))


# ---------- Block builder ----------


def make_multiline_block(
    max_lines: int = 5,
    seed: int | None = None,
    n_shots: int = 3,
    max_path_retries: int = 10,
    max_shot_retries: int = 50,
    path_budget: int = 128,
) -> dict[str, Any]:
    """Compose a chain of 2..max_lines DSL functions with Grid input and Grid output.

    Each function may take multiple arguments from: prior variables (I, x1, x2, ...)
    and/or DSL constants. At least one argument per step is a prior variable. The last
    function returns a Grid. After sampling, unused temps are removed and xN are renumbered.
    """
    rng = __import__("random").Random(seed)

    # 1) Sample candidate path plans
    plans = _sample_path_plans(
        rng,
        max_len=max_lines,
        min_len=3,
        path_budget=path_budget,
        middle_pool_size=64,
    )
    if not plans:
        # fallback: single-step that produces Grid and can be satisfied with existing I/consts
        singles = []
        for meta in _FN_META:
            if not meta.produces_grid:
                continue
            step = _legacy_try_make_step_plan(meta, rng)
            if step:
                singles.append([step])
        if not singles:
            raise RuntimeError(
                "No satisfiable DSL plans found (need constants/vars for parameters)."
            )
        plans = singles

    # Try multiple plans until we can generate all shots successfully
    for attempt in range(min(max_path_retries, len(plans))):
        raw_plan = plans[attempt]
        plan = _prune_and_relabel(raw_plan)

        # 2) Render assistant code body
        body_lines = []
        for s in plan:
            meta = _FN_META[s.idx]
            call = _render_call(meta, s.param_srcs)
            body_lines.append(f"    {s.out_var} = {call}")
        body_lines.append(f"    O = {plan[-1].out_var}")
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
            "    x1 = dsl_function(I, xk_or_const, ...)\n"
            "    # line_2\n"
            "    x2 = dsl_function(I_or_x1, xj_or_const, ...)\n"
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

                # Evaluate plan
                env_vals: dict[str, Any] = {}
                input_value = rand_grid()
                env_vals["I"] = input_value

                for s in plan:
                    meta = _FN_META[s.idx]
                    args_vals: list[Any] = []
                    for p in s.param_srcs:
                        if p.var_name is not None:
                            args_vals.append(env_vals[p.var_name])
                        else:
                            args_vals.append(p.const_val)
                    out = meta.func(*args_vals)
                    env_vals[s.out_var] = out

                output_value = env_vals[plan[-1].out_var]
                shots.append({
                    "inputs": {"I": to_jsonable(input_value)},
                    "output": to_jsonable(output_value),
                })
                seed_bump += 1

            except Exception:
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
