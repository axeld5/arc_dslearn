"""Block generation logic for DSL function training data.

This module orchestrates the generation of training blocks by composing DSL functions
with Grid input and Grid output. It coordinates multiple submodules for planning,
path sampling, quality checking, and more.
"""

from __future__ import annotations

from typing import Any

from src.arc_dslearn.data_gene_callable.candidates import _ParamSrc
from src.arc_dslearn.data_gene_callable.context import planning_phase
from src.arc_dslearn.data_gene_callable.data_processing import compact_format, to_jsonable
from src.arc_dslearn.data_gene_callable.dsl_meta import (
    DSL_FUNCTIONS_BLOCK,
    _FnMeta,
    _pick_constant_for_annotation,
    get_fn_meta,
)
from src.arc_dslearn.data_gene_callable.generators import rand_grid
from src.arc_dslearn.data_gene_callable.path_sampling import (
    _render_call,
    _sample_path_plans,
)
from src.arc_dslearn.data_gene_callable.plan_cleanup import (
    _prune_and_relabel,
    _StepPlan,
)
from src.arc_dslearn.data_gene_callable.planning_memo import clear_planning_memo
from src.arc_dslearn.data_gene_callable.quality_checks import _check_shots_quality
from src.arc_dslearn.data_gene_callable.types_utils import (
    _accepts_grid,
    _expand_atoms_cached,
    _is_callable_anno,
)

# ---------- Legacy single-flow fallback (ensures progress) ----------


def _legacy_try_make_step_plan(meta: _FnMeta, rng) -> _StepPlan | None:
    """Legacy: choose one flow position (Grid) and fill the rest with constants only, over the required positional prefix."""
    from src.arc_dslearn.data_gene_callable.dsl_meta import get_callable_consts

    _CALLABLE_CONSTS = get_callable_consts()
    _FN_META = get_fn_meta()

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
            if _is_callable_anno(sp.anno):
                cname, cval = rng.choice(_CALLABLE_CONSTS)
                param_srcs.append(_ParamSrc(var_name=None, const_name=cname, const_val=cval))
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
    path_budget: int = 64,
    include_examples_str: bool = True,  # memory knob: disable to skip big pretty strings
    planning_shape: tuple[int, int] = (6, 6),  # tiny grids for planning
    shot_shape: tuple[int, int] | None = (
        30,
        30,
    ),  # large grids for final shots; None=use rand_grid default
) -> dict[str, Any]:
    """Compose a chain of 2..max_lines DSL functions with Grid input and Grid output.

    Each function may take multiple arguments from: prior variables (I, x1, x2, ...)
    and/or DSL constants. x1 must consume I (Grid); the last function returns a Grid.
    Middle steps can be callable/numeric only; DCE ensures only used temps remain.
    """
    rng = __import__("random").Random(seed)
    _FN_META = get_fn_meta()

    # 1) Sample candidate path plans (wrapped in planning_phase for memoization)
    with planning_phase():
        plans = _sample_path_plans(
            rng,
            max_len=max_lines,
            min_len=3,
            path_budget=path_budget,
            planning_shape=planning_shape,
            max_intermediate_cells=planning_shape[0]
            * planning_shape[1]
            * 2,  # allow larger intermediates
            max_intermediate_items=512,  # increased for longer paths
            max_paths=32,  # try more paths to find long ones
            path_timeout=45.0,  # more time for longer paths
        )
    if not plans:
        # fallback: single-step that produces Grid and can be satisfied with existing I/consts
        singles = []
        for meta in _FN_META:
            if not meta.produces_grid or not meta.accepts_grid:
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
                # keep RNG deterministic per shot without affecting global state
                local_seed = (
                    (seed or 0) + seed_bump * 1000 if seed is not None else rng.randrange(10**9)
                )
                __import__("random").Random(local_seed)  # kept for future per-shot rng if needed

                # Evaluate plan on a *full* grid for shots
                if shot_shape is None:
                    input_value = rand_grid()
                else:
                    sh, sw = shot_shape
                    try:
                        input_value = rand_grid(h=sh, w=sw)
                    except TypeError:
                        input_value = rand_grid()

                env_vals: dict[str, Any] = {"I": input_value}
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
            examples = ""
            if include_examples_str:
                examples = "\n".join(
                    f"# Example {j + 1}\nI = {compact_format(s['inputs']['I'])}\n# Desired → {compact_format(s['output'])}\n"
                    for j, s in enumerate(shots)
                )

            # Check quality of shots
            is_high_quality, quality_reason = _check_shots_quality(shots)

            return {
                "name": "|".join(_FN_META[s.idx].name for s in plan),
                "system_prompt": system_prompt,
                "user_prompt": (examples + "\n### Task\n" + user_template)
                if include_examples_str
                else ("### Task\n" + user_template),
                "assistant_prompt": assistant_prompt,
                "shots": shots,
                "lines": len(plan),
                "quality": "high" if is_high_quality else "low",
                "quality_reason": quality_reason,
            }

    raise RuntimeError("Failed to generate block with sampled multi-arg plans.")


# ---------- Cache maintenance (memory control) ----------


def clear_internal_caches() -> None:
    """Clear internal LRU caches to release memory between large batches."""
    from src.arc_dslearn.data_gene_callable.candidates import _cached_param_candidates

    _expand_atoms_cached.cache_clear()
    _cached_param_candidates.cache_clear()
    clear_planning_memo()


# Exports
__all__ = [
    "make_multiline_block",
    "clear_internal_caches",
    "DSL_FUNCTIONS_BLOCK",
]
