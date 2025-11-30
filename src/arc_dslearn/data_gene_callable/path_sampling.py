"""Path planning and sampling logic for DSL function chains."""

from __future__ import annotations

from typing import Any

from src.arc_dslearn.data_gene_callable.candidates import (
    _Binding,
    _cached_param_candidates,
    _can_satisfy,
    _env_signature,
    _ParamSrc,
    _reorder_candidates_by_role,
    _visible_vars,
)
from src.arc_dslearn.data_gene_callable.context import _check_deadline, _run_with_timeout
from src.arc_dslearn.data_gene_callable.dsl_meta import (
    _FnMeta,
    _is_pair_applicator,
    _meta_wants_callable,
    _produces_tuple_type,
    _weighted_shuffle,
    get_all_idx,
    get_fn_meta,
)
from src.arc_dslearn.data_gene_callable.plan_cleanup import _StepPlan
from src.arc_dslearn.data_gene_callable.planning_memo import (
    _call_with_optional_memo,
    _is_grid_like,
    _make_small_grid_pool,
    _too_big,
)
from src.arc_dslearn.data_gene_callable.role_templates import (
    get_hof_micro_templates,
    get_role_hint,
    get_source_sink_pairs,
)
from src.arc_dslearn.data_gene_callable.types_utils import _is_callable_anno

# ---------- Call rendering ----------


def _render_call(meta: _FnMeta, srcs: tuple[_ParamSrc, ...]) -> str:
    """Render a python call, using keywords if any parameter is omitted or any earlier param is omitted."""
    # pretty direct-call: apply(fn, x) -> fn(x)
    if meta.name in {"apply", "mapply"} and len(srcs) >= 2:
        fn_src, *arg_srcs = srcs
        if fn_src.var_name is not None:
            args_txt = ", ".join((s.var_name or s.const_name) for s in arg_srcs if not s.omit)
            return f"{fn_src.var_name}({args_txt})"

    # pretty direct-call: papply/prapply(fn, a, b) -> fn(a, b)
    if meta.name in {"papply", "prapply", "mpapply"} and len(srcs) >= 3:
        fn_src, a_src, b_src = srcs[:3]
        if fn_src.var_name is not None and not any(s.omit for s in (a_src, b_src)):
            args_txt = ", ".join((s.var_name or s.const_name) for s in (a_src, b_src))
            return f"{fn_src.var_name}({args_txt})"

    # fall back to default rendering
    any_omitted = any(s.omit for s in srcs)
    if any_omitted:
        parts = []
        for spec, s in zip(meta.params, srcs, strict=False):
            if s.omit:
                continue
            val = s.var_name if s.var_name is not None else s.const_name
            parts.append(f"{spec.name}={val}")
        return f"{meta.name}({', '.join(parts)})"
    else:
        parts = []
        for s in srcs:
            val = s.var_name if s.var_name is not None else s.const_name
            parts.append(val)
        return f"{meta.name}({', '.join(parts)})"


# ---------- Path length sampling ----------


def _biased_length(rng, min_len: int, max_len: int) -> int:
    """Sample path length with a distribution that favors longer but achievable paths.

    Uses a capped linear distribution that:
    - Favors longer paths (linear growth up to a "sweet spot")
    - Caps weights after ~15-20 steps to avoid wasting attempts on very long paths
    - Still allows occasional very long path attempts
    """
    lengths = list(range(min_len, max_len + 1))

    # Linear weights up to step 15, then flatten (longer paths are rare but possible)
    # This concentrates attempts on achievable lengths (5-15) while still trying longer
    sweet_spot = 15
    weights = []
    for _i, length in enumerate(lengths):
        if length <= sweet_spot:
            weights.append(length)  # Linear growth: 3, 4, 5, ..., 15
        else:
            # Slowly decay for very long paths: 15, 14, 13, ... (but floor at 5)
            decay_weight = max(5, sweet_spot - (length - sweet_spot) // 2)
            weights.append(decay_weight)

    return rng.choices(lengths, weights=weights, k=1)[0]


# ---------- Binding and running ----------


def _try_bind_and_run(
    meta: _FnMeta,
    env_vals: dict[str, Any],
    rng,
    max_trials: int = 8,
    early_trials: int = 3,
    require_grid_from_I: bool = False,  # enforce x1 consumes I as Grid
    max_intermediate_cells: int = 64,
    max_intermediate_items: int = 256,
    last_callable_var: str | None = None,  # bias toward consuming this callable
) -> tuple[tuple[_ParamSrc, ...], Any, bool, bool] | None:
    """Try to build a valid call; return (param_srcs, return_value, used_grid, used_I_grid) on success."""
    params = meta.params

    # (1) variable window
    visible = _visible_vars(env_vals)
    env_sig = _env_signature(env_vals, visible)
    cand_lists: list[list[_Binding]] = []

    # Precompute candidate sources per parameter (variables + constants + "omit" for optionals)
    for i, sp in enumerate(params):
        want_callable = _is_callable_anno(sp.anno)
        from src.arc_dslearn.data_gene_callable.types_utils import _param_atoms

        atoms_key = _param_atoms(sp.anno)  # frozenset[str], hashable
        role_hint = get_role_hint(meta.name, i) if want_callable else None
        param_srcs = list(
            _cached_param_candidates(meta.name, i, atoms_key, env_sig, want_callable, role_hint)
        )

        # Bind values now (variables pull from env, constants already carry const_val)
        cands: list[_Binding] = []
        for ps in param_srcs:
            if ps.var_name is not None:
                if ps.var_name not in env_vals:
                    continue
                val = env_vals[ps.var_name]
                # additional guard: ensure callable-ness aligns with want_callable
                if want_callable != callable(val):
                    continue
                cands.append(_Binding(src=ps, value=val))
            else:
                cands.append(_Binding(src=ps, value=ps.const_val))

        # omission for optionals
        if not sp.required:
            cands.append(
                _Binding(
                    src=_ParamSrc(var_name=None, const_name=None, const_val=None, omit=True),
                    value=None,
                )
            )

        cand_lists.append(cands)

    # (2) role templates bias
    _reorder_candidates_by_role(meta, cand_lists, env_vals)

    # (3) producer→consumer coupling: prefer last_callable_var for callable params
    if last_callable_var is not None:
        for i, sp in enumerate(params):
            if _is_callable_anno(sp.anno):
                # move bindings that use last_callable_var to the front
                cl = cand_lists[i]
                prefer = [b for b in cl if b.src.var_name == last_callable_var]
                others = [b for b in cl if b.src.var_name != last_callable_var]
                if prefer:
                    cand_lists[i] = prefer + others

    # Try randomized attempts (early few, then remainder if needed)
    def _attempts(limit: int):
        for _ in range(limit):
            # Check deadline before each attempt
            _check_deadline()

            picks: list[_Binding] = []
            used_var = False
            used_var_names: set[str] = set()  # Track used variables for diversity

            for cands in cand_lists:
                var_cands = [b for b in cands if b.src.var_name is not None and not b.src.omit]

                # For diversity: prefer variables not already used (for same-type params)
                # This helps functions like manhattan(a, b) get different patches
                unused_var_cands = [b for b in var_cands if b.src.var_name not in used_var_names]

                if unused_var_cands and rng.random() < 0.8:
                    # Prefer unused variable
                    b = rng.choice(unused_var_cands)
                elif var_cands and rng.random() < 0.75:
                    b = rng.choice(var_cands)
                else:
                    b = rng.choice(cands)

                if b.src.var_name is not None:
                    used_var = True
                    used_var_names.add(b.src.var_name)
                picks.append(b)

            # Make sure required params are not omitted
            ok_required = True
            for idx, (b, sp) in enumerate(zip(picks, params, strict=False)):
                if sp.required and b.src.omit:
                    non_omit = [c for c in cand_lists[idx] if not c.src.omit]
                    if non_omit:
                        b = rng.choice(non_omit)
                        picks[idx] = b
                    else:
                        ok_required = False
                        break
            if not ok_required:
                continue

            # Ensure at least one variable is used (connectivity)
            if not used_var and any(b.src.var_name is not None for cl in cand_lists for b in cl):
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

            any_omit = any(b.src.omit for b in picks)

            try:
                if any_omit:
                    kwargs = {}
                    for sp, b in zip(params, picks, strict=False):
                        if b.src.omit:
                            continue
                        kwargs[sp.name] = b.value
                    ret = meta.func(**kwargs)  # keywords: call directly
                    param_srcs = tuple(b.src for b in picks)
                    used_grid = any(
                        bs.src.var_name is not None and _is_grid_like(env_vals[bs.src.var_name])
                        for bs in picks
                        if bs.src.var_name is not None
                    )
                    used_I_grid = any(
                        bs.src.var_name == "I" for bs in picks if bs.src.var_name is not None
                    )
                    if require_grid_from_I and not used_I_grid:
                        raise RuntimeError("first step didn't consume I as Grid")
                else:
                    args = [b.value for b in picks]
                    ret = _call_with_optional_memo(meta.name, args)
                    param_srcs = tuple(b.src for b in picks)
                    used_grid = any(
                        bs.src.var_name is not None and _is_grid_like(env_vals[bs.src.var_name])
                        for bs in picks
                        if bs.src.var_name is not None
                    )
                    used_I_grid = any(
                        bs.src.var_name == "I" for bs in picks if bs.src.var_name is not None
                    )
                    if require_grid_from_I and not used_I_grid:
                        raise RuntimeError("first step didn't consume I as Grid")

                # (3) size guard (planning only)
                if _too_big(
                    ret, max_cells=max_intermediate_cells, max_items=max_intermediate_items
                ):
                    raise RuntimeError("intermediate too big")

                return param_srcs, ret, used_grid, used_I_grid
            except Exception:
                continue
        return None

    # shrink trial budget when graphs get long
    scale = 1.0 if len(env_vals) < 20 else (0.6 if len(env_vals) < 50 else 0.4)
    res = _attempts(max(1, int(early_trials * scale)))
    if res is not None:
        return res
    return _attempts(max(0, int((max_trials - early_trials) * scale)))


# ---------- Single path building ----------


def _try_build_single_path(
    rng,
    pool: tuple,
    min_len: int,
    max_len: int,
    allow_setup_steps: int,
    max_intermediate_cells: int,
    max_intermediate_items: int,
) -> list[_StepPlan] | None:
    """Try to build a single valid path plan. Returns the plan or None if failed.

    Key feature: If targeting a long path (e.g., 40 steps) but failing at step 25,
    we try to salvage the 25-step partial path by adding a final Grid-producing step.
    This makes long path attempts much more efficient.
    """
    import src.arc_dslearn.arc_dsl.dsl as dsl

    _FN_META = get_fn_meta()
    _ALL_IDX = get_all_idx()

    # fresh dry-run environment
    dry_env: dict[str, Any] = {}
    # ALWAYS pick from the small pool in planning (increases memo hits)
    dry_env["I"] = rng.choice(pool)
    # Seed one callable so higher-order primitives can chain immediately
    dry_env["F0"] = dsl.identity

    # choose target length
    L = _biased_length(rng, min_len=min_len, max_len=max_len)
    plan: list[_StepPlan] = []

    ok = True
    ever_used_grid = False
    have_callable_var = any(callable(v) for k, v in dry_env.items() if k != "I")
    last_callable_var = None
    last_produced_fn = None  # track last function name for source/sink biasing
    last_produced_tuple = False  # track if last step produced tuple-like output
    grid_consumed_at_step = None  # track when we first consumed I

    # Track salvageable state: (plan_copy, env_copy, step_no) at last valid point
    # We can salvage if: grid was consumed AND we have enough steps for min_len
    last_salvageable: tuple[list[_StepPlan], dict[str, Any], int] | None = None

    # (5) negative cache for this attempt
    failed_meta_env: set[tuple[int, tuple[tuple[str, bool], ...]]] = set()

    for step_no in range(1, L):
        # Cooperative timeout check - raises TimeoutException if deadline exceeded
        _check_deadline()

        # Prefilter by feasibility
        feasible = [i for i in _ALL_IDX if _can_satisfy(_FN_META[i], dry_env)]
        if not feasible:
            ok = False
            break

        # Relaxed constraint: I must be consumed within first allow_setup_steps
        # Early steps can build callables, but we need grid consumption soon
        if step_no <= allow_setup_steps and grid_consumed_at_step is None:
            # Within setup window: prefer but don't require grid consumption
            # Boost grid consumers
            grid_consumers = [i for i in feasible if _FN_META[i].accepts_grid]
            if grid_consumers and step_no == allow_setup_steps:
                # Last chance: must consume grid
                feasible = grid_consumers
            elif grid_consumers and rng.random() < 0.7:
                # Soft preference for consuming grid early
                feasible = grid_consumers
        elif grid_consumed_at_step is None:
            # Past setup window and haven't consumed grid yet: fail
            ok = False
            break

        # build a base candidate pool
        cands = list(feasible)

        # Source/sink biasing: if last step produced a known source, bias toward compatible sinks
        _SOURCE_SINK_PAIRS = get_source_sink_pairs()
        if last_produced_fn and last_produced_fn in _SOURCE_SINK_PAIRS:
            preferred_sinks = _SOURCE_SINK_PAIRS[last_produced_fn]
            sink_metas = [i for i in feasible if _FN_META[i].name in preferred_sinks]
            if sink_metas:
                # Heavily upweight these sinks
                cands += sink_metas * 3

        # Micro-template biasing: if last_callable_var exists, check for micro-template matches
        _HOF_MICRO_TEMPLATES = get_hof_micro_templates()
        if last_callable_var and last_produced_fn:
            for template in _HOF_MICRO_TEMPLATES:
                if template.producer == last_produced_fn:
                    # Strongly bias toward the consumer
                    consumer_metas = [i for i in feasible if _FN_META[i].name == template.consumer]
                    if consumer_metas:
                        cands += consumer_metas * 4  # strong bias

        # Tuple/product pipeline biasing: if last step produced tuple-like, bias toward pair applicators
        if last_produced_tuple:
            pair_applicator_metas = [i for i in feasible if _is_pair_applicator(i)]
            if pair_applicator_metas:
                cands += pair_applicator_metas * 3  # strong bias toward papply/prapply/mpapply

        # 1) if we have callable vars, upweight metas that CONSUME callables
        if have_callable_var:
            consumer_metas = [i for i in feasible if _meta_wants_callable(i)]
            cands += consumer_metas
            # Extra boost for immediate callable consumption throughout the path
            if last_callable_var and step_no <= L:  # boost throughout entire path
                cands += consumer_metas

        # 2) if we don't have callable vars, upweight metas that PRODUCE callables
        # Allow callable production at any point, not just early (needed for long paths)
        if (not have_callable_var) and (step_no <= max(5, L // 3)):
            cands += [i for i in feasible if _FN_META[i].returns_callable]

        # 3) Boost combinator functions (fork, chain, compose) - key patterns in solvers
        # These are essential for complex compositions like fork() + compose() chains
        combinator_names = {"fork", "chain", "compose"}
        combinator_metas = [i for i in feasible if _FN_META[i].name in combinator_names]
        if combinator_metas:
            # Moderate boost (2x) to encourage combinator usage throughout the path
            cands += combinator_metas * 2

        # 4) Count how many patch/object variables we have for two-patch function boosting
        patch_vars = sum(
            1
            for v in dry_env.values()
            if isinstance(v, frozenset)
            and v
            and not callable(v)
            and isinstance(next(iter(v)), tuple)  # looks like Object/Indices
        )

        # 5) Boost extractors (first, last, argmax, argmin) after container-producing functions
        # These enable two-patch functions by creating individual patches
        if last_produced_fn in {"objects", "fgpartition", "partition", "frontiers"}:
            extractor_names = {"first", "last", "argmax", "argmin", "extract"}
            extractor_metas = [i for i in feasible if _FN_META[i].name in extractor_names]
            if extractor_metas:
                cands += extractor_metas * 3  # strong boost after container producers

        # 6) Boost two-patch functions when we have 2+ patch variables
        if patch_vars >= 2:
            two_patch_names = {
                "manhattan",
                "adjacent",
                "gravitate",
                "position",
                "hmatching",
                "vmatching",
            }
            two_patch_metas = [i for i in feasible if _FN_META[i].name in two_patch_names]
            if two_patch_metas:
                cands += two_patch_metas * 3  # strong boost when we can actually use them

        metas_order = _weighted_shuffle(cands, rng)

        picked: tuple[int, tuple[_ParamSrc, ...], Any] | None = None
        for mi in metas_order:
            meta = _FN_META[mi]
            # Negative cache guard
            visible = _visible_vars(dry_env)
            env_sig = _env_signature(dry_env, visible)
            if (mi, env_sig) in failed_meta_env:
                continue

            res = _try_bind_and_run(
                meta,
                dry_env,
                rng,
                require_grid_from_I=(step_no == 1),
                max_intermediate_cells=max_intermediate_cells,
                max_intermediate_items=max_intermediate_items,
                last_callable_var=last_callable_var,
            )
            if res is None:
                failed_meta_env.add((mi, env_sig))
                continue
            param_srcs, ret, used_grid, _used_I_grid = res
            picked = (mi, param_srcs, ret)
            ever_used_grid = ever_used_grid or used_grid
            if _FN_META[mi].returns_callable:
                have_callable_var = True
            break

        if picked is None:
            ok = False
            break

        mi, param_srcs, ret_val = picked
        out_name = f"x{step_no}"
        plan.append(_StepPlan(idx=mi, out_var=out_name, param_srcs=param_srcs))
        dry_env[out_name] = ret_val

        # Track function name for source/sink biasing
        picked_meta = _FN_META[mi]
        last_produced_fn = picked_meta.name

        # Track callable variable for producer→consumer coupling
        if picked_meta.returns_callable:
            last_callable_var = out_name
        else:
            # occasionally clear it to avoid hard coupling, but less often to keep scaffolds alive
            if rng.random() < 0.2:  # reduced from 0.3
                last_callable_var = None

        # Track if this step produced tuple-like output
        last_produced_tuple = _produces_tuple_type(picked_meta)

        # Track when I was first consumed
        if grid_consumed_at_step is None and used_grid:
            grid_consumed_at_step = step_no

        # Save salvageable checkpoint: if grid consumed and we have enough steps
        # We need at least min_len steps total (current + 1 for final step)
        # Only save every few steps to avoid overhead
        if (
            grid_consumed_at_step is not None
            and len(plan) + 1 >= min_len
            and (step_no % 3 == 0 or step_no >= L - 2)
        ):
            last_salvageable = (list(plan), dict(dry_env), step_no)

    # Try to salvage if main path failed but we have a valid partial
    if not ok or not ever_used_grid or grid_consumed_at_step is None:
        if last_salvageable is not None:
            # Restore to last salvageable state and try to add final step
            plan, dry_env, salvage_step = last_salvageable
            L = salvage_step + 1  # Adjust L for final step naming
        else:
            return None

    # final step: must return Grid. Prefer metas annotated as producing Grid; but accept
    # runtime-validated returns that are structurally a Grid (tuple-of-tuples-of-ints).
    final_cands = [i for i, m in enumerate(_FN_META) if m.produces_grid] or _ALL_IDX
    final_cands = [i for i in final_cands if _can_satisfy(_FN_META[i], dry_env)]
    if not final_cands:
        return None
    metas_order = _weighted_shuffle(final_cands, rng)

    picked_last: tuple[int, tuple[_ParamSrc, ...], Any] | None = None
    for mi in metas_order:
        meta = _FN_META[mi]
        res = _try_bind_and_run(
            meta,
            dry_env,
            rng,
            max_trials=10,
            early_trials=3,
            max_intermediate_cells=max_intermediate_cells,
            max_intermediate_items=max_intermediate_items,
        )
        if res is None:
            continue
        param_srcs, ret, _used_grid, _used_I_grid = res

        # Check runtime return looks like Grid if meta isn't annotated as such
        if not meta.produces_grid and (
            not (isinstance(ret, tuple) and ret and isinstance(ret[0], tuple))
        ):
            continue

        picked_last = (mi, param_srcs, ret)
        break

    if picked_last is None:
        return None

    mi, param_srcs, ret_val = picked_last
    out_name = f"x{L}"
    plan.append(_StepPlan(idx=mi, out_var=out_name, param_srcs=param_srcs))
    dry_env[out_name] = ret_val

    # avoid all-same-function paths
    names = {_FN_META[s.idx].name for s in plan}
    if len(names) == 1:
        return None

    return plan


# ---------- Path sampling with timeout ----------


def _sample_path_plans(
    rng,
    max_len: int,
    min_len: int = 3,
    path_budget: int = 128,
    planning_shape: tuple[int, int] = (6, 6),
    max_intermediate_cells: int = 128,  # Increased to allow longer paths
    max_intermediate_items: int = 512,  # Increased to allow longer paths
    max_paths: int = 24,
    allow_setup_steps: int = 3,  # Increased to allow more setup flexibility
    path_timeout: float = 15.0,  # Increased timeout for longer paths
) -> list[list[_StepPlan]]:
    """Build plans by *executing steps during sampling* on a single dry-run input.

    The last chosen function must return a Grid (verified by return typing OR by runtime check).
    Enforces: I consumed within first 1-2 steps, xLast returns Grid. Middle steps can be anything.
    Setup steps can build callables before consuming I.

    Each path exploration has a timeout; paths that take too long are discarded.
    """
    paths: list[list[_StepPlan]] = []
    _FN_META = get_fn_meta()
    if not _FN_META:
        return paths

    # Create small fixed pool for planning phase to maximize memoization hits
    # Pool has 5 grids with random dimensions between 4-6 (both height and width)
    pool = _make_small_grid_pool(rng, k=5, min_dim=4, max_dim=6)

    for _ in range(path_budget):
        if len(paths) >= max_paths:
            break

        # Try to build a single path with timeout
        plan, timed_out = _run_with_timeout(
            _try_build_single_path,
            path_timeout,
            rng,
            pool,
            min_len,
            max_len,
            allow_setup_steps,
            max_intermediate_cells,
            max_intermediate_items,
        )

        if timed_out or plan is None:
            # Path exploration took too long or failed, skip it
            continue

        paths.append(plan)

    return paths


# Exports
__all__ = [
    "_render_call",
    "_biased_length",
    "_try_bind_and_run",
    "_try_build_single_path",
    "_sample_path_plans",
]
