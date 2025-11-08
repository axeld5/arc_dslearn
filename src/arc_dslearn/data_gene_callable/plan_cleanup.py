"""Plan cleanup: dead-code elimination and renumbering."""

from __future__ import annotations

from dataclasses import dataclass

from src.arc_dslearn.data_gene_callable.candidates import _ParamSrc

# ---------- Step plan ----------


@dataclass(frozen=True, slots=True)
class _StepPlan:
    idx: int  # index into _FN_META
    out_var: str  # variable defined by this step (e.g., "x3")
    param_srcs: tuple[_ParamSrc, ...]  # one per passed positional parameter (var or const)


# ---------- Dead code elimination ----------


def _live_vars(plan: list[_StepPlan]) -> set[str]:
    """Compute variables that are needed to obtain the final output (the last step's out_var).

    Enhanced to preserve callable-producing steps that are used within 1-2 steps,
    even if they appear dead in simple backward traversal.
    """
    from src.arc_dslearn.data_gene_callable.dsl_meta import get_fn_meta

    _FN_META = get_fn_meta()

    needed: set[str] = set()
    if not plan:
        return needed
    needed.add(plan[-1].out_var)

    # Standard backward liveness analysis
    for step in reversed(plan):
        if step.out_var in needed:
            for src in step.param_srcs:
                if src.var_name is not None:
                    needed.add(src.var_name)

    # Enhanced: keep callable-producing steps alive if consumed within next 1-2 steps
    # This prevents premature DCE of HOF scaffolds
    for i, step in enumerate(plan):
        meta = _FN_META[step.idx]
        if meta.returns_callable:
            # Check if this callable is used in the next 1-2 steps
            for j in range(i + 1, min(i + 3, len(plan))):
                future_step = plan[j]
                for src in future_step.param_srcs:
                    if src.var_name == step.out_var:
                        # This callable is consumed soon, keep it alive
                        needed.add(step.out_var)
                        # And its dependencies
                        for dep_src in step.param_srcs:
                            if dep_src.var_name is not None:
                                needed.add(dep_src.var_name)
                        break

    # Keep input I only if referenced
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


# Exports
__all__ = [
    "_StepPlan",
    "_live_vars",
    "_prune_and_relabel",
]
