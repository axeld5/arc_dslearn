"""Candidate caching and parameter source selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.arc_dslearn.data_gene_callable.dsl_meta import (
    _callable_arity,
    _FnMeta,
    get_callable_consts,
    get_const_by_atom,
    get_unary_callable_consts,
)
from src.arc_dslearn.data_gene_callable.role_templates import (
    _is_binary_combiner,
    _is_geometry_callable,
    _is_predicate_callable,
    _is_scorer_ending,
    _is_selector_callable,
    _is_unary_transform,
    get_desired_arity,
    get_role_templates,
)
from src.arc_dslearn.data_gene_callable.types_utils import _param_atoms

# ---------- Parameter source ----------


@dataclass(frozen=True, slots=True)
class _ParamSrc:
    # Exactly one of (var_name) or (const_name,const_val) is set, unless 'omit' is True
    var_name: str | None
    const_name: str | None
    const_val: Any | None
    omit: bool = False  # True => don't pass this parameter at all


# ---------- Variable window ----------


def _visible_vars(env_vals: dict[str, Any], window: int = 5) -> tuple[str, ...]:
    """Keep 'I', callable vars, and only the last `window` non-callable xN variables."""
    names = [k for k in env_vals if k != "I"]
    xs = [n for n in names if n.startswith("x")]
    xs.sort(key=lambda s: int(s[1:]) if s[1:].isdigit() else -1)
    recent = xs[-window:]
    callables = [n for n in names if callable(env_vals[n])]
    keep = ["I"] + [n for n in callables + recent if n not in ("I",)]
    # Deduplicate preserving order
    seen = set()
    result = []
    for n in keep:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return tuple(result)


def _env_signature(
    env_vals: dict[str, Any], visible: tuple[str, ...]
) -> tuple[tuple[str, bool, int | None], ...]:
    sig = []
    if "I" in env_vals:
        sig.append((
            "I",
            callable(env_vals["I"]),
            _callable_arity(env_vals["I"]) if callable(env_vals["I"]) else None,
        ))
    for k in visible:
        if k == "I":
            continue
        v = env_vals[k]
        sig.append((k, callable(v), _callable_arity(v) if callable(v) else None))
    return tuple(sig)


# ---------- Candidate caching ----------


@lru_cache(maxsize=32768)
def _cached_param_candidates(
    fn_name: str,
    p_index: int,
    atoms_key: frozenset[str],
    env_sig: tuple[tuple[str, bool, int | None], ...],
    want_callable: bool,
    role_hint: str | None = None,  # "scorer", "predicate", "selector", "geometry"
) -> tuple[_ParamSrc, ...]:
    cands: list[_ParamSrc] = []
    desired = get_desired_arity(fn_name, p_index)
    _CALLABLE_CONSTS = get_callable_consts()
    _CONST_BY_ATOM = get_const_by_atom()
    _UNARY_CALLABLE_CONSTS = get_unary_callable_consts()

    # 1) variables first (filtered by callable-ness and arity if desired)
    for var_name, is_call, arity in env_sig:
        if (
            not want_callable
            and not is_call
            or want_callable
            and is_call
            and (desired is None or arity == desired)
        ):
            cands.append(_ParamSrc(var_name=var_name, const_name=None, const_val=None))

    # 2) constants next
    if want_callable:
        pool = _CALLABLE_CONSTS

        # Role-based filtering for callables
        if role_hint == "scorer":
            # Prefer scorers ending in size/width/height/color for argmax/sfilter
            pool = [t for t in _CALLABLE_CONSTS if _is_scorer_ending(t[0])]
            if not pool:  # fallback to all if empty
                pool = _CALLABLE_CONSTS
        elif role_hint == "predicate":
            # Prefer predicates for mfilter/sfilter
            pool = [t for t in _CALLABLE_CONSTS if _is_predicate_callable(t[0])]
            if not pool:
                pool = _CALLABLE_CONSTS
        elif role_hint == "selector":
            # Prefer selectors for extract/center operations
            pool = [t for t in _CALLABLE_CONSTS if _is_selector_callable(t[0])]
            if not pool:
                pool = _CALLABLE_CONSTS
        elif role_hint == "geometry":
            # Prefer geometry for mapply on indices/objects
            pool = [t for t in _CALLABLE_CONSTS if _is_geometry_callable(t[0])]
            if not pool:
                pool = _CALLABLE_CONSTS
        elif role_hint == "unary_transform":
            # For fork/chain/compose branches - prefer unary transform functions
            pool = [t for t in _CALLABLE_CONSTS if _is_unary_transform(t[0])]
            if not pool:
                pool = _CALLABLE_CONSTS
        elif role_hint == "binary_combiner":
            # For fork outer function - prefer binary combiners
            pool = [t for t in _CALLABLE_CONSTS if _is_binary_combiner(t[0])]
            if not pool:
                pool = _CALLABLE_CONSTS

        # Further filter by arity
        if desired == 1:
            pool = [t for t in pool if _callable_arity(t[1]) == 1]
            if not pool:  # fallback
                pool = _UNARY_CALLABLE_CONSTS
        elif desired == 2:
            pool = [t for t in pool if _callable_arity(t[1]) == 2]
            if not pool:
                pool = [t for t in _CALLABLE_CONSTS if _callable_arity(t[1]) == 2]

        for cname, cval in pool:
            cands.append(_ParamSrc(var_name=None, const_name=cname, const_val=cval))
    else:
        atoms = set(atoms_key)
        atoms.discard("Grid")
        for atom in atoms:
            for cname, cval in _CONST_BY_ATOM.get(atom, ()):
                cands.append(_ParamSrc(var_name=None, const_name=cname, const_val=cval))

    # 3) fallback: if arity filter produced nothing, allow any callable to avoid dead-ends
    if want_callable and not cands:
        for var_name, is_call, _ in env_sig:
            if is_call:
                cands.append(_ParamSrc(var_name=var_name, const_name=None, const_val=None))
        for cname, cval in _CALLABLE_CONSTS:
            cands.append(_ParamSrc(var_name=None, const_name=cname, const_val=cval))

    return tuple(cands)


# ---------- Role-based reordering ----------


@dataclass(frozen=True, slots=True)
class _Binding:
    # What the code will show:
    src: _ParamSrc
    # What we used in the dry-run to validate the step:
    value: Any


def _reorder_candidates_by_role(
    meta: _FnMeta, cand_lists: list[list[_Binding]], env_vals: dict[str, Any]
) -> None:
    """Stable-reorder candidate lists in-place to match preferred role templates when available."""
    tmpl_list = get_role_templates().get(meta.name)
    if not tmpl_list:
        return
    tmpl = tmpl_list[0]  # pick first template for simplicity
    if len(tmpl) != len(cand_lists):
        return

    def is_var(b: _Binding) -> bool:
        return b.src.var_name is not None

    def is_const(b: _Binding) -> bool:
        return b.src.var_name is None and b.src.const_name is not None

    def is_callable_val(b: _Binding) -> bool:
        v = b.value if b.src.var_name is None else env_vals.get(b.src.var_name)
        return callable(v)

    def is_tuple_val(b: _Binding) -> bool:
        v = b.value if b.src.var_name is None else env_vals.get(b.src.var_name)
        return isinstance(v, tuple)

    def key_for(role: str, b: _Binding) -> int:
        # Lower key == higher priority
        if role == "var_callable":
            return 0 if (is_var(b) and is_callable_val(b)) else 1
        if role == "const_callable":
            return 0 if (is_const(b) and is_callable_val(b)) else 1
        if role == "var_non_callable":
            return 0 if (is_var(b) and not is_callable_val(b)) else 1
        if role == "var_tuple":
            return 0 if (is_var(b) and is_tuple_val(b)) else 1
        return 1

    for i, role in enumerate(tmpl):
        # stable sort by role preference
        cand_lists[i].sort(key=lambda b: key_for(role, b))


# ---------- Feasibility prefilter ----------


def _env_has_callable(env_vals: dict[str, Any]) -> bool:
    return any(callable(v) for v in env_vals.values())


def _env_has_non_callable(env_vals: dict[str, Any]) -> bool:
    return any(not callable(v) for v in env_vals.values())


def _can_satisfy(meta: _FnMeta, env_vals: dict[str, Any]) -> bool:
    """Cheap static check: is it possible to satisfy required params by env or constants (incl. callables)."""
    from src.arc_dslearn.data_gene_callable.types_utils import _accepts_grid, _is_callable_anno

    _CALLABLE_CONSTS = get_callable_consts()
    _CONST_BY_ATOM = get_const_by_atom()

    has_call = _env_has_callable(env_vals)
    has_non = _env_has_non_callable(env_vals)
    for sp in meta.params:
        if not sp.required:
            continue
        if _is_callable_anno(sp.anno):
            if not (has_call or _CALLABLE_CONSTS):
                return False
        else:
            atoms = _param_atoms(sp.anno)
            has_const = any(_CONST_BY_ATOM.get(a) for a in atoms if a != "Grid")
            if not (has_non or has_const or _accepts_grid(sp.anno)):
                return False
    return True


# Exports
__all__ = [
    "_ParamSrc",
    "_Binding",
    "_visible_vars",
    "_env_signature",
    "_cached_param_candidates",
    "_reorder_candidates_by_role",
    "_can_satisfy",
    "_env_has_callable",
    "_env_has_non_callable",
]
