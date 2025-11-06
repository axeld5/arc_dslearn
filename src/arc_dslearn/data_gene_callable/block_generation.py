"""Block generation logic for DSL function training data (multi-arg vars + cleanup + fallback) with callable support, performance & memory optimizations, grid-size knobs, restricted variable window, role templates (bias toward useful patterns), early shape/size guards, heavy-op budget per path, and per-meta negative cache (fail-fast)."""

from __future__ import annotations

import collections.abc as cabc
import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Tuple, Union, get_args, get_origin
from typing import Callable as TypingCallable
from typing import Container as TypingContainer

import src.arc_dslearn.arc_dsl.arc_types as T
import src.arc_dslearn.arc_dsl.constants as C
import src.arc_dslearn.arc_dsl.dsl as dsl
from src.arc_dslearn.data_gene_callable.data_processing import compact_format, to_jsonable
from src.arc_dslearn.data_gene_callable.generators import rand_grid

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


def _public_dsl_functions() -> list[tuple[str, TypingCallable[..., Any]]]:
    """Cache public DSL callables once."""
    return [(n, f) for n, f in inspect.getmembers(dsl, inspect.isfunction) if not n.startswith("_")]


_DSL_FUNCS: list[tuple[str, TypingCallable[..., Any]]] = _public_dsl_functions()
_CALLABLE_CONSTS: list[tuple[str, TypingCallable[..., Any]]] = _DSL_FUNCS[:]


def dsl_functions_summary() -> str:
    """Return a one-line bullet list of DSL functions."""
    return "\n".join(f"- {name}" for name, _ in _DSL_FUNCS)


DSL_FUNCTIONS_BLOCK = dsl_functions_summary()


# ---------- Type helpers (alias/union aware) ----------


def _is_grid_structural(anno_or_val: Any) -> bool:
    """Tuple[Tuple[int]] structural recognition (besides T.Grid/'Grid'). Accepts types or values."""
    # If value instance of grid-like
    if isinstance(anno_or_val, tuple) and anno_or_val and isinstance(anno_or_val[0], tuple):
        # assume grid of ints
        return True
    anno = anno_or_val
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
    # Containers
    if anno in (cabc.Container, TypingContainer, "Container", "ContainerContainer"):
        return frozenset({"Container"})  # soft tag
    origin = get_origin(anno)
    if origin in (cabc.Container, TypingContainer):
        return frozenset({"Container"})

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


@lru_cache(maxsize=32768)
def _expand_atoms_cached(anno: Any) -> frozenset[str]:
    """Wrap around _expand_atoms to avoid repeating work."""
    return _expand_atoms(anno)


def _param_atoms(anno: Any) -> frozenset[str]:
    return _expand_atoms_cached(anno)


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


# ---------- Callable & constants helpers ----------


def _is_callable_anno(anno: Any) -> bool:
    """Return True if annotation is typing.Callable *or* collections.abc.Callable, or Union containing it."""
    if anno is TypingCallable or anno is cabc.Callable:
        return True
    origin = get_origin(anno)
    if origin in (TypingCallable, cabc.Callable):
        return True
    if origin is Union:
        return any(_is_callable_anno(a) for a in get_args(anno))
    return isinstance(anno, str) and anno.lower() == "callable"


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


@dataclass(frozen=True, slots=True)
class _ParamSpec:
    name: str
    anno: Any
    required: bool  # True if no default


@dataclass(frozen=True, slots=True)
class _FnMeta:
    name: str
    func: TypingCallable[..., Any]
    params: tuple[_ParamSpec, ...]  # ordered positional-or-kw params
    flow_positions: tuple[int, ...]  # indices whose annotation accepts Grid (may be empty)
    produces_grid: bool
    return_atoms: frozenset[str]  # expanded atoms of return type
    accepts_grid: bool  # any param can take a Grid
    returns_callable: bool  # return is a Callable


def _collect_fn_meta() -> list[_FnMeta]:
    metas: list[_FnMeta] = []
    for name, func in _DSL_FUNCS:
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
        returns_callable = _is_callable_anno(ret)

        metas.append(
            _FnMeta(
                name=name,
                func=func,
                params=tuple(params),
                flow_positions=flow_idxs,
                produces_grid=_produces_grid(ret),
                return_atoms=_expand_atoms_cached(ret),
                accepts_grid=bool(flow_idxs),
                returns_callable=returns_callable,
            )
        )
    return metas


_FN_META: list[_FnMeta] = _collect_fn_meta()
_ALL_IDX: list[int] = list(range(len(_FN_META)))

# Precompute simple cost model to downweight heavy operators during sampling
_HEAVY_NAMES = {
    "objects",
    "frontiers",
    "occurrences",
    "compress",
    "upscale",
    "downscale",
    "hsplit",
    "vsplit",
}
_COST: dict[int, int] = {i: (3 if _FN_META[i].name in _HEAVY_NAMES else 1) for i in _ALL_IDX}


def _weighted_shuffle(ids: list[int], rng) -> list[int]:
    """Sample without replacement using weights inversely proportional to cost."""
    if not ids:
        return ids
    weights = [1.0 / max(1, _COST[i]) for i in ids]
    # Efraimidis–Spirakis
    keys = [rng.random() ** (1.0 / w) for w in weights]
    return [i for _, i in sorted(zip(keys, ids, strict=False), reverse=True)]


# ---------- Random path sampling (multi-arg) ----------


@dataclass(frozen=True, slots=True)
class _ParamSrc:
    # Exactly one of (var_name) or (const_name,const_val) is set, unless 'omit' is True
    var_name: str | None
    const_name: str | None
    const_val: Any | None
    omit: bool = False  # True => don't pass this parameter at all


def _render_call(meta: _FnMeta, srcs: tuple[_ParamSrc, ...]) -> str:
    """Render a python call, using keywords if any parameter is omitted or any earlier param is omitted."""
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


@dataclass(frozen=True, slots=True)
class _StepPlan:
    idx: int  # index into _FN_META
    out_var: str  # variable defined by this step (e.g., "x3")
    param_srcs: tuple[_ParamSrc, ...]  # one per passed positional parameter (var or const)


def _biased_length(rng, min_len: int, max_len: int) -> int:
    """Favor longer paths softly."""
    lengths = list(range(min_len, max_len + 1))
    weights = list(range(1, len(lengths) + 1))
    return rng.choices(lengths, weights=weights, k=1)[0]


@dataclass(frozen=True, slots=True)
class _Binding:
    # What the code will show:
    src: _ParamSrc
    # What we used in the dry-run to validate the step:
    value: Any


# ---------- (1) Variable window ----------


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


# ---------- Candidate caching for parameters (perf) ----------


@lru_cache(maxsize=4096)
def _callable_arity(fn) -> int | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    req = 0
    for p in sig.parameters.values():
        if (
            p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect._empty
        ):
            req += 1
    return req


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


_UNARY_CALLABLE_CONSTS, _MULTIARG_CALLABLE_CONSTS = [], []
for cname, cval in _CALLABLE_CONSTS:
    ar = _callable_arity(cval)
    if ar == 1:
        _UNARY_CALLABLE_CONSTS.append((cname, cval))
    elif ar and ar > 1:
        _MULTIARG_CALLABLE_CONSTS.append((cname, cval))

_KEYLIKE_PARAM_NAMES = {"compfunc", "condition", "key", "predicate", "proj", "fn"}

# desired callable arities per (fn_name, param_index)
# (if param names differ, index is safest)
_DESIRED_ARITY: dict[str, dict[int, int]] = {
    # returns unary: outer(inner(x))
    "compose": {0: 1, 1: 1},
    # h(g(f(x)))
    "chain": {0: 1, 1: 1, 2: 1},
    # outer(a(x), b(x))  => outer must be binary, a & b unary
    "fork": {0: 2, 1: 1, 2: 1},
    # power(function, n) => function is unary
    "power": {0: 1},
    # also help the consumers:
    "apply": {0: 1},
    "mapply": {0: 1},
    "rapply": {},  # its first arg is a container of callables
    "papply": {0: 2},
    "mpapply": {0: 2},
    "prapply": {0: 2},
    "order": {1: 1},
    "valmax": {1: 1},
    "valmin": {1: 1},
    "sfilter": {1: 1},
}


@lru_cache(maxsize=32768)
def _cached_param_candidates(
    fn_name: str,
    p_index: int,
    atoms_key: frozenset[str],
    env_sig: tuple[tuple[str, bool, int | None], ...],
    want_callable: bool,
) -> tuple[_ParamSrc, ...]:
    cands: list[_ParamSrc] = []
    desired = _DESIRED_ARITY.get(fn_name, {}).get(p_index)

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
        if desired == 1:
            pool = _UNARY_CALLABLE_CONSTS  # build these once using _callable_arity
        elif desired == 2:
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


# ---------- (2) Role templates (soft bias) ----------

# Roles: 'var_callable', 'const_callable', 'var_non_callable', 'var_tuple'
_ROLE_TEMPLATES: dict[str, list[tuple[str, ...]]] = {
    "compose": [("var_callable", "var_callable")],
    "chain": [("var_callable", "var_callable", "var_callable")],
    "power": [("var_callable", "var_non_callable")],  # second is int exponent
    "lbind": [("const_callable", "var_non_callable")],
    "rbind": [("const_callable", "var_non_callable")],
    "fork": [("var_callable", "var_callable", "var_callable")],
    "apply": [("const_callable", "var_non_callable")],
    "order": [("var_non_callable", "const_callable")],
    "papply": [("const_callable", "var_tuple", "var_tuple")],
    "mpapply": [("const_callable", "var_tuple", "var_tuple")],
    "mapply": [("const_callable", "var_non_callable")],
}


def _reorder_candidates_by_role(
    meta: _FnMeta, cand_lists: list[list[_Binding]], env_vals: dict[str, Any]
) -> None:
    """Stable-reorder candidate lists in-place to match preferred role templates when available."""
    tmpl_list = _ROLE_TEMPLATES.get(meta.name)
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


# ---------- Feasibility prefilter (perf) ----------


def _env_has_callable(env_vals: dict[str, Any]) -> bool:
    return any(callable(v) for v in env_vals.values())


def _env_has_non_callable(env_vals: dict[str, Any]) -> bool:
    return any(not callable(v) for v in env_vals.values())


def _can_satisfy(meta: _FnMeta, env_vals: dict[str, Any]) -> bool:
    """Cheap static check: is it possible to satisfy required params by env or constants (incl. callables)."""
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


# ---------- Grid size knobs ----------


def _rand_small_grid():
    """Use a smaller grid for planning to keep intermediates tiny; fall back if generator doesn't accept sizing."""
    try:
        return rand_grid(h=6, w=6)  # if your generator supports these kwargs
    except TypeError:
        return rand_grid()


# ---------- Optional call memoization for planning (perf) ----------


def _is_hashable(x: Any) -> bool:
    try:
        hash(x)
        return True
    except TypeError:
        return False


def _make_hashable(x: Any) -> Any:
    # minimal: tuples and frozensets for common containers
    if isinstance(x, list):
        return tuple(_make_hashable(e) for e in x)
    if isinstance(x, set):
        return frozenset(_make_hashable(e) for e in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in x.items()))
    if isinstance(x, tuple):
        return tuple(_make_hashable(e) for e in x)
    return x


@lru_cache(maxsize=4096)
def _call_memo(fn_name: str, args: tuple) -> Any:
    fn = getattr(dsl, fn_name)
    return fn(*args)


def _call_with_optional_memo(fn_name, args_list):
    try:
        key = tuple(_make_hashable(a) if not _is_hashable(a) else a for a in args_list)
        return _call_memo(fn_name, key)
    except Exception:
        return getattr(dsl, fn_name)(*args_list)


# ---------- (3) Early shape/size guards (planning only) ----------


def _too_big(v: Any, max_cells: int = 64, max_items: int = 256) -> bool:
    """Return True when an intermediate is deemed too large for planning."""
    # grid
    if isinstance(v, tuple) and v and isinstance(v[0], tuple):
        h = len(v)
        w = len(v[0]) if h else 0
        return h * w > max_cells
    # objects/indices as frozenset
    if isinstance(v, frozenset):
        return len(v) > max_items
    # tuple blobs
    if isinstance(v, tuple):
        return len(v) > max_items
    return False


# ---------- (4) Heavy-op budget per path ----------


def _is_heavy_meta(mi: int) -> bool:
    return _FN_META[mi].name in _HEAVY_NAMES


def _meta_wants_callable(mi: int) -> bool:
    return any(_is_callable_anno(sp.anno) for sp in _FN_META[mi].params)


# ---------- (5) Negative cache (meta, env_sig) ----------

# local to a path sampling attempt (reinitialized per attempt)


def _try_bind_and_run(
    meta: _FnMeta,
    env_vals: dict[str, Any],
    rng,
    max_trials: int = 8,
    early_trials: int = 3,
    require_grid_from_I: bool = False,  # enforce x1 consumes I as Grid
    max_intermediate_cells: int = 64,
    max_intermediate_items: int = 256,
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
        atoms_key = _param_atoms(sp.anno)  # frozenset[str], hashable
        param_srcs = list(_cached_param_candidates(meta.name, i, atoms_key, env_sig, want_callable))

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

    # Try randomized attempts (early few, then remainder if needed)
    def _attempts(limit: int):
        for _ in range(limit):
            picks: list[_Binding] = []
            used_var = False
            for cands in cand_lists:
                var_cands = [b for b in cands if b.src.var_name is not None and not b.src.omit]
                b = (
                    rng.choice(var_cands)
                    if var_cands and rng.random() < 0.75
                    else rng.choice(cands)
                )
                if b.src.var_name is not None:
                    used_var = True
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
                        bs.src.var_name is not None
                        and _is_grid_structural(env_vals[bs.src.var_name])
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
                        bs.src.var_name is not None
                        and _is_grid_structural(env_vals[bs.src.var_name])
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


def _sample_path_plans(
    rng,
    max_len: int,
    min_len: int = 3,
    path_budget: int = 128,
    planning_shape: tuple[int, int] = (6, 6),
    heavy_cap: int = 2,
    max_intermediate_cells: int = 64,
    max_intermediate_items: int = 256,
    max_paths: int = 24,
) -> list[list[_StepPlan]]:
    """Build plans by *executing steps during sampling* on a single dry-run input.

    The last chosen function must return a Grid (verified by return typing OR by runtime check).
    Enforces: x1 consumes I (Grid), xLast returns Grid. Middle steps can be anything.
    """
    paths: list[list[_StepPlan]] = []
    if not _FN_META:
        return paths

    for _ in range(path_budget):
        if len(paths) >= max_paths:
            break
        # fresh dry-run environment
        dry_env: dict[str, Any] = {}
        ph, pw = planning_shape
        if max_len >= 80:
            ph = min(ph, 4)
            pw = min(pw, 4)
        try:
            dry_env["I"] = rand_grid(h=ph, w=pw)
        except TypeError:
            dry_env["I"] = _rand_small_grid()
        # Seed one callable so higher-order primitives can chain immediately
        dry_env["F0"] = dsl.identity

        # choose target length
        L = _biased_length(rng, min_len=min_len, max_len=max_len)
        plan: list[_StepPlan] = []

        ok = True
        ever_used_grid = False
        have_callable_var = any(callable(v) for k, v in dry_env.items() if k != "I")

        # (5) negative cache for this attempt
        failed_meta_env: set[tuple[int, tuple[tuple[str, bool], ...]]] = set()
        heavy_used = 0

        for step_no in range(1, L):
            # Prefilter by feasibility
            feasible = [i for i in _ALL_IDX if _can_satisfy(_FN_META[i], dry_env)]
            if not feasible:
                ok = False
                break

            # x1 MUST consume a Grid (from I)
            if step_no == 1:
                feasible = [i for i in feasible if _FN_META[i].accepts_grid]
                if not feasible:
                    ok = False
                    break

            # Heavy-op budget
            if heavy_used >= heavy_cap:
                feasible = [i for i in feasible if not _is_heavy_meta(i)]
                if not feasible:
                    ok = False
                    break

            if have_callable_var:
                # duplicate callable-consuming metas to increase their sampling weight
                boosted = []
                for i in feasible:
                    boosted.append(i)
                    if _meta_wants_callable(i):
                        boosted.append(i)  # simple 2x weight
                metas_order = _weighted_shuffle(boosted, rng)
            else:
                metas_order = _weighted_shuffle(feasible, rng)

            # Boost callable-producing metas early if none exist
            if not have_callable_var and step_no <= 3:
                boosted = []
                for i in feasible:
                    boosted.append(i)
                    if _FN_META[i].returns_callable:
                        boosted.append(i)
                metas_order = _weighted_shuffle(boosted, rng)
            else:
                metas_order = _weighted_shuffle(feasible, rng)

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
                )
                if res is None:
                    failed_meta_env.add((mi, env_sig))
                    continue
                param_srcs, ret, used_grid, _used_I_grid = res
                picked = (mi, param_srcs, ret)
                ever_used_grid = ever_used_grid or used_grid
                if _FN_META[mi].returns_callable:
                    have_callable_var = True
                if _is_heavy_meta(mi):
                    heavy_used += 1
                break

            if picked is None:
                ok = False
                break

            mi, param_srcs, ret_val = picked
            out_name = f"x{step_no}"
            plan.append(_StepPlan(idx=mi, out_var=out_name, param_srcs=param_srcs))
            dry_env[out_name] = ret_val

        # must have at least one grid usage (ensured by x1 rule)
        if not ok or not ever_used_grid:
            continue

        # final step: must return Grid. Prefer metas annotated as producing Grid; but accept
        # runtime-validated returns that are structurally a Grid (tuple-of-tuples-of-ints).
        final_cands = [i for i, m in enumerate(_FN_META) if m.produces_grid] or _ALL_IDX
        final_cands = [i for i in final_cands if _can_satisfy(_FN_META[i], dry_env)]
        if heavy_used >= heavy_cap:
            final_cands = [i for i in final_cands if not _is_heavy_meta(i)]
        if not final_cands:
            continue
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


# ---------- Quality checkers ----------


def _is_empty_grid(grid: Any) -> bool:
    """Check if a grid is empty (all zeros)."""
    if not isinstance(grid, tuple) or not grid:
        return False
    if not isinstance(grid[0], tuple):
        return False
    return all(all(cell == 0 for cell in row) for row in grid)


def _grids_are_similar(grid1: Any, grid2: Any) -> bool:
    """Check if two grids are identical."""
    return grid1 == grid2


def _has_invalid_values(grid: Any) -> bool:
    """Check if grid contains values outside [0-9] range."""
    if not isinstance(grid, tuple) or not grid:
        return False
    if not isinstance(grid[0], tuple):
        return False
    for row in grid:
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell > 9:
                return True
    return False


def _check_shots_quality(shots: list[dict[str, Any]]) -> tuple[bool, str]:
    """Check if shots meet quality criteria.

    Returns
    -------
        (is_high_quality, reason_if_low_quality)

    """
    if not shots:
        return False, "no_shots"

    # Extract output grids
    outputs = [shot.get("output") for shot in shots]

    # Check 1: All empty outputs
    all_empty = all(_is_empty_grid(out) for out in outputs)
    if all_empty:
        return False, "all_empty_outputs"

    # Check 2: All outputs similar (identical)
    if len(outputs) >= 2:
        first_output = outputs[0]
        all_similar = all(_grids_are_similar(first_output, out) for out in outputs[1:])
        if all_similar:
            return False, "all_outputs_identical"

    # Check 3: Any output has invalid values (not in [0-9])
    has_invalid = any(_has_invalid_values(out) for out in outputs)
    if has_invalid:
        return False, "invalid_values_outside_0_9"

    return True, ""


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

    # 1) Sample candidate path plans
    plans = _sample_path_plans(
        rng,
        max_len=max_lines,
        min_len=3,
        path_budget=path_budget,
        planning_shape=planning_shape,
        heavy_cap=2,
        max_intermediate_cells=planning_shape[0]
        * planning_shape[1],  # respect chosen planning size
        max_intermediate_items=256,
        max_paths=24,
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
    _expand_atoms_cached.cache_clear()
    _cached_param_candidates.cache_clear()
    _call_memo.cache_clear()
