"""Planning-aware memoization for DSL function calls."""

from __future__ import annotations

from typing import Any

import src.arc_dslearn.arc_dsl.dsl as dsl
from src.arc_dslearn.data_gene_callable.context import _check_deadline, _in_planning
from src.arc_dslearn.data_gene_callable.generators import rand_grid

# ---------- Grid sizing helpers ----------


def _is_grid_like(x: Any) -> bool:
    """Check if value is a grid (tuple of tuples)."""
    return isinstance(x, tuple) and x and isinstance(x[0], tuple)


def _cells(x: Any) -> int:
    """Return number of cells in a grid-like structure."""
    if _is_grid_like(x):
        return len(x) * (len(x[0]) if x else 0)
    return 0


def _make_small_grid_pool(
    rng, k: int = 5, min_dim: int = 4, max_dim: int = 6
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Create a small pool of fixed grids for planning phase (increases memo hits).

    Each grid has random dimensions between min_dim and max_dim (both height and width).
    """
    pool = []
    for _ in range(k):
        h = rng.randint(min_dim, max_dim)
        w = rng.randint(min_dim, max_dim)
        try:
            pool.append(rand_grid(h=h, w=w))
        except TypeError:
            pool.append(rand_grid())
    return tuple(pool)


def _rand_small_grid():
    """Use a smaller grid for planning to keep intermediates tiny; fall back if generator doesn't accept sizing."""
    try:
        return rand_grid(h=6, w=6)  # if your generator supports these kwargs
    except TypeError:
        return rand_grid()


# ---------- Planning-aware memoization ----------

_SMALL_MEMO: dict[tuple, Any] = {}
_SMALL_MEMO_MAX = 2048  # tiny and self-clearing
_GRID_CELL_CACHE_LIMIT = 64  # ~8x8; grids bigger than this are never cached (in or out)


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


def _hashable_arg(a: Any) -> Any:
    """Convert an argument to a hashable form for caching."""
    return _make_hashable(a) if not _is_hashable(a) else a


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


def _should_cache_call(fn_name: str, args_list: list[Any]) -> bool:
    """Decide if this call should be cached (only in planning mode, only for small grids)."""
    if not _in_planning():
        return False
    # refuse if any arg looks like a grid over the tiny limit
    return all(not (_is_grid_like(a) and _cells(a) > _GRID_CELL_CACHE_LIMIT) for a in args_list)


def _should_cache_result(res: Any) -> bool:
    """Decide if this result should be cached (avoid large grids/containers)."""
    if _is_grid_like(res) and _cells(res) > _GRID_CELL_CACHE_LIMIT:
        return False
    # Also avoid massive containers (reuse existing guard with stricter limits)
    return not _too_big(res, max_cells=_GRID_CELL_CACHE_LIMIT, max_items=128)


def _call_with_optional_memo(fn_name, args_list):
    """Call DSL function with optional memoization in planning mode."""
    # Check deadline before potentially expensive DSL call
    _check_deadline()

    # Generation phase: just call directly
    if not _should_cache_call(fn_name, args_list):
        return getattr(dsl, fn_name)(*args_list)

    key = (fn_name,) + tuple(_hashable_arg(a) for a in args_list)
    if key in _SMALL_MEMO:
        return _SMALL_MEMO[key]

    res = getattr(dsl, fn_name)(*args_list)

    if _should_cache_result(res):
        if len(_SMALL_MEMO) >= _SMALL_MEMO_MAX:
            _SMALL_MEMO.clear()
        _SMALL_MEMO[key] = res

    return res


def clear_planning_memo() -> None:
    """Clear planning memoization cache."""
    _SMALL_MEMO.clear()


# Exports
__all__ = [
    "_is_grid_like",
    "_cells",
    "_make_small_grid_pool",
    "_rand_small_grid",
    "_too_big",
    "_call_with_optional_memo",
    "clear_planning_memo",
]
