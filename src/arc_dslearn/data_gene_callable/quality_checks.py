"""Quality checking for generated training shots."""

from __future__ import annotations

from typing import Any


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


# Exports
__all__ = [
    "_is_empty_grid",
    "_grids_are_similar",
    "_has_invalid_values",
    "_check_shots_quality",
]
