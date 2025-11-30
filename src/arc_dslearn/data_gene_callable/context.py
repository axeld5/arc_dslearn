"""Planning mode context and timeout helpers for path exploration."""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager

# ---------- Planning mode context ----------

_PLANNING = contextvars.ContextVar("_PLANNING", default=False)


@contextmanager
def planning_phase():
    """Context manager to mark planning/exploration phase (vs. generation phase)."""
    token = _PLANNING.set(True)
    try:
        yield
    finally:
        _PLANNING.reset(token)


def _in_planning() -> bool:
    """Return True if currently in planning phase."""
    return _PLANNING.get()


# ---------- Timeout helper for path exploration ----------


class TimeoutException(Exception):
    """Raised when path exploration exceeds time limit."""

    pass


# Global deadline for cooperative timeout checking within functions
_deadline = contextvars.ContextVar("_deadline", default=None)


def _check_deadline():
    """Check if we've exceeded the deadline. Raise TimeoutException if so.

    Call this periodically in long-running functions for cooperative timeout.
    """
    deadline = _deadline.get()
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutException("Operation exceeded deadline")


def _set_deadline(seconds: float):
    """Set a deadline for cooperative timeout checking."""
    return _deadline.set(time.monotonic() + seconds)


def _clear_deadline(token):
    """Clear the deadline."""
    _deadline.reset(token)


def _run_with_timeout(func, timeout_seconds: float, *args, **kwargs):
    """Run a function with a timeout. Returns (result, timed_out).

    If timed_out is True, result is None and the function exceeded the timeout.

    Uses cooperative timeout via deadline checking within the function.
    No threads involved - simpler and more reliable on Windows.
    """
    token = _set_deadline(timeout_seconds)
    try:
        result = func(*args, **kwargs)
        return result, False
    except TimeoutException:
        return None, True
    except Exception:
        raise
    finally:
        _clear_deadline(token)
