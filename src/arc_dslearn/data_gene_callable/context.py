"""Planning mode context and timeout helpers for path exploration."""

from __future__ import annotations

import contextvars
import threading
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


def _run_with_timeout(func, timeout_seconds: float, *args, **kwargs):
    """Run a function with a timeout. Returns (result, timed_out).

    If timed_out is True, result is None and the function exceeded the timeout.
    """
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread is still running, timeout occurred
        return None, True

    if exception[0] is not None:
        raise exception[0]

    return result[0], False
