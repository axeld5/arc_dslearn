"""Utility functions for JSON handling."""

from typing import Any

from typing import Any, Iterable

class OrderedFrozenSet(frozenset):
    __slots__ = ('_order',)

    def __new__(cls, iterable: Iterable[Any]):
        data = list(iterable)
        obj = super().__new__(cls, data)
        obj._order = tuple(data)          # preserve insertion / JSON order
        return obj

    def __iter__(self):
        # iterate in the original order
        return iter(self._order)

    def __repr__(self):
        # nice, stable repr that shows the preserved order
        return f"OrderedFrozenSet({list(self._order)!r})"


def from_jsonable(x: Any) -> Any:
    """Convert a JSON-able value to a Python value."""
    if isinstance(x, dict):
        if "__frozenset__" in x:
            return OrderedFrozenSet(from_jsonable(v) for v in x["__frozenset__"])
        if "__tuple__" in x:
            return tuple(from_jsonable(v) for v in x["__tuple__"])
        if "__set__" in x:
            return {from_jsonable(v) for v in x["__set__"]}
        if "__str__" in x:
            return x["__str__"]
        return {k: from_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [from_jsonable(v) for v in x]
    return x
