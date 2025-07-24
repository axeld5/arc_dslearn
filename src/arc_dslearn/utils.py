"""Utility functions for JSON handling."""

from typing import Any, Iterable, Iterator


class OrderedFrozenSet(frozenset[Any]):
    """A frozenset that preserves the order of its elements."""

    def __new__(cls, iterable: Iterable[Any]) -> "OrderedFrozenSet":
        """Create a new OrderedFrozenSet with the given iterable."""
        data = list(iterable)
        obj = super().__new__(cls, data)
        # Store order as a private attribute without using __slots__
        object.__setattr__(obj, "_OrderedFrozenSet__order", tuple(data))
        return obj

    def __iter__(self) -> Iterator[Any]:
        """Iterate in the original order."""
        return iter(object.__getattribute__(self, "_OrderedFrozenSet__order"))

    def __repr__(self) -> str:
        """Return a string representation that shows the preserved order."""
        order = object.__getattribute__(self, "_OrderedFrozenSet__order")
        return f"OrderedFrozenSet({list(order)!r})"


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
