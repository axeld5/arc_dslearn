"""Data processing utilities for training data generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Tuple


def compact_format(x: Any) -> str:
    """Create ultra-compact string representation to minimize tokens."""
    if isinstance(x, (list, tuple)) and len(x) > 0:
        # Check if it's a grid (nested lists/tuples of integers)
        if isinstance(x[0], (list, tuple)) and all(isinstance(row, (list, tuple)) for row in x):
            # Grid format: use semicolon for rows, comma for columns
            return ";".join(",".join(map(str, row)) for row in x)
        # Regular list/tuple: compact comma-separated
        elif all(isinstance(item, int) for item in x):
            return ",".join(map(str, x))
        else:
            # Mixed types - use minimal brackets
            return "[" + ",".join(compact_format(item) for item in x) + "]"
    elif isinstance(x, (frozenset, set)):
        items = list(x)
        return "{" + ",".join(compact_format(item) for item in items) + "}"
    elif isinstance(x, dict):
        # Use minimal dict format
        items = [f"{k}:{compact_format(v)}" for k, v in x.items()]
        return "{" + ",".join(items) + "}"
    elif isinstance(x, (int, float, bool)) or x is None:
        return str(x)
    else:
        return str(x)


def to_jsonable(x: Any) -> Any:
    """Convert a value to a JSON-able format."""
    if isinstance(x, frozenset):
        return {"__frozenset__": [to_jsonable(v) for v in x]}
    if isinstance(x, tuple):
        return {"__tuple__": [to_jsonable(v) for v in x]}
    if isinstance(x, set):
        return {"__set__": [to_jsonable(v) for v in x]}
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, int, float, str, bool)) or x is None:
        return x
    return {"__str__": str(x)}  # last resort


def preprocess_json_file(input_file: str, output_file: str) -> None:
    """Prepare data for dataset loading without double JSON encoding."""
    with open(input_file, "r") as f:
        data = json.load(f)

    # Keep inputs and outputs as objects - no need to stringify them
    # The double JSON encoding was causing massive token bloat
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def create_train_eval_split(
    src_file: str = "train_set.json",
    train_out: str = "train_split.json",
    eval_out: str = "eval_split.json",
    split_seed: int = 42,
    eval_frac: float = 0.10,
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the training data into train and eval sets."""
    src_path = Path(src_file)
    train_path = Path(train_out)
    eval_path = Path(eval_out)

    data = json.loads(src_path.read_text())
    random.Random(split_seed).shuffle(data)

    split = int(len(data) * (1 - eval_frac))
    train_data, eval_data = data[:split], data[split:]

    train_path.write_text(json.dumps(train_data, indent=2))
    eval_path.write_text(json.dumps(eval_data, indent=2))

    print(f"✓ wrote {len(train_data)} train   → {train_path}")
    print(f"✓ wrote {len(eval_data)} eval    → {eval_path}")

    return train_data, eval_data


def prepare_datasets_for_loading() -> None:
    """Preprocess the split files for dataset loading."""
    # Preprocess both train and eval splits, overwriting original files
    preprocess_json_file("train_split.json", "train_split.json")
    preprocess_json_file("eval_split.json", "eval_split.json")

    print("✓ Preprocessed train_split.json")
    print("✓ Preprocessed eval_split.json")
