"""Data processing utilities for training data generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Tuple


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
    """Convert inputs and outputs to JSON strings for consistency."""
    with open(input_file, "r") as f:
        data = json.load(f)

    # Convert inputs and outputs to JSON strings for consistency
    for record in data:
        if record["shots"]:
            for shot in record["shots"]:
                shot["inputs"] = json.dumps(shot["inputs"])
                shot["output"] = json.dumps(shot["output"])

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
