"""Data processing utilities for training data generation."""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path
from typing import Any, Callable, Tuple, get_origin

import src.arc_dslearn.arc_dsl.dsl as dsl
from src.arc_dslearn.utils import from_jsonable


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


def _should_skip_function(func: Callable[..., Any]) -> bool:
    """Check if a function should be skipped during testing."""
    sig = inspect.signature(func)

    # skip functions that *return* a Callable
    ret_anno = sig.return_annotation
    if ret_anno is not inspect._empty and (
        ret_anno is Callable or get_origin(ret_anno) is Callable
    ):
        return True

    # skip functions that *take* a Callable as parameter
    has_callable_param = any(
        param.annotation is Callable or get_origin(param.annotation) is Callable
        for param in sig.parameters.values()
        if param.annotation is not inspect._empty
    )
    return has_callable_param


def _get_valid_dsl_functions() -> dict[str, Callable]:
    """Get all valid DSL functions that can be tested."""
    valid_functions = {}
    for name, func in inspect.getmembers(dsl, inspect.isfunction):
        if name.startswith("_"):
            continue
        if _should_skip_function(func):
            continue
        valid_functions[name] = func
    return valid_functions


def _can_function_solve_io(func: Callable, inputs: dict[str, Any], expected_output: Any) -> bool:
    """Test if a function can solve the given I/O pair."""
    try:
        # Get function signature to determine how to call it
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Handle single parameter functions (most common)
        if len(params) == 1:
            param_name = params[0]
            if param_name in inputs:
                input_value = inputs[param_name]
            else:
                # Try common parameter names and fallback to first value
                for key in ["x", "grid", "patch", "obj", "object", "piece", "container", "a"]:
                    if key in inputs:
                        input_value = inputs[key]
                        break
                else:
                    # Use the first (or only) input value
                    input_value = next(iter(inputs.values()))

            result = func(input_value)
            return result == expected_output

        # Handle multi-parameter functions
        elif len(params) == len(inputs):
            # Try to match parameter names
            if set(params) <= set(inputs.keys()):
                kwargs = {param: inputs[param] for param in params}
                result = func(**kwargs)
                return result == expected_output
            else:
                # Try positional arguments in order
                args = list(inputs.values())
                result = func(*args)
                return result == expected_output

        # Cannot determine how to call this function
        return False

    except Exception:
        # Function failed on this input
        return False


def remove_answer_overlap(
    input_file: str, output_file: str | None = None, min_solutions: int = 1, max_solutions: int = 1
) -> Tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove training blocks where multiple DSL functions can solve ALL shots for that block.

    This identifies true ambiguity where multiple functions could be the correct answer.

    Args:
    ----
        input_file: Path to the JSON file containing training blocks
        output_file: Path to save the filtered data (optional, if None, overwrites input_file)
        min_solutions: Minimum number of functions that should solve ALL shots in a block
        max_solutions: Maximum number of functions that should solve ALL shots in a block

    Returns:
    -------
        Tuple of (filtered_data, overlap_stats)

    """
    if output_file is None:
        output_file = input_file

    with open(input_file, "r") as f:
        data = json.load(f)

    # Get all valid DSL functions
    print("Loading DSL functions...")
    dsl_functions = _get_valid_dsl_functions()
    print(f"Found {len(dsl_functions)} valid DSL functions to test")

    # Track statistics
    total_blocks = 0
    blocks_removed = 0
    ambiguous_blocks = []
    unsolvable_blocks = []
    functions_affected = set()

    filtered_data = []

    for block in data:
        func_name = block["name"]
        total_blocks += 1

        if "shots" not in block or not block["shots"]:
            filtered_data.append(block)
            continue

        # Convert all shots from JSON format back to original
        shots_data = []
        for shot in block["shots"]:
            inputs = from_jsonable(shot["inputs"])
            expected_output = from_jsonable(shot["output"])
            shots_data.append((inputs, expected_output))

        # Test which DSL functions can solve ALL shots in this block
        solving_functions = []
        for dsl_name, dsl_func in dsl_functions.items():
            can_solve_all = True
            for inputs, expected_output in shots_data:
                if not _can_function_solve_io(dsl_func, inputs, expected_output):
                    can_solve_all = False
                    break

            if can_solve_all:
                solving_functions.append(dsl_name)

        num_solutions = len(solving_functions)

        # Check if this block meets our criteria
        if min_solutions <= num_solutions <= max_solutions:
            filtered_data.append(block)
        else:
            blocks_removed += 1
            functions_affected.add(func_name)

            if num_solutions == 0:
                unsolvable_blocks.append({
                    "function": func_name,
                    "shots": shots_data[:3],  # Include first 3 shots for debugging
                    "solving_functions": solving_functions,
                })
            else:
                ambiguous_blocks.append({
                    "function": func_name,
                    "shots": shots_data[:3],  # Include first 3 shots for debugging
                    "solving_functions": solving_functions,
                    "num_solutions": num_solutions,
                })

    # Save filtered data
    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=2)

    # Prepare statistics
    overlap_stats = {
        "total_blocks": total_blocks,
        "blocks_removed": blocks_removed,
        "blocks_kept": total_blocks - blocks_removed,
        "ambiguous_blocks": len(ambiguous_blocks),
        "unsolvable_blocks": len(unsolvable_blocks),
        "functions_affected": list(functions_affected),
        "blocks_before": len(data),
        "blocks_after": len(filtered_data),
        "criteria": f"{min_solutions}-{max_solutions} solutions for ALL shots",
        "dsl_functions_tested": len(dsl_functions),
        # Include details for debugging (limited to prevent huge output)
        "ambiguous_examples": ambiguous_blocks[:5],
        "unsolvable_examples": unsolvable_blocks[:5],
    }

    print(f"✓ Removed {blocks_removed}/{total_blocks} blocks with ambiguous or unsolvable patterns")
    print(
        f"✓ Found {len(ambiguous_blocks)} ambiguous blocks (multiple DSL functions can solve ALL shots)"
    )
    print(
        f"✓ Found {len(unsolvable_blocks)} unsolvable blocks (no DSL function can solve ALL shots)"
    )
    print(f"✓ Blocks: {len(data)} → {len(filtered_data)}")
    print(f"✓ Functions affected: {len(functions_affected)}")
    if functions_affected:
        print(
            f"✓ Affected functions: {', '.join(sorted(functions_affected)[:10])}{'...' if len(functions_affected) > 10 else ''}"
        )

    return filtered_data, overlap_stats


def preprocess_json_file(input_file: str, output_file: str) -> None:
    """Prepare data for dataset loading without double JSON encoding."""
    with open(input_file, "r") as f:
        data = json.load(f)
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
