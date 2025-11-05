"""Check which DSL functions are used and unused in the generated data."""

import inspect
import json
from collections import Counter
from pathlib import Path


def get_all_dsl_functions():
    """Extract all function names from the DSL module."""
    from src.arc_dslearn.arc_dsl import dsl

    functions = []
    for name, obj in inspect.getmembers(dsl):
        if inspect.isfunction(obj) and not name.startswith("_"):
            functions.append(name)
    return sorted(functions)


def get_used_functions_from_jsonl(jsonl_path):
    """Extract all used function names from the JSONL file."""
    used_functions = []
    total_samples = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_samples += 1
                sample = json.loads(line)
                # Extract function names from the "name" field (pipe-separated)
                if "name" in sample:
                    functions = sample["name"].split("|")
                    used_functions.extend(functions)

    return used_functions, total_samples


def analyze_function_usage(jsonl_path):
    """Analyze which DSL functions are used and unused."""
    print(f"\n{'=' * 80}")
    print("DSL Function Usage Analysis")
    print(f"{'=' * 80}\n")

    # Get all DSL functions
    all_dsl_functions = get_all_dsl_functions()
    print(f"Total DSL functions available: {len(all_dsl_functions)}")

    # Get used functions from JSONL
    used_functions_list, total_samples = get_used_functions_from_jsonl(jsonl_path)
    print(f"Total samples analyzed: {total_samples}")
    print(f"Total function usages: {len(used_functions_list)}\n")

    # Count function usage
    function_counts = Counter(used_functions_list)
    used_functions_set = set(used_functions_list)

    # Find unused functions
    unused_functions = set(all_dsl_functions) - used_functions_set

    print(f"{'=' * 80}")
    print(f"USED FUNCTIONS: {len(used_functions_set)}")
    print(f"{'=' * 80}\n")

    # Sort by usage count (descending) and then alphabetically
    sorted_used = sorted(function_counts.items(), key=lambda x: (-x[1], x[0]))

    for func, count in sorted_used:
        percentage = (count / len(used_functions_list)) * 100
        print(f"  {func:20s} - {count:4d} times ({percentage:5.2f}%)")

    print(f"\n{'=' * 80}")
    print(f"UNUSED FUNCTIONS: {len(unused_functions)}")
    print(f"{'=' * 80}\n")

    if unused_functions:
        sorted_unused = sorted(unused_functions)
        # Print in columns for better readability
        cols = 4
        for i in range(0, len(sorted_unused), cols):
            row = sorted_unused[i : i + cols]
            print("  " + "  ".join(f"{func:20s}" for func in row))
    else:
        print("  All DSL functions are being used!")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total DSL functions:     {len(all_dsl_functions)}")
    print(
        f"  Used functions:          {len(used_functions_set)} ({len(used_functions_set) / len(all_dsl_functions) * 100:.1f}%)"
    )
    print(
        f"  Unused functions:        {len(unused_functions)} ({len(unused_functions) / len(all_dsl_functions) * 100:.1f}%)"
    )
    print(f"  Total samples:           {total_samples}")
    print(f"  Avg functions per sample: {len(used_functions_list) / total_samples:.2f}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Default JSONL file path (same as in pilot.py)
    jsonl_path = Path(__file__).parent.parent.parent.parent / "data" / "blocks_200.jsonl"

    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        print("Please run the pilot.py script first to generate the data.")
    else:
        analyze_function_usage(jsonl_path)
