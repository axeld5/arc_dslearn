"""Tests for DSL function representation coverage.

This module verifies that all relevant DSL functions are represented
in the generated training data.
"""

import inspect
import os
import sys

# Add the parent directory to the path so we can import arc_dslearn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.arc_dslearn.arc_dsl.dsl as dsl
from src.arc_dslearn.data_gene.block_generation import should_skip_function
from src.arc_dslearn.data_gene.pilot import main


def test_dsl_functions_available():
    """Test that DSL functions are available and filterable."""
    all_dsl_functions = [
        name
        for name, func in inspect.getmembers(dsl, inspect.isfunction)
        if not name.startswith("_")
    ]
    assert len(all_dsl_functions) > 0, "Should have DSL functions available"

    # Filter to functions that should be included (not skipped)
    expected_functions = []
    for name, func in inspect.getmembers(dsl, inspect.isfunction):
        if not name.startswith("_") and not should_skip_function(func):
            expected_functions.append(name)

    assert len(expected_functions) > 0, "Should have some non-skipped DSL functions"


def test_function_coverage_in_generated_blocks():
    """Test DSL function coverage in generated blocks."""
    training_blocks = main(generation_seed=42)
    assert len(training_blocks) > 0, "Should generate training blocks"

    # Get expected functions
    expected_functions = []
    for name, func in inspect.getmembers(dsl, inspect.isfunction):
        if not name.startswith("_") and not should_skip_function(func):
            expected_functions.append(name)

    # Extract function names from generated code
    represented_functions = set()
    for block in training_blocks:
        completion = block.get("assistant_prompt", "")
        # Look for function calls in the completion
        for func_name in expected_functions:
            if func_name in completion:
                represented_functions.add(func_name)

    # Test that we have reasonable coverage of DSL functions
    coverage_ratio = (
        len(represented_functions) / len(expected_functions) if expected_functions else 0
    )
    assert (
        coverage_ratio > 0.8
    ), f"Expected >80% function coverage, got {coverage_ratio:.1%} ({len(represented_functions)}/{len(expected_functions)})"


def test_valid_code_contexts():
    """Test that represented functions appear in valid contexts."""
    training_blocks = main(generation_seed=42)
    assert len(training_blocks) > 0, "Should generate training blocks"

    # Get expected functions
    expected_functions = []
    for name, func in inspect.getmembers(dsl, inspect.isfunction):
        if not name.startswith("_") and not should_skip_function(func):
            expected_functions.append(name)

    # Extract function names from generated code
    represented_functions = set()
    for block in training_blocks:
        completion = block.get("assistant_prompt", "")
        for func_name in expected_functions:
            if func_name in completion:
                represented_functions.add(func_name)

    valid_contexts = 0
    total_contexts = 0
    for block in training_blocks[:10]:  # Check first 10 blocks
        completion = block.get("assistant_prompt", "")
        if "def solve(" in completion and any(f in completion for f in represented_functions):
            total_contexts += 1
            # Basic check: completion should have proper Python structure
            if completion.count("def solve(") == 1 and "return" in completion:
                valid_contexts += 1

    if total_contexts > 0:
        context_validity = valid_contexts / total_contexts
        assert (
            context_validity > 0.9
        ), f"Expected >90% valid contexts, got {context_validity:.1%} ({valid_contexts}/{total_contexts})"
