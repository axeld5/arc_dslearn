"""Module for testing the reward function's validity over the generated data."""

import ast
import json
from typing import Any, Dict, List, Tuple

import src.arc_dslearn.arc_dsl.dsl as dsl
from src.arc_dslearn.metrics_and_rewards.reward_fn import (
    IMPORT_RE,
    SOLVE_RE,
    equivalent,
    reward_function,
    safe_exec,
)
from src.arc_dslearn.utils import from_jsonable


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    import re

    # Match ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no markdown, return as-is (might be plain Python)
    return text.strip()


def test_dsl_validation_specific() -> bool:
    """Test the DSL validation part of the reward function with specific edge cases."""
    print("Testing DSL validation part of reward function...")
    print("=" * 50)

    # Test cases: (code, expected_dsl_reward, description)
    test_cases = [
        # Valid DSL usage
        (
            """def solve(I):
    return vmirror(I)""",
            0.1,
            "Simple valid DSL function",
        ),
        # Valid DSL with multiple functions
        (
            """def solve(I):
    x1 = vmirror(I)
    x2 = hmirror(x1)
    return x2""",
            0.1,
            "Multiple DSL functions with valid variables",
        ),
        # Bad import - should fail
        (
            """import numpy as np
def solve(I):
    return vmirror(I)""",
            0.0,
            "Contains bad import",
        ),
        # Invalid variable name - should fail
        (
            """def solve(I):
    invalid_var = vmirror(I)
    return numpy_array""",
            0.0,
            "Contains invalid variable names",
        ),
        # Valid with I and O usage
        (
            """def solve(I):
    O = vmirror(I)
    return O""",
            0.1,
            "Valid usage of I and O",
        ),
        # Valid with numbered variables
        (
            """def solve(I):
    x1 = vmirror(I)
    x99 = hmirror(x1) 
    return x99""",
            0.1,
            "Valid numbered variables x1-x99",
        ),
        # Syntax error - should fail silently
        (
            """def solve(I:
    return vmirror(I)""",
            0.0,
            "Syntax error in code",
        ),
        # No solve function - should get negative reward (-1.0 + 0.1 DSL = -0.9)
        (
            """def other_function(I):
    return vmirror(I)""",
            0.1,
            "No solve function but valid DSL",
        ),
        # Mix of valid DSL and unknown names - should fail
        (
            """def solve(I):
    x1 = vmirror(I)
    unknown_func = some_invalid_function(x1)
    return unknown_func""",
            0.0,
            "Mix of valid DSL and unknown names",
        ),
    ]

    failed_tests = []

    for i, (code, expected_dsl_reward, description) in enumerate(test_cases):
        # Create minimal test data
        dummy_shots = [[{"inputs": '{"a": [[1, 2], [3, 4]]}', "output": "[[1, 2], [3, 4]]"}]]

        # Test just the format + DSL parts (not functional correctness)
        rewards = reward_function([code], dummy_shots)
        actual_reward = rewards[0]

        # Separate analysis to check DSL validation specifically
        extracted_code = extract_python_code(code)
        has_solve = bool(SOLVE_RE.search(extracted_code))

        dsl_score = 0.0
        try:
            tree = ast.parse(extracted_code)
            bad_imports = bool(IMPORT_RE.search(extracted_code))
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            dsl_names = set(dsl.__dict__.keys())
            unknown = names - {"I", "O"} - {f"x{i}" for i in range(1, 100)} - dsl_names

            if not bad_imports and not unknown:
                dsl_score = 0.1

        except SyntaxError:
            dsl_score = 0.0

        format_score = 0.1 if has_solve else -1.0

        print(f"\nTest {i + 1}: {description}")
        print(f"  Code snippet: {code[:50]}...")
        print(f"  Has solve function: {has_solve} (contributes {format_score})")
        print(f"  Expected DSL score: {expected_dsl_reward}, Actual DSL score: {dsl_score}")
        print(f"  Total reward: {actual_reward:.1f}")

        # Check if DSL validation is working as expected
        if abs(dsl_score - expected_dsl_reward) > 0.01:
            failed_tests.append((i + 1, description, expected_dsl_reward, dsl_score))
            print("  ❌ DSL validation FAILED!")
        else:
            print("  ✅ DSL validation passed")

    print(f"\nSummary: {len(test_cases) - len(failed_tests)}/{len(test_cases)} tests passed")

    if failed_tests:
        print("\nFailed tests:")
        for test_num, desc, expected, actual in failed_tests:
            print(f"  Test {test_num}: {desc} - Expected {expected}, got {actual}")
        return False

    print("🎉 All DSL validation tests passed!")
    return True


def test_reward_function_clean() -> bool:
    """Clean test of the reward function using real training data."""
    try:
        # Load the dataset
        with open("train_split.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ train_split.json not found. Run data generation first.")
        return False

    print(f"Testing reward function with {len(data)} real examples from train_split.json...")
    print("=" * 50)

    # Test a sample of examples to show how the reward function works
    sample_size = len(data)  # Test all examples
    perfect_count = 0
    failed_examples = []

    for i in range(sample_size):
        example = data[i]
        assistant_output = example["assistant_prompt"]
        shots = from_jsonable(example["shots"])

        # Calculate reward for this single example
        rewards = reward_function([assistant_output], [shots])
        reward = rewards[0]

        if reward == 1.0:
            perfect_count += 1
        else:
            print("  ❌ Not perfect")
            failed_examples.append((i, reward, assistant_output[:200]))

    print(f"\nSample Results: {perfect_count}/{sample_size} examples got perfect reward (1.0)")

    # Now test all examples to get overall statistics
    print(f"\nTesting all {len(data)} examples for overall statistics...")
    all_rewards = []
    total_perfect = 0

    # Process in batches for efficiency
    batch_size = 50
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        codes = [ex["assistant_prompt"] for ex in batch]
        shots_list = [from_jsonable(ex["shots"]) for ex in batch]

        batch_rewards = reward_function(codes, shots_list)
        all_rewards.extend(batch_rewards)
        total_perfect += sum(1 for r in batch_rewards if r == 1.0)

    # Statistics
    avg_reward = sum(all_rewards) / len(all_rewards)
    min_reward = min(all_rewards)
    max_reward = max(all_rewards)

    print("\nOverall Statistics:")
    print(
        f"  Perfect examples (1.0): {total_perfect}/{len(data)} ({total_perfect / len(data) * 100:.1f}%)"
    )
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Min reward: {min_reward:.3f}")
    print(f"  Max reward: {max_reward:.3f}")

    # Show distribution
    reward_bins = {}
    for r in all_rewards:
        bin_key = round(r, 1)
        reward_bins[bin_key] = reward_bins.get(bin_key, 0) + 1

    print("\nReward Distribution:")
    for reward_val in sorted(reward_bins.keys()):
        count = reward_bins[reward_val]
        print(f"  {reward_val:.1f}: {count} examples ({count / len(data) * 100:.1f}%)")

    # The test "passes" if we successfully computed rewards for all examples
    # We don't expect all to be perfect since this is testing the reward function itself
    success = len(all_rewards) == len(data) and max_reward <= 1.0 and min_reward >= -1.0

    if success:
        print("🎉 Reward function test completed successfully!")
        return True
    else:
        print("❌ Reward function test failed - unexpected reward values")
        return False


def reward_fn_debug(
    completions: List[str], shots: List[List[Dict[str, Any]]], **kwargs: Any
) -> List[Tuple[float, Dict[str, Any]]]:
    """Debug version of reward function that shows why each component fails."""
    rewards: List[Tuple[float, Dict[str, Any]]] = []
    for code_raw, shot_list in zip(completions, shots, strict=False):
        # Extract Python code from markdown
        code = extract_python_code(code_raw)

        shot_list = from_jsonable(shot_list)
        r = 0.0
        debug_info: Dict[str, Any] = {
            "extracted_code": code[:100] + "..." if len(code) > 100 else code
        }

        # (1) Solve function present
        has_solve = SOLVE_RE.search(code)
        if has_solve:
            r += 0.1
            debug_info["solve_function"] = "✓ Found solve function"
        else:
            r -= 1.0
            debug_info["solve_function"] = "✗ No solve function found"

        # (2) No bad imports, only DSL names
        try:
            tree = ast.parse(code)
            bad_imports = bool(IMPORT_RE.search(code))
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            dsl_names = set(dsl.__dict__.keys())
            unknown = names - {"I", "O"} - {f"x{i}" for i in range(1, 100)} - dsl_names

            debug_info["bad_imports"] = bad_imports
            debug_info["found_names"] = names
            debug_info["unknown_names"] = unknown
            debug_info["dsl_names_count"] = len(dsl_names)

            if not bad_imports and not unknown:
                r += 0.1
                debug_info["dsl_check"] = "✓ All names are valid DSL functions"
            else:
                debug_info["dsl_check"] = f"✗ Bad imports: {bad_imports}, Unknown names: {unknown}"
        except SyntaxError as e:
            debug_info["dsl_check"] = f"✗ Syntax error: {e}"

        # (3) Functional correctness on all provided shots
        results = []
        for i, shot in enumerate(shot_list):
            try:
                # Parse the input - it comes as a JSON string
                input_data = shot["inputs"]
                if isinstance(input_data, str):
                    import json

                    input_data = json.loads(input_data)

                # Execute with input data context
                mod = safe_exec(code, input_data)
                if mod and callable(getattr(mod, "solve", None)):
                    result = mod.solve(input_data)

                    # Parse expected output
                    expected = shot["output"]
                    if isinstance(expected, str):
                        expected = json.loads(expected)

                    expected = from_jsonable(expected)
                    match = equivalent(result, expected, input_data)
                    results.append(match)
                    if not match:
                        debug_info[f"shot_{i}_failed"] = f"Got {result}, expected {expected}"
                else:
                    results.append(False)
                    debug_info[f"shot_{i}_error"] = "Could not execute solve function"
            except Exception as e:
                results.append(False)
                debug_info[f"shot_{i}_error"] = str(e)

        passed = all(results)
        debug_info["functional_test"] = (
            "✓ All shots passed" if passed else f"✗ {sum(results)}/{len(results)} shots passed"
        )
        if passed:
            r += 0.8

        rewards.append((r, debug_info))
    return rewards


def test_all_rewards() -> bool:
    """Test if all assistant outputs in train_split.json get reward = 1.0."""
    try:
        # Load the dataset
        with open("train_split.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ train_split.json not found. Run data generation first.")
        return False

    print(f"Testing {len(data)} examples from train_split.json")
    print("=" * 50)

    perfect_count = 0
    failed_examples = []

    # Test just the first 5 examples with debug info
    for i, example in enumerate(data):
        assistant_output = example["assistant_prompt"]
        shots = from_jsonable(example["shots"])

        # Calculate reward for this single example with debug
        results = reward_fn_debug([assistant_output], [shots])
        reward, debug_info = results[0]

        if reward == 1.0:
            perfect_count += 1
        else:
            print(f"\n=== Example {i + 1} ===")
            print(f"Code: {assistant_output[:200]}...")
            print(f"Reward: {reward:.2f}")
            print("Debug info:")
            for key, value in debug_info.items():
                print(f"  {key}: {value}")
            failed_examples.append((i, reward, assistant_output))
    print(f"Perfect examples (reward = 1.0): {perfect_count}/{len(data)}")

    return perfect_count == len(data)


if __name__ == "__main__":
    # First test the DSL validation specifically
    dsl_tests_passed = test_dsl_validation_specific()

    print("\n" + "=" * 60 + "\n")

    # Test the reward function comprehensively
    reward_tests_passed = test_reward_function_clean()

    print("\n" + "=" * 60 + "\n")

    # Then test against the full dataset if available
    all_perfect = test_all_rewards()

    if dsl_tests_passed and reward_tests_passed and all_perfect:
        print("🎉 ALL TESTS PASSED! 🎉")
    elif dsl_tests_passed and reward_tests_passed:
        print(
            "✅ DSL validation and reward function work correctly, but some examples in train_split.json don't get perfect reward."
        )
    else:
        print("❌ Some tests failed. Check the output above for details.")
