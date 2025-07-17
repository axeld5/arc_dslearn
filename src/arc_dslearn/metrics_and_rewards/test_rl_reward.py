"""Module for testing the reward function's validity over the generated data."""

import ast
import json
from typing import Any, Dict, List, Tuple

from src.arc_dslearn.metrics_and_rewards.reward_fn import IMPORT_RE, SOLVE_RE, equivalent, safe_exec
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
            dsl_names = set(__import__("arc_dsl.dsl").dsl.__dict__.keys())
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
    # Load the dataset
    with open("train_split.json", "r") as f:
        data = json.load(f)

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
    all_perfect = test_all_rewards()
    if all_perfect:
        print("🎉 ALL EXAMPLES GET PERFECT REWARD! 🎉")
    else:
        print("❌ Some examples don't get perfect reward.")
