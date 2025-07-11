#!/usr/bin/env python3
import json
import ast
import re
import types
import builtins
import contextlib
import signal
from typing import List

# Import the DSL and from_jsonable function
import arc_dsl.dsl as dsl

from collections import Counter

def canonical(x):
    """Return a deterministic representation that ignores order
    for containers that are *logically* unordered."""
    if isinstance(x, (int, float, bool, str)) or x is None:
        return x

    if isinstance(x, (set, frozenset)):
        return frozenset(canonical(v) for v in sorted(x, key=repr))

    if isinstance(x, tuple):
        # Are the elements *all* simple scalars?  Then order matters
        # (e.g. a grid row).  Otherwise we sort.
        if all(isinstance(v, (int, float, bool, str)) for v in x):
            return tuple(canonical(v) for v in x)
        return tuple(sorted((canonical(v) for v in x), key=repr))

    if isinstance(x, list):
        return tuple(canonical(v) for v in x)

    if isinstance(x, dict):
        return tuple(sorted((k, canonical(v)) for k, v in x.items()))
    
    if all(isinstance(v, int) for v in x):          # NEW – scalar tuple
        return tuple(sorted(x))                    # NEW – deterministic    

    return x

def equivalent(a, b, shot_inputs=None):
    """True if values should be considered equal in ARC-DSL land."""
    if canonical(a) == canonical(b):
        return True

    # special-case: colour ties ------------------------------------------
    if (isinstance(a, int) and isinstance(b, int) and
        isinstance(shot_inputs, dict) and 'obj' in shot_inputs):
        obj = from_jsonable(shot_inputs['obj'])
        if isinstance(obj, frozenset):          # sanity guard
            colors = {pix for pix, _ in obj}
            if a in colors and b in colors:
                freq   = {c: dsl.colorcount(obj, c) for c in colors}
                maxcnt = max(freq.values())
                max_colors = {c for c, f in freq.items() if f == maxcnt}  # NEW
                return a in max_colors and b in max_colors               # NEW  
    return False

def from_jsonable(x):
    if isinstance(x, dict):
        if "__frozenset__" in x:
            return frozenset(from_jsonable(v) for v in x["__frozenset__"])
        if "__tuple__" in x:
            return tuple(from_jsonable(v) for v in x["__tuple__"])
        if "__set__" in x:
            return set(from_jsonable(v) for v in x["__set__"])
        if "__str__" in x:
            return x["__str__"]
        return {k: from_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [from_jsonable(v) for v in x]
    return x

# Copy the reward function components from rl_script.py
SOLVE_RE = re.compile(r"\bdef\s+solve\s*\(\s*I\s*\)\s*:")
IMPORT_RE = re.compile(r"^\s*import\s+", re.M)

@contextlib.contextmanager
def time_limit(seconds: int):
    def handler(signum, frame): raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try: yield
    finally: signal.alarm(0)

def create_smart_wrappers(mod):
    """Attach smart wrappers for every public DSL primitive.

    A call like   foo(I)   is translated to   foo(I['a'], I['b'])   (etc.),
    and each extracted value is converted with `from_jsonable` before the
    real DSL function is invoked.
    """
    import inspect

    def make_wrapper(dsl_func):
        try:
            sig = inspect.signature(dsl_func)
        except (TypeError, ValueError):          # C built-ins (bool, int, …)
            return dsl_func                      # → leave them untouched

        param_names = [
            p.name for p in sig.parameters.values()
            if p.default is p.empty and p.kind is p.POSITIONAL_OR_KEYWORD
        ]
        arity = len(param_names)

        def wrapper(*args, **kwargs):
            if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                bundle = args[0]

                # ––– 1. perfect match on parameter names –––––––––––––––––
                if set(param_names) <= bundle.keys():
                    vals = [from_jsonable(bundle[n]) for n in param_names]

                # ––– 2. fallback: same cardinality –––––––––––––––––––––––
                elif len(bundle) == arity:
                    vals = [from_jsonable(v) for v in bundle.values()]

                # ––– 3. single-parameter convenience –––––––––––––––––––––
                elif arity == 1:
                    # common synonyms
                    for k in ('patch', 'obj', 'object', 'piece',
                              'grid', 'container', 'a'):
                        if k in bundle:
                            vals = [from_jsonable(bundle[k])]
                            break
                    else:
                        # first (or only) value
                        vals = [from_jsonable(next(iter(bundle.values())))]

                # ––– 4. we cannot guess ––––––––––––––––––––––––––––––––––
                else:
                    return dsl_func(*args, **kwargs)

                return dsl_func(*vals)
            return dsl_func(*args, **kwargs)

        return wrapper

    for name, func in dsl.__dict__.items():
        if callable(func) and not name.startswith("_"):
            setattr(mod, name, make_wrapper(func))

def safe_exec(code: str, input_data=None):
    mod               = types.ModuleType("submission")
    mod.dsl           = dsl
    mod.__builtins__  = {n: getattr(builtins, n) for n in ("range", "len",
                                                           "enumerate", "zip")}

    # expose raw DSL names first …
    for name, func in dsl.__dict__.items():
        if callable(func) and not name.startswith("_"):
            setattr(mod, name, func)

    # … then replace them with the smart versions
    create_smart_wrappers(mod)

    # make the *entire* input bundle available both as a global ‘I’
    # and as the sole argument that we’ll pass to solve()
    processed = from_jsonable(input_data)
    mod.I     = processed

    with time_limit(2):
        exec(code, mod.__dict__)
    return mod

def reward_fn(completions, shots, **_):
    """Reward = 0.1 (format) + 0.1 (DSL only) + 0.8 (all shots pass)."""
    rewards: List[float] = []
    for code_raw, shot_list in zip(completions, shots):
        # Extract Python code from markdown
        code = extract_python_code(code_raw)
        
        shot_list = from_jsonable(shot_list)
        r = 0.0

        # (1) Solve function present
        r += 0.1 if SOLVE_RE.search(code) else -1.0

        # (2) No bad imports, only DSL names
        try:
            tree = ast.parse(code)
            bad_imports = bool(IMPORT_RE.search(code))
            names      = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            unknown    = names - {"I", "O"} - {
                f"x{i}" for i in range(1, 100)
            } - set(__import__("arc_dsl.dsl").dsl.__dict__.keys())
            if not bad_imports and not unknown:
                r += 0.1
        except SyntaxError:
            pass

        # (3) Functional correctness on all provided shots
        all_passed = True
        for shot in shot_list:
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
                    
                    if not equivalent(result, expected, input_data):
                        all_passed = False
                        break
                else:
                    all_passed = False
                    break
            except Exception:
                all_passed = False
                break
        
        if all_passed:
            r += 0.8
            
        rewards.append(r)
    return rewards

def extract_python_code(text):
    """Extract Python code from markdown code blocks."""
    import re
    # Match ```python ... ``` blocks
    pattern = r'```python\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no markdown, return as-is (might be plain Python)
    return text.strip()

def reward_fn_debug(completions, shots, **_):
    """Debug version of reward function that shows why each component fails."""
    rewards: List[float] = []
    for code_raw, shot_list in zip(completions, shots):
        # Extract Python code from markdown
        code = extract_python_code(code_raw)
        
        shot_list = from_jsonable(shot_list)
        r = 0.0
        debug_info = {"extracted_code": code[:100] + "..." if len(code) > 100 else code}

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
        debug_info["functional_test"] = f"✓ All shots passed" if passed else f"✗ {sum(results)}/{len(results)} shots passed"
        if passed:
            r += 0.8
        
        rewards.append((r, debug_info))
    return rewards

def test_all_rewards():
    """Test if all assistant outputs in train_split.json get reward = 1.0"""
    
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
            print(f"\n=== Example {i+1} ===")
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