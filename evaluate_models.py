#!/usr/bin/env python
# evaluate_models.py
# Python ≥3.9 -- transformers ≥4.42  -- peft ≥0.10

from __future__ import annotations
import ast, builtins, contextlib, io, json, signal, types, re, inspect
from pathlib import Path
from time import time
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import AutoModelForCausalLMWithValueHead

# ------------------------------------------------------------------ paths
BASE_MODEL   = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LORA_SFT_DIR = "qwen25_coder_lora/final"
LORA_RL_DIR  = "qwen25_coder_grpo/final"   # contains value-head
EVAL_FILE    = "eval_split.json"
MAX_NEW      = 256
TEMPERATURE  = 0.0                         # deterministic

# ------------------------------------------------------------------ util
import arc_dsl.dsl as dsl
DSL_FUNCS = {n for n, f in dsl.__dict__.items() if callable(f) and not n.startswith("_")}

def from_jsonable(x):
    """Convert JSON-serializable data back to Python objects."""
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

def extract_python_code(text):
    """Extract Python code from markdown code blocks."""
    # Match ```python ... ``` blocks
    pattern = r'```python\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no markdown, return as-is (might be plain Python)
    return text.strip()

def create_smart_wrappers(mod):
    """Attach smart wrappers for every public DSL primitive.

    A call like   foo(I)   is translated to   foo(I['a'], I['b'])   (etc.),
    and each extracted value is converted with `from_jsonable` before the
    real DSL function is invoked.
    """
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

def safe_exec(code: str, input_data=None) -> types.ModuleType | None:
    """Execute code safely with proper DSL wrappers and input handling."""
    mod = types.ModuleType("submission")
    mod.dsl = dsl
    mod.__builtins__ = {k: getattr(builtins, k) for k in ("range", "len", "enumerate", "zip")}
    
    # expose raw DSL names first …
    for name, func in dsl.__dict__.items():
        if callable(func) and not name.startswith("_"):
            setattr(mod, name, func)

    # … then replace them with the smart versions
    create_smart_wrappers(mod)

    # make the *entire* input bundle available both as a global 'I'
    # and as the sole argument that we'll pass to solve()
    if input_data is not None:
        processed = from_jsonable(input_data)
        mod.I = processed
    
    try:
        with time_limit(2):
            exec(code, mod.__dict__)
        return mod
    except Exception:
        return None

@contextlib.contextmanager
def time_limit(seconds: int):
    def handler(signum, frame): raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try: yield
    finally: signal.alarm(0)

# ------------------------------------------------------------------ load eval
eval_data = json.loads(Path(EVAL_FILE).read_text())

# ------------------------------------------------------------------ tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

def build_prompt(sample):
    msgs = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user",   "content": sample["user_prompt"]},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

# ------------------------------------------------------------------ models
def load_policy(name: str):
    if name == "base":
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    if name == "sft":
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        return PeftModel.from_pretrained(base, LORA_SFT_DIR)
    if name == "rl":
        # value-head still works with .generate
        return AutoModelForCausalLM.from_pretrained(
            LORA_RL_DIR, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    raise ValueError(name)

models = {k: load_policy(k).eval() for k in ("base", "sft", "rl")}

# ------------------------------------------------------------------ evaluation loop
def accuracy(model) -> float:
    print(f"Evaluating model: {model}...")
    ok = 0
    for i, sample in enumerate(eval_data):
        prompt = build_prompt(sample)
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **input_ids,
                max_new_tokens=MAX_NEW,
                temperature=TEMPERATURE,
                do_sample=False
            )
        gen_text = tokenizer.decode(gen[0][input_ids.input_ids.shape[-1]:], skip_special_tokens=False)

        # Extract Python code from markdown if present
        code = extract_python_code(gen_text)
        
        # Convert shots to proper format
        shots = from_jsonable(sample["shots"])
        
        # functional check
        solved = False
        all_passed = True
        
        for shot in shots:
            try:
                # Parse the input - it might come as a JSON string
                input_data = shot["inputs"]
                if isinstance(input_data, str):
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
            ok += 1
            solved = True
                
        if i > 0 and i % 10 == 0:
            print(f"\nSample {i}:")
            print(f"User prompt: {sample['user_prompt']}")
            print(f"Generated solution: {gen_text}")
            print(f"Extracted code: {code}")
            print(f"Solved correctly: {solved}")
            
    return ok / len(eval_data)

start = time()
results = {name: accuracy(m) for name, m in models.items()}
runtime = time() - start

print("\nFunctional accuracy on eval split:")
for k, v in results.items():
    print(f" {k:>4}: {v*100:5.1f} %")
print(f"(processed {len(eval_data)} tasks in {runtime/60:.1f} min)")
