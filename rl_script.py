#!/usr/bin/env python
# train_grpo_qwen25_coder.py
# Python ≥3.9  –  trl ≥0.19.1  –  transformers ≥4.42  –  peft ≥0.10

from __future__ import annotations
import ast
import re
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import (
    GRPOConfig,
    GRPOTrainer,
    AutoModelForCausalLMWithValueHead,
)

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

# ---------------------------------------------------------------------
# 0. Paths & constants
# ---------------------------------------------------------------------
BASE_MODEL   = "Qwen/Qwen2.5-Coder-1.5B"
LORA_PATH    = "qwen25_coder_lora/final"         # ← your SFT–LoRA adapter
DATA_PATH    = "train_split.json"                  # same JSON as before
MAX_LEN      = 4096

# ---------------------------------------------------------------------
# 1. Tokenizer (ChatML template)
# ---------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# ---------------------------------------------------------------------
# 2. Policy model  (4-bit + LoRA + value head)
# ---------------------------------------------------------------------
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(                          # NEW
    base,
    LORA_PATH,
    is_trainable=True,          # ← **must be True** so LoRA weights get grads
)

# ---------------------------------------------------------------------
# 3. Dataset ⇒  {"prompt", "reference"}
# ---------------------------------------------------------------------
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")

def to_rl(example):
    msgs = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user",   "content": example["user_prompt"]},
    ]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    # keep the full shots list so that reward_fn can check correctness
    example["shots"] = from_jsonable(example["shots"])

    return {"prompt": prompt, "shots": example["shots"]}

ds = raw_ds.map(to_rl, remove_columns=raw_ds.column_names, num_proc=4)

# ---------------------------------------------------------------------
# 4. Helper: list of *public* DSL function names
# ---------------------------------------------------------------------
import arc_dsl.dsl as dsl
DSL_FUNCS = {name for name, f in dsl.__dict__.items() if callable(f) and not name.startswith("_")}

# ---------------------------------------------------------------------
# 5. Improved reward function components
# ---------------------------------------------------------------------
import ast, builtins, types, contextlib, io, signal, json, inspect
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

SOLVE_RE   = re.compile(r"\bdef\s+solve\s*\(\s*I\s*\)\s*:")
IMPORT_RE  = re.compile(r"^\s*import\s+", re.M)

@contextlib.contextmanager
def time_limit(seconds: int):
    def handler(signum, frame): raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try: yield
    finally: signal.alarm(0)

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

    # make the *entire* input bundle available both as a global 'I'
    # and as the sole argument that we'll pass to solve()
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

# ---------------------------------------------------------------------
# 5. GRPO config  – add **mandatory** generation parameters
# ---------------------------------------------------------------------
grpo_cfg = GRPOConfig(
    output_dir          = "qwen25_coder_grpo",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    num_train_epochs    = 3,
    learning_rate       = 2e-5,
    lr_scheduler_type   = "cosine",
    logging_steps       = 10,
    save_steps          = 100,
    # -------- GRPO-specific -----------
    num_generations     = 4,             # G in the paper
    max_prompt_length   = 8192,          # leave room for completions
    max_completion_length = 128,
    remove_unused_columns = False,       # we keep "shots"
    push_to_hub         = False,
    deepspeed="ds_config_zero2.json",
    ddp_find_unused_parameters=False
)

# ---------------------------------------------------------------------
# 6. Trainer
#     • `prompt_column` is **not** a valid argument (caused the crash).
#     • Pass the tokenizer via `processing_class`.
#     • `reward_funcs` must be **list or callable** – both work, but
#       passing a list keeps the API identical to the working script.
# ---------------------------------------------------------------------
trainer = GRPOTrainer(
    model              = model,
    processing_class   = tokenizer,
    reward_funcs       = [reward_fn],
    args               = grpo_cfg,
    train_dataset      = ds,
)

trainer.train()
trainer.save_model("qwen25_coder_grpo/final")