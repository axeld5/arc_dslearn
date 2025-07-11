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
    device_map="auto",
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
# 5. Reward function
# ---------------------------------------------------------------------
import ast, builtins, types, contextlib, io, signal

SOLVE_RE   = re.compile(r"\bdef\s+solve\s*\(\s*I\s*\)\s*:")
IMPORT_RE  = re.compile(r"^\s*import\s+", re.M)

@contextlib.contextmanager
def time_limit(seconds: int):
    def handler(signum, frame): raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try: yield
    finally: signal.alarm(0)

def safe_exec(code: str) -> types.ModuleType | None:
    """Exec in a stripped environment and return the resulting module."""
    mod = types.ModuleType("submission")
    mod.dsl = dsl                               # expose DSL only
    mod.__builtins__ = {}                       # no default built-ins
    # allow the minimal handful of harmless built-ins
    for b in ("range", "len", "enumerate", "zip"):
        mod.__builtins__[b] = getattr(builtins, b)
    try:
        with time_limit(2):                     # protect from hangs
            exec(code, mod.__dict__)
        return mod
    except Exception:
        return None

def reward_fn(completions, shots, **_):
    """Reward = 0.1 (format) + 0.1 (DSL only) + 0.8 (all shots pass)."""
    rewards: List[float] = []
    for code, shot_list in zip(completions, shots):
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
        mod = safe_exec(code)
        if mod and callable(getattr(mod, "solve", None)):
            try:
                passed = all(mod.solve(s["inputs"]) == s["output"] for s in shot_list)
            except Exception:
                passed = False
            if passed:
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
    max_prompt_length   = 4096,          # leave room for completions
    max_completion_length = 512,
    remove_unused_columns = False,       # we keep “shots”
    push_to_hub         = False,
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