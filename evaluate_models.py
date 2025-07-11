#!/usr/bin/env python
# evaluate_models.py
# Python ≥3.9 -- transformers ≥4.42  -- peft ≥0.10

from __future__ import annotations
import ast, builtins, contextlib, io, json, signal, types
from pathlib import Path
from time import time

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

def safe_exec(code: str) -> types.ModuleType | None:
    mod = types.ModuleType("submission")
    mod.dsl = dsl
    mod.__builtins__ = {k: getattr(builtins, k) for k in ("range", "len", "enumerate", "zip")}
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

        # functional check
        mod = safe_exec(gen_text)
        solved = False
        if mod and callable(getattr(mod, "solve", None)):
            try:
                with time_limit(2):
                    if all(mod.solve(s["inputs"]) == s["output"] for s in sample["shots"]):
                        ok += 1
                        solved = True
            except Exception:
                pass
                
        if i > 0 and i % 10 == 0:
            print(f"\nSample {i}:")
            print(f"User prompt: {sample['user_prompt'][:100]}...")
            print(f"Generated solution: {gen_text[:100]}...")
            print(f"Solved correctly: {solved}")
            
    return ok / len(eval_data)

start = time()
results = {name: accuracy(m) for name, m in models.items()}
runtime = time() - start

print("\nFunctional accuracy on eval split:")
for k, v in results.items():
    print(f" {k:>4}: {v*100:5.1f} %")
print(f"(processed {len(eval_data)} tasks in {runtime/60:.1f} min)")
