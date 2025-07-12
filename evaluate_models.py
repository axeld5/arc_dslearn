#!/usr/bin/env python
# evaluate_models.py
# Python ≥3.9 -- transformers ≥4.42  -- peft ≥0.10

from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from reward_fn_batched import equivalent, safe_exec, extract_python_code
from json_utils import from_jsonable

# ------------------------------------------------------------------ paths
BASE_MODEL   = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LORA_SFT_DIR = "qwen2.5_1.5b_coder_dslearn_os_sft/final"
LORA_RL_DIR  = "qwen2.5_1.5b_coder_dslearn_os_rl/final"
EVAL_FILE    = "eval_split.json"

MAX_NEW      = 128
TEMPERATURE  = 0.0                         # deterministic
BATCH_SIZE   = 16
NUM_THREADS  = 8

SHOW_SAMPLES = True
SAMPLE_EVERY = 20

# ------------------------------------------------------------------ models
def build_prompt(sample, tokenizer):
    msgs = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user", "content": sample["user_prompt"]},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def load_policy(name: str):
    if name == "base":
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
    if name == "sft":
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="flash_attention_2"     
        )
        return PeftModel.from_pretrained(base, LORA_SFT_DIR)
    if name == "rl":
        # value-head still works with .generate
        return AutoModelForCausalLM.from_pretrained(
            LORA_RL_DIR, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="flash_attention_2" 
        )
    raise ValueError(name)

models = {k: load_policy(k).eval() for k in ("base", "sft", "rl")}

raw_eval = json.loads(Path(EVAL_FILE).read_text())
for sample in raw_eval:
    shots_py = []
    for shot in from_jsonable(sample["shots"]):
        inp = shot["inputs"]
        if isinstance(inp, str):
            inp = json.loads(inp)
        inp = from_jsonable(inp)

        out = shot["output"]
        if isinstance(out, str):
            out = json.loads(out)
        out = from_jsonable(out)
        
        shots_py.append({"inputs": inp, "output": out})
    sample["shots_py"] = shots_py

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompts = [build_prompt(sample, tokenizer) for sample in raw_eval]

# ------------------------------------------------------------------ evaluation loop
_module_cache = {}

def check_sample(sample, gen_text) -> bool:
    """Returns True if the generated solution solves the task."""
    code = extract_python_code(gen_text)
    mod = _module_cache.get(code)
    if mod is None:
        try:
            mod = safe_exec(code, sample["shots_py"][0]["inputs"])
            _module_cache[code] = mod
        except Exception:
            return False
    solve = getattr(mod, "solve", None)
    if not callable(solve):
        return False
    try:
        for shot in sample["shots_py"]:
            mod.I = shot["inputs"]
            if not equivalent(solve(shot["inputs"]), shot["output"], shot["inputs"]):
                return False
        return True
    except Exception:
        return False


def accuracy(model) -> float:
    print(f"→ Evaluating model: {model.__class__.__name__}")
    ok = 0
    n = len(raw_eval)
    for batch_start in range(0, n, BATCH_SIZE):
        slc = slice(batch_start, batch_start + BATCH_SIZE)
        batch_prompts = prompts[slc]

        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        input_lens = enc["attention_mask"].sum(-1)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                temperature=TEMPERATURE,
                do_sample=False,
                use_cache=True,
            )
        futures, pool = [], ThreadPoolExecutor(max_workers=NUM_THREADS)
        for i, sample_idx in enumerate(range(batch_start, min(batch_start+BATCH_SIZE, n))):
            sample = raw_eval[sample_idx]
            gen_text = tokenizer.decode(gen[i][input_lens[i]:], skip_special_tokens=False)
            if SHOW_SAMPLES and i % SAMPLE_EVERY == 0:
                solved = check_sample(sample, gen_text)
                ok += int(solved)
                print(f"\n— sample {sample_idx}…")
                print(gen_text)
                print(f"\nSolved: {solved}")
            else:
                futures.append(pool.submit(check_sample, sample, gen_text))
        for fut in as_completed(futures):
            ok += int(fut.result())
        pool.shutdown(wait=False)
        if batch_start % 50 == 0:
            done = batch_start + len(batch_prompts)
            print(f"→ Evaluated {done}/{n} tasks ({ok}/{done} correct)")
    return ok / n

start = time()
results = {name: accuracy(m) for name, m in models.items()}
runtime = time() - start

print("\nFunctional accuracy on eval split:")
for k, v in results.items():
    print(f" {k:>4}: {v*100:5.1f} %")
print(f"(processed {len(raw_eval)} tasks in {runtime/60:.1f} min)")
