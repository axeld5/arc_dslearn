"""Script for evaluating the models."""

from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path
from time import time
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.arc_dslearn.metrics_and_rewards.reward_fn import equivalent, extract_python_code, safe_exec
from src.arc_dslearn.utils import from_jsonable

if __name__ == "__main__":
    # ──────────────────────────── bookkeeping dicts ─────────────────────────
    tot_per_fun: Counter[str] = Counter()  # global frequency
    err_per_fun: Dict[str, Counter[str]] = {name: Counter() for name in ("base", "sft", "rl")}
    results: Dict[str, float] = {}

    # overall accuracy
    # ─────────────────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------ paths
    BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    LORA_SFT_DIR = "qwen2.5_1.5b_coder_dslearn_os_sft/final"
    LORA_RL_DIR = "qwen2.5_1.5b_coder_dslearn_os_rl/final"
    EVAL_FILE = "eval_split.json"

    MAX_NEW = 64
    TEMPERATURE = 0.0  # deterministic
    BATCH_SIZE = 16
    NUM_THREADS = 8

    SHOW_SAMPLES = True
    SAMPLE_EVERY = 20

    # ------------------------------------------------------------------ models
    def build_prompt(sample: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
        """Format the prompt."""
        msgs = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["user_prompt"]},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # type: ignore

    def load_policy(name: str) -> Any:
        """Load the models depending on name."""
        attn_impl = "flash_attention_2" if platform.system() == "Linux" else "eager"
        if name == "base":
            return AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
        if name == "sft":
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
            return PeftModel.from_pretrained(base, LORA_SFT_DIR)
        if name == "rl":
            # value-head still works with .generate
            return AutoModelForCausalLM.from_pretrained(
                LORA_RL_DIR,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_impl,
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

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)  # type: ignore
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [build_prompt(sample, tokenizer) for sample in raw_eval]

    for sample in raw_eval:  # run *once* before any evaluation
        tot_per_fun[sample["name"]] += 1

    # ------------------------------------------------------------------ evaluation loop
    _module_cache: Dict[str, Any] = {}

    def check_sample(sample: Dict[str, Any], gen_text: str) -> bool:
        """Return True if the generated solution solves the task."""
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
                mod.I = shot["inputs"]  # type: ignore[union-attr]
                if not equivalent(solve(shot["inputs"]), shot["output"], shot["inputs"]):
                    return False
            return True
        except Exception:
            return False

    def check_sample_tagged(sample: Dict[str, Any], gen_text: str) -> Tuple[str, bool]:
        """Check sample and returns solved status."""
        solved = check_sample(sample, gen_text)
        return sample["name"], solved  # <─ return both

    def accuracy(model: Any, tag: str) -> float:
        """Get model accuracies."""
        print(f"→ Evaluating model: {tag}")
        ok = 0
        n = len(raw_eval)
        for i, sample in enumerate(raw_eval):
            prompt = prompts[i]
            enc = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW,
                    temperature=TEMPERATURE,
                    do_sample=False,
                    use_cache=True,
                )
            gen_text = tokenizer.decode(
                gen[0][enc.input_ids.shape[-1] :], skip_special_tokens=False
            )
            name, solved = check_sample_tagged(sample, gen_text)  # ← now every sample counts
            ok += int(solved)
            if not solved:
                err_per_fun[tag][name] += 1
            if SHOW_SAMPLES and i % SAMPLE_EVERY == 0:
                print(f"\n— sample {i}…")
                print(gen_text)
                print(f"\nSolved: {solved}")
        return ok / n

    start = time()
    for tag, model in models.items():
        results[tag] = accuracy(model, tag)

    runtime = time() - start

    # ─────────────────────────────────────────────────────────────────────────

    # ───────────────────── plot ❶ error-rate per model ───────────────────────
    df_total = pd.DataFrame({"occ": pd.Series(tot_per_fun)})

    for tag, counter in err_per_fun.items():
        df = df_total.copy()
        df["err"] = pd.Series(counter).fillna(0).astype(int)
        df["error_rate"] = df["err"] / df["occ"]
        top10 = df.sort_values("error_rate", ascending=False).head(10)

        ax = top10.sort_values("error_rate")["error_rate"].plot(
            kind="barh", figsize=(7, 4), title=f"Top-10 DSL primitives by error-rate  ({tag})"
        )
        ax.set_xlabel("error rate")
        ax.set_ylabel("DSL primitive")
        plt.tight_layout()
        plt.savefig(f"error_rate_{tag}.png")  # → figure per model
        plt.close()

    # ───────────────────── plot ❷ overall accuracy bar ───────────────────────
    acc_df = pd.Series(results).sort_values().to_frame("accuracy")
    ax = acc_df["accuracy"].plot(
        kind="barh", figsize=(6, 3), title="Functional accuracy on eval split"
    )
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig("model_accuracy.png")
    plt.close()

    print("\nFunctional accuracy on eval split:")
    for tag, acc in results.items():
        print(f"{tag:>4}: {acc * 100:5.1f} %")
    print(f"(processed {len(raw_eval)} tasks in {runtime / 60:.1f} min)")
    print("\nSaved plots:")
    print(" • model_accuracy.png")
    for tag in err_per_fun:
        print(f" • error_rate_{tag}.png")
