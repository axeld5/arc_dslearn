"""Script for evaluating the SFT model only."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import time
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.arc_dslearn.metrics_and_rewards.reward_fn import equivalent, extract_python_code, safe_exec
from src.arc_dslearn.utils import from_jsonable


def evaluate_sft_model() -> Dict[str, float]:
    """Evaluate SFT model with original DSL prompt."""
    # ──────────────────────────── bookkeeping dicts ─────────────────────────
    tot_per_fun: Counter[str] = Counter()  # global frequency
    err_per_fun: Counter[str] = Counter()
    results: Dict[str, float] = {}

    # ------------------------------------------------------------------ paths
    LORA_SFT_DIR = "qwen2.5_coder_dslearn_os_sft/final"
    EVAL_FILE = "eval_split.json"

    MAX_NEW = 64
    TEMPERATURE = 0.0  # deterministic

    SHOW_SAMPLES = True
    SAMPLE_EVERY = 100

    # ------------------------------------------------------------------ prompt building
    def build_prompt(sample: Dict[str, Any], tokenizer: Any) -> str:
        """Format the prompt using original DSL system prompt."""
        msgs = [
            {"role": "system", "content": sample["system_prompt"]},  # Original DSL prompt
            {"role": "user", "content": sample["user_prompt"]},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------ model loading
    print("Loading SFT model...")
    model = AutoModelForCausalLM.from_pretrained(LORA_SFT_DIR).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(LORA_SFT_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------ data loading
    print("Loading evaluation data...")
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

    prompts = [build_prompt(sample, tokenizer) for sample in raw_eval]

    for sample in raw_eval:
        tot_per_fun[sample["name"]] += 1

    # ------------------------------------------------------------------ evaluation functions
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
                mod.I = shot["inputs"]
                if not equivalent(solve(shot["inputs"]), shot["output"], shot["inputs"]):
                    return False
            return True
        except Exception:
            return False

    def check_sample_tagged(sample: Dict[str, Any], gen_text: str) -> Tuple[str, bool]:
        """Check sample and returns solved status."""
        solved = check_sample(sample, gen_text)
        return sample["name"], solved

    # ------------------------------------------------------------------ evaluation loop
    print("→ Evaluating SFT model (with DSL prompt)")
    start = time()

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
        gen_text = tokenizer.decode(gen[0][enc.input_ids.shape[-1] :], skip_special_tokens=False)
        name, solved = check_sample_tagged(sample, gen_text)
        ok += int(solved)
        if not solved:
            err_per_fun[name] += 1
        if SHOW_SAMPLES and i % SAMPLE_EVERY == 0:
            print(f"\n— sample {i}…")
            print(gen_text)
            print(f"\nSolved: {solved}")

    accuracy = ok / n
    results["sft"] = accuracy
    runtime = time() - start

    # ───────────────────── DSL function error summary ───────────────────────
    functions_with_multiple_errors = {
        name: count for name, count in err_per_fun.items() if count > 0
    }
    if functions_with_multiple_errors:
        print(f"\n📊 DSL Functions with >0 errors (total: {len(functions_with_multiple_errors)}):")
        # Sort by error count descending
        sorted_errors = sorted(
            functions_with_multiple_errors.items(), key=lambda x: x[1], reverse=True
        )
        for func_name, error_count in sorted_errors:
            total_occurrences = tot_per_fun[func_name]
            error_rate = (error_count / total_occurrences) * 100
            print(
                f"  • {func_name}: {error_count} errors / {total_occurrences} total ({error_rate:.1f}%)"
            )
    else:
        print("\n📊 No DSL functions have more than 0 errors!")

    # ───────────────────── plot ❶ error-rate per function ───────────────────────
    df_total = pd.DataFrame({"occ": pd.Series(tot_per_fun)})
    df = df_total.copy()
    df["err"] = pd.Series(err_per_fun).fillna(0).astype(int)
    df["error_rate"] = df["err"] / df["occ"]
    top10 = df.sort_values("error_rate", ascending=False).head(10)

    ax = top10.sort_values("error_rate")["error_rate"].plot(
        kind="barh",
        figsize=(7, 4),
        title="Top-10 DSL primitives by error-rate (SFT - DSL prompt)",
    )
    ax.set_xlabel("error rate")
    ax.set_ylabel("DSL primitive")
    plt.tight_layout()
    plt.savefig("chart_results/error_rate_sft_only.png")
    plt.close()

    # ───────────────────── plot ❷ overall accuracy bar ───────────────────────
    acc_df = pd.Series(results).sort_values().to_frame("accuracy")
    ax = acc_df["accuracy"].plot(
        kind="barh", figsize=(6, 3), title="SFT Model Accuracy on eval split (DSL prompt)"
    )
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig("chart_results/sft_model_accuracy.png")
    plt.close()

    print("\nSFT Model Functional accuracy on eval split (DSL prompt):")
    print(f"SFT: {accuracy * 100:5.1f} %")
    print(f"(processed {len(raw_eval)} tasks in {runtime / 60:.1f} min)")
    print("\nSaved plots:")
    print(" • chart_results/sft_model_accuracy.png")
    print(" • chart_results/error_rate_sft_only.png")

    return results


if __name__ == "__main__":
    evaluate_sft_model()
