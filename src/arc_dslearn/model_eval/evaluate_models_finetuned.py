"""Script for evaluating the fine-tuned models (SFT & RL) plus base model with DSL prompt."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import time
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from unsloth import FastLanguageModel

from src.arc_dslearn.metrics_and_rewards.reward_fn import equivalent, extract_python_code, safe_exec
from src.arc_dslearn.utils import from_jsonable


def evaluate_finetuned_models() -> Dict[str, float]:
    """Evaluate SFT & RL models plus base model with original DSL prompt."""
    # ──────────────────────────── bookkeeping dicts ─────────────────────────
    tot_per_fun: Counter[str] = Counter()  # global frequency
    err_per_fun: Dict[str, Counter[str]] = {name: Counter() for name in ("base", "sft", "rl")}
    results: Dict[str, float] = {}

    # ------------------------------------------------------------------ paths
    BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
    LORA_SFT_DIR = "qwen2.5_coder_dslearn_os_sft_unsloth/final"
    LORA_RL_DIR = "qwen2.5_coder_dslearn_os_rl_unsloth/final"
    EVAL_FILE = "eval_split.json"

    MAX_NEW = 64
    TEMPERATURE = 0.0  # deterministic
    MAX_SEQ_LENGTH = 8192

    SHOW_SAMPLES = True
    SAMPLE_EVERY = 20

    # ------------------------------------------------------------------ models
    def build_prompt(sample: Dict[str, Any], tokenizer: Any) -> str:
        """Format the prompt using original DSL system prompt."""
        msgs = [
            {"role": "system", "content": sample["system_prompt"]},  # Original DSL prompt
            {"role": "user", "content": sample["user_prompt"]},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def load_policy(name: str) -> Any:
        """Load the models using Unsloth for optimal performance."""
        if name == "base":
            model, _ = FastLanguageModel.from_pretrained(
                model_name=BASE_MODEL,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,  # Auto-detect
                load_in_4bit=True,
                device_map="balanced",
            )
            FastLanguageModel.for_inference(model)
            return model

        elif name == "sft":
            model, _ = FastLanguageModel.from_pretrained(
                model_name=LORA_SFT_DIR,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
                device_map="balanced",
            )
            FastLanguageModel.for_inference(model)
            return model

        elif name == "rl":
            model, _ = FastLanguageModel.from_pretrained(
                model_name=LORA_RL_DIR,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
                device_map="balanced",
            )
            FastLanguageModel.for_inference(model)
            return model

        raise ValueError(f"Unknown model name: {name}")

    models = {k: load_policy(k) for k in ("base", "sft", "rl")}

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

    # Load tokenizer
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        device_map="balanced",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [build_prompt(sample, tokenizer) for sample in raw_eval]

    for sample in raw_eval:
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

    def accuracy(model: Any, tag: str) -> float:
        """Get model accuracies."""
        print(f"→ Evaluating model: {tag} (with DSL prompt)")
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
            name, solved = check_sample_tagged(sample, gen_text)
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

    # ───────────────────── plot ❶ error-rate per model ───────────────────────
    df_total = pd.DataFrame({"occ": pd.Series(tot_per_fun)})

    for tag, counter in err_per_fun.items():
        df = df_total.copy()
        df["err"] = pd.Series(counter).fillna(0).astype(int)
        df["error_rate"] = df["err"] / df["occ"]
        top10 = df.sort_values("error_rate", ascending=False).head(10)

        ax = top10.sort_values("error_rate")["error_rate"].plot(
            kind="barh",
            figsize=(7, 4),
            title=f"Top-10 DSL primitives by error-rate  ({tag} - DSL prompt)",
        )
        ax.set_xlabel("error rate")
        ax.set_ylabel("DSL primitive")
        plt.tight_layout()
        plt.savefig(f"chart_results/error_rate_{tag}_dsl.png")
        plt.close()

    # ───────────────────── plot ❷ overall accuracy bar ───────────────────────
    acc_df = pd.Series(results).sort_values().to_frame("accuracy")
    ax = acc_df["accuracy"].plot(
        kind="barh", figsize=(6, 3), title="Functional accuracy on eval split (DSL prompt)"
    )
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig("chart_results/model_accuracy_dsl.png")
    plt.close()

    print("\nFunctional accuracy on eval split (DSL prompt):")
    for tag, acc in results.items():
        print(f"{tag:>4}: {acc * 100:5.1f} %")
    print(f"(processed {len(raw_eval)} tasks in {runtime / 60:.1f} min)")
    print("\nSaved plots:")
    print(" • chart_results/model_accuracy_dsl.png")
    for tag in err_per_fun:
        print(f" • chart_results/error_rate_{tag}_dsl.png")

    return results


if __name__ == "__main__":
    evaluate_finetuned_models()
