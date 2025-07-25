"""Script for evaluating the base model with simplified input-to-output prompt."""

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


def evaluate_base_model_simple() -> Dict[str, float]:
    """Evaluate base model with simplified input-to-output prompt."""
    # ──────────────────────────── bookkeeping dicts ─────────────────────────
    tot_per_fun: Counter[str] = Counter()  # global frequency
    err_per_fun: Dict[str, Counter[str]] = {"base_simple": Counter()}
    results: Dict[str, float] = {}

    # ------------------------------------------------------------------ paths
    BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
    EVAL_FILE = "eval_split.json"

    MAX_NEW = 256  # More tokens for potentially longer solutions
    TEMPERATURE = 0.0  # deterministic
    MAX_SEQ_LENGTH = 8192

    SHOW_SAMPLES = True
    SAMPLE_EVERY = 20

    # ------------------------------------------------------------------ models
    def build_simple_prompt(sample: Dict[str, Any], tokenizer: Any) -> str:
        """Format the prompt with simplified system prompt for base model."""
        # Extract examples from the sample
        examples_text = ""
        for i, shot in enumerate(sample["shots_py"], 1):
            examples_text += f"# Example {i}\n"
            examples_text += f"Input: {shot['inputs']}\n"
            examples_text += f"Expected Output: {shot['output']}\n\n"

        simple_system_prompt = """You are a Python programmer. Your task is to write a function that transforms the given inputs to the expected outputs.

Look at the input-output examples carefully and write a Python function called `solve` that implements the pattern you observe.

Requirements:
- Write only the `solve(I)` function
- The function should take one parameter `I` (the input)
- Return the correct output based on the pattern you identify
- Use standard Python operations and data structures
- Do not import any external libraries"""

        simple_user_prompt = f"""{examples_text}Write a Python function `solve(I)` that transforms the input to the expected output based on the pattern shown in the examples above.

```python
def solve(I):
    # Your implementation here
    return result
```"""

        msgs = [
            {"role": "system", "content": simple_system_prompt},
            {"role": "user", "content": simple_user_prompt},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def load_base_model() -> Any:
        """Load the base model using Unsloth for optimal performance."""
        model, _ = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
            device_map="balanced",
        )
        FastLanguageModel.for_inference(model)
        return model

    model = load_base_model()

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

    prompts = [build_simple_prompt(sample, tokenizer) for sample in raw_eval]

    for sample in raw_eval:
        tot_per_fun[sample["name"]] += 1

    # ------------------------------------------------------------------ evaluation loop
    _module_cache: Dict[str, Any] = {}

    def check_sample_simple(sample: Dict[str, Any], gen_text: str) -> bool:
        """Return True if the generated solution solves the task (more flexible parsing)."""
        code = extract_python_code(gen_text)

        # If no code block found, try to extract function directly
        if not code or "def solve" not in code:
            lines = gen_text.split("\n")
            func_lines = []
            in_function = False
            for line in lines:
                if "def solve" in line:
                    in_function = True
                if in_function:
                    func_lines.append(line)
                    # Simple heuristic: stop at next function or end
                    if (
                        line.strip()
                        and not line.startswith(" ")
                        and not line.startswith("\t")
                        and "def solve" not in line
                    ):
                        if func_lines:
                            func_lines.pop()  # Remove the non-function line
                        break
            if func_lines:
                code = "\n".join(func_lines)

        if not code:
            return False

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
        solved = check_sample_simple(sample, gen_text)
        return sample["name"], solved

    def accuracy_base_simple() -> float:
        """Get base model accuracy with simple prompt."""
        print("→ Evaluating base model with simple input-to-output prompt")
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
                err_per_fun["base_simple"][name] += 1
            if SHOW_SAMPLES and i % SAMPLE_EVERY == 0:
                print(f"\n— sample {i}…")
                print(gen_text)
                print(f"\nSolved: {solved}")
        return ok / n

    start = time()
    results["base_simple"] = accuracy_base_simple()
    runtime = time() - start

    # ───────────────────── plot ❶ error-rate per model ───────────────────────
    df_total = pd.DataFrame({"occ": pd.Series(tot_per_fun)})

    for tag, counter in err_per_fun.items():
        df = df_total.copy()
        df["err"] = pd.Series(counter).fillna(0).astype(int)
        df["error_rate"] = df["err"] / df["occ"]
        top10 = df.sort_values("error_rate", ascending=False).head(10)

        ax = top10.sort_values("error_rate")["error_rate"].plot(
            kind="barh", figsize=(7, 4), title=f"Top-10 tasks by error-rate ({tag})"
        )
        ax.set_xlabel("error rate")
        ax.set_ylabel("task name")
        plt.tight_layout()
        plt.savefig(f"chart_results/error_rate_{tag}.png")
        plt.close()

    # ───────────────────── plot ❷ overall accuracy bar ───────────────────────
    acc_df = pd.Series(results).sort_values().to_frame("accuracy")
    ax = acc_df["accuracy"].plot(
        kind="barh", figsize=(6, 3), title="Base model accuracy with simple prompt"
    )
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig("chart_results/model_accuracy_base_simple.png")
    plt.close()

    print("\nBase model accuracy with simple input-to-output prompt:")
    for tag, acc in results.items():
        print(f"{tag:>12}: {acc * 100:5.1f} %")
    print(f"(processed {len(raw_eval)} tasks in {runtime / 60:.1f} min)")
    print("\nSaved plots:")
    print(" • chart_results/model_accuracy_base_simple.png")
    for tag in err_per_fun:
        print(f" • chart_results/error_rate_{tag}.png")

    return results


if __name__ == "__main__":
    evaluate_base_model_simple()
