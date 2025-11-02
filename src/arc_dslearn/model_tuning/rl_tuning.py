"""RL script for training the model."""

from __future__ import annotations

import os
import platform
from typing import Any, Dict

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    GSPOConfig,
    GSPOTrainer,
)

from src.arc_dslearn.metrics_and_rewards.reward_fn import reward_fn
from src.arc_dslearn.utils import from_jsonable

# ---------------------------------------------------------------------
# 0. Paths & constants
# ---------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    login(os.getenv("HF_TOKEN"))
    BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
    LORA_PATH = "qwen2.5_coder_dslearn_os_sft/final"  # ← your SFT–LoRA adapter
    DATA_PATH = "train_split.json"  # same JSON as before

    # ---------------------------------------------------------------------
    # 1. Tokenizer (ChatML template)
    # ---------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)  # type: ignore

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------------------
    # 2. Loading model  (LoRA)
    # ---------------------------------------------------------------------
    attn_impl = "flash_attention_2" if platform.system() == "Linux" else "eager"
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        LORA_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to("cuda")

    # ---------------------------------------------------------------------
    # 3. Dataset ⇒  {"prompt", "reference"}
    # ---------------------------------------------------------------------
    raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")

    def to_rl(example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the dataset to the RL format."""
        msgs = [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["user_prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # keep the full shots list so that reward_fn can check correctness
        example["shots"] = from_jsonable(example["shots"])

        return {"prompt": prompt, "shots": example["shots"]}

    ds = raw_ds.map(to_rl, remove_columns=raw_ds.column_names, num_proc=4)

    # ---------------------------------------------------------------------
    # 4. GSPO config  – add **mandatory** generation parameters
    # ---------------------------------------------------------------------
    grpo_cfg = GSPOConfig(
        output_dir="qwen2.5_coder_dslearn_os_rl",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        optim="paged_adamw_8bit",
        logging_dir="rl_tb_logs",  # <- where events get written
        report_to="tensorboard",  # or "wandb", "csv", …
        # -------- GRPO-specific -----------
        num_generations=4,  # G in the paper
        max_prompt_length=8192,  # leave room for completions
        max_completion_length=64,
        remove_unused_columns=False,  # we keep "shots"
        push_to_hub=True,
        deepspeed="src/arc_dslearn/model_tuning/ds_config_zero2.json",
        ddp_find_unused_parameters=False,
    )

    # ---------------------------------------------------------------------
    # 6. Trainer
    #     • `prompt_column` is **not** a valid argument (caused the crash).
    #     • Pass the tokenizer via `processing_class`.
    #     • `reward_funcs` must be **list or callable** – both work, but
    #       passing a list keeps the API identical to the working script.
    # ---------------------------------------------------------------------
    trainer = GSPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=ds,
    )

    trainer.train()
    trainer.save_model("qwen2.5_coder_dslearn_os_rl/final")

    # Optional: Save to hub
    model.push_to_hub(
        "axel-darmouni/qwen2.5-coder-arc-dslearn-rl",
        tokenizer=tokenizer,
        token=os.getenv("HF_TOKEN"),
    )
