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
from reward_fn_batched import reward_fn
from json_utils import from_jsonable

from huggingface_hub import login

login()

# ---------------------------------------------------------------------
# 0. Paths & constants
# ---------------------------------------------------------------------
BASE_MODEL   = "Qwen/Qwen2.5-Coder-1.5B"
LORA_PATH    = "qwen2.5_1.5b_coder_dslearn_os_sft/final"         # ← your SFT–LoRA adapter
DATA_PATH    = "train_split.json"                  # same JSON as before

# ---------------------------------------------------------------------
# 1. Tokenizer (ChatML template)
# ---------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# ---------------------------------------------------------------------
# 2. Loading model  (LoRA)
# ---------------------------------------------------------------------
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
model = PeftModel.from_pretrained(                          # NEW
    base,
    LORA_PATH,
    is_trainable=True,          # ← **must be True** so LoRA weights get grads
).to("cuda")

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
# 4. GRPO config  – add **mandatory** generation parameters
# ---------------------------------------------------------------------
grpo_cfg = GRPOConfig(
    output_dir          = "qwen2.5_1.5b_coder_dslearn_os_rl",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    num_train_epochs    = 1,
    learning_rate       = 2e-5,
    lr_scheduler_type   = "cosine",
    logging_steps       = 10,
    save_steps          = 100,
    optim='paged_adamw_8bit', 
    logging_dir="tb_logs",              # <- where events get written
    report_to="tensorboard",            # or "wandb", "csv", …
    # -------- GRPO-specific -----------
    num_generations     = 4,             # G in the paper
    max_prompt_length   = 8192,          # leave room for completions
    max_completion_length = 64,
    remove_unused_columns = False,       # we keep "shots"
    push_to_hub         = True,
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
trainer.save_model("qwen2.5_1.5b_coder_dslearn_os_rl/final")