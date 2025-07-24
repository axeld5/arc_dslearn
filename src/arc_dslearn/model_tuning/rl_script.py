"""RL script for training the model using Unsloth for faster training."""

from __future__ import annotations

import os
from typing import Any, Dict
import torch

from unsloth import FastLanguageModel
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from trl import (
    GRPOConfig,
    GRPOTrainer,
)

from ..metrics_and_rewards.reward_fn import reward_function
from ..utils import from_jsonable

# ---------------------------------------------------------------------
# 0. Paths & constants
# ---------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    login(os.getenv("HF_TOKEN"))
    BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
    LORA_PATH = "qwen2.5_7b_coder_dslearn_os_sft_unsloth/final"  # Updated path for Unsloth SFT model
    DATA_PATH = "train_split.json"
    MAX_LEN = 2048

    # ---------------------------------------------------------------------
    # 1. Load model and tokenizer with Unsloth optimizations
    # ---------------------------------------------------------------------
    # First load the base model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_LEN,
        dtype=None,  # Auto-detect dtype
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
        device_map="balanced",
    )

    # Configure LoRA using Unsloth (for RL fine-tuning)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimization
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Load weights from the SFT checkpoint if it exists
    try:
        # Try to load the adapter weights from the SFT checkpoint
        import os
        if os.path.exists(LORA_PATH):
            print(f"Loading SFT adapter weights from {LORA_PATH}")
            # Load the saved adapter weights
            model.load_state_dict(torch.load(f"{LORA_PATH}/adapter_model.bin"), strict=False)
    except Exception as e:
        print(f"Could not load SFT weights from {LORA_PATH}: {e}")
        print("Starting RL training from base model...")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    # 4. GRPO config with Unsloth optimizations
    # ---------------------------------------------------------------------
    grpo_cfg = GRPOConfig(
        output_dir="qwen2.5_coder_dslearn_os_rl_unsloth",
        per_device_train_batch_size=2,  # Can increase with Unsloth optimizations
        gradient_accumulation_steps=4,  # Reduced due to higher batch size
        num_train_epochs=1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="adamw_8bit",  # Memory efficient optimizer
        logging_dir="rl_tb_logs_unsloth",
        report_to="tensorboard",
        # -------- GRPO-specific -----------
        num_generations=4,  # G in the paper
        max_prompt_length=MAX_LEN - 64,  # leave room for completions
        max_completion_length=64,
        remove_unused_columns=False,  # we keep "shots"
        push_to_hub=True,
        fp16=False,
        bf16=True,
        seed=3407,
        ddp_find_unused_parameters=False,
    )

    # ---------------------------------------------------------------------
    # 5. Enable faster training with Unsloth
    # ---------------------------------------------------------------------
    FastLanguageModel.for_training(model)

    # ---------------------------------------------------------------------
    # 6. Trainer with Unsloth optimized model
    # ---------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_function],
        args=grpo_cfg,
        train_dataset=ds,
    )

    trainer.train()
    
    # Save the model
    trainer.save_model("qwen2.5_coder_dslearn_os_rl_unsloth/final")
    
    # Optional: Save to hub
    model.push_to_hub("axel-darmouni/qwen2.5-coder-arc-dslearn-rl", tokenizer=tokenizer, token=os.getenv("HF_TOKEN"))
