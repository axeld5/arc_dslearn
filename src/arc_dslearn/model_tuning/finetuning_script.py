"""Finetuning script using Unsloth for faster training."""

import os
from typing import Any, Dict

from unsloth import FastLanguageModel
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTTrainer
from transformers import TrainingArguments


if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
    DATA_FILE = "train_split.json"
    MAX_LEN = 2048
    
    load_dotenv()
    login(os.getenv("HF_TOKEN"))

    # Load model and tokenizer with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LEN,
        dtype=None,  # Auto-detect dtype
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
        device_map = "balanced",
    )

    # Configure LoRA using Unsloth
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

    # Load and format dataset
    raw_ds = load_dataset("json", data_files=DATA_FILE, split="train")

    def format_chat_template(example: Dict[str, Any]) -> Dict[str, str]:
        """Format the example into a chat template for SFT."""
        messages = [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["user_prompt"]},
            {"role": "assistant", "content": example["assistant_prompt"]},
        ]
        
        # Use Unsloth's chat template formatting
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": text}

    # Format the dataset
    formatted_ds = raw_ds.map(format_chat_template, remove_columns=raw_ds.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="qwen2.5_7b_coder_dslearn_os_sft_unsloth",
        per_device_train_batch_size=2,  # Can increase with Unsloth optimizations
        gradient_accumulation_steps=4,  # Reduced due to higher batch size
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        logging_dir="tb_logs_unsloth",
        report_to="tensorboard",
        remove_unused_columns=False,
        push_to_hub=True,
        seed=3407,
        optim="adamw_8bit",  # Memory efficient optimizer
        ddp_find_unused_parameters = False,
    )

    # Use Unsloth's optimized SFT trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_ds,
        dataset_text_field="text",  # Column containing the formatted text
        max_seq_length=MAX_LEN,
        dataset_num_proc=4,
        packing=False,  # Disable packing for better stability with complex data
        args=training_args,
    )

    # Enable faster training with Unsloth
    FastLanguageModel.for_training(model)

    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model("qwen2.5_7b_coder_dslearn_os_sft_unsloth/final")
    
    # Optional: Save to hub
    model.push_to_hub("axel-darmouni/qwen2.5-coder-arc-dslearn-sft", tokenizer=tokenizer, token=os.getenv("HF_TOKEN"))
