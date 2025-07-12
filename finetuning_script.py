from pathlib import Path
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"       # base (not –Instruct)
DATA_FILE  = "train_split.json"                    # produced by your script
MAX_LEN    = 8192                               # stay well below 32 k context

# 1 . Tokeniser & model (4-bit quant + LoRA)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True          # needed for Qwen chat template
)
# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

# LoRA on the attention projection matrices
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()  # sanity-check

# 2 . Dataset ----------------------------------------------------------------
# First, preprocess the JSON to ensure consistent data types (same as script.py)
def preprocess_json_file(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Convert inputs and outputs to JSON strings for consistency
    for record in data:
        if record['shots']:
            for shot in record['shots']:
                shot['inputs'] = json.dumps(shot['inputs'])
                shot['output'] = json.dumps(shot['output'])
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

# Preprocess the data
preprocess_json_file("train_split.json", "train_split_processed.json")

# Load the preprocessed dataset
raw_ds = load_dataset("json",
                      data_files="train_split_processed.json",
                      split="train")

def preprocess(example):
    # Build full message sequence
    messages = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example["user_prompt"]},
        {"role": "assistant", "content": example["assistant_prompt"]}
    ]
    
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Messages without the assistant's message (to compute the prefix)
    prefix_messages = messages[:-1]
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True  # ensures <|assistant|> is added
    )
    
    # Tokenize full and prefix (no padding during preprocessing)
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LEN,
        padding=False  # Don't pad during preprocessing
    )
    prefix_tokens = tokenizer(
        prefix_text,
        truncation=True,
        max_length=MAX_LEN,
        padding=False
    )
    
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    
    # Create labels as a proper list copy
    labels = list(input_ids)  # Ensure it's a flat list
    prefix_len = len(prefix_tokens["input_ids"])
    
    # Mask all tokens up to the assistant's reply
    for i in range(min(prefix_len, len(labels))):
        labels[i] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenised_ds = raw_ds.map(
    preprocess,
    remove_columns=raw_ds.column_names,
    num_proc=4
)

# 3 . Training ---------------------------------------------------------------
# Enable padding in the data collator

from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8          # keep your 8-byte alignment
    label_pad_token_id: int = -100              # ignored by the loss

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1) pull out labels before calling tokenizer.pad()
        labels = [feat.pop("labels") for feat in features]

        # 2) pad input_ids & attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # 3) pad labels to the same sequence length – fill with -100
        max_len = batch["input_ids"].size(1)
        batch["labels"] = torch.stack([
            torch.tensor(l + [self.label_pad_token_id] * (max_len - len(l)))
            for l in labels
        ])

        return batch
    
collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

args = TrainingArguments(
    output_dir="qwen25_coder_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,          # effective 32
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    fp16=False,                             # we're already in BF16
    bf16=True,
    logging_steps=25,
    save_steps=500,
    save_total_limit=2,
    logging_dir="tb_logs",              # <- where events get written
    report_to="tensorboard",      
    remove_unused_columns=False,
    deepspeed="ds_config_zero3.json",
    ddp_find_unused_parameters=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenised_ds,
    data_collator=collator
)

trainer.train()
trainer.save_model("qwen25_coder_lora/final")