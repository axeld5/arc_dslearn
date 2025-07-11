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
MAX_LEN    = 4096                                # stay well below 32 k context

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
    device_map="auto",
    trust_remote_code=True
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
    """
    • Input  = system + user  (chat template with <|assistant|> start token)
    • Labels = assistant reply  (mask the prefix with –100)
    """
    # ---- build chat up to assistant ----------------------------------------
    msgs_prompt = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user",   "content": example["user_prompt"]}
    ]
    prefix = tokenizer.apply_chat_template(
        msgs_prompt,
        tokenize=False,
        add_generation_prompt=True     # appends "<|assistant|>"
    )
    full_text = prefix + example["assistant_prompt"]

    # Tokenize the full text
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        return_tensors=None
    )
    
    # Tokenize just the prefix to get the correct length
    prefix_tokens = tokenizer(
        prefix,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        return_tensors=None
    )
    
    # Create labels - mask the prefix with -100, keep the assistant response
    labels = full_tokens["input_ids"].copy()
    prefix_len = len(prefix_tokens["input_ids"])
    
    # Mask the prefix (system + user + assistant start token)
    for i in range(min(prefix_len, len(labels))):
        labels[i] = -100
    
    return {
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels
    }

tokenised_ds = raw_ds.map(
    preprocess,
    remove_columns=raw_ds.column_names,
    num_proc=4
)

# 3 . Training ---------------------------------------------------------------
# Enable padding in the data collator
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
    pad_to_multiple_of=8,  # For better performance
    return_tensors="pt"
)

args = TrainingArguments(
    output_dir="qwen25_coder_lora",
    per_device_train_batch_size=4,
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
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenised_ds,
    data_collator=collator
)

trainer.train()
trainer.save_model("qwen25_coder_lora/final")

# Clean up the temporary file
import os
os.remove("train_split_processed.json")