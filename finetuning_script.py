from pathlib import Path
import torch
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

# 1 . Tokeniser & model (4-bit quant + LoRA)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True          # needed for Qwen chat template
)
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

# 2 . Dataset ----------------------------------------------------------------
raw_ds = load_dataset("json", data_files=DATA_FILE, split="train")

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
    full  = prefix + example["assistant_prompt"]

    tok = tokenizer(
        full,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors=None
    )
    pref_len = len(
        tokenizer(prefix, add_special_tokens=False)["input_ids"]
    )
    labels = [-100] * pref_len + tok["input_ids"][pref_len:]
    tok["labels"] = labels
    return tok

tokenised_ds = raw_ds.map(
    preprocess,
    remove_columns=raw_ds.column_names,
    num_proc=4
)

# 3 . Training ---------------------------------------------------------------
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="qwen25_coder_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,          # effective 32
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    fp16=False,                             # we’re already in BF16
    bf16=True,
    logging_steps=25,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenised_ds,
    data_collator=collator
)

trainer.train()
trainer.save_model("qwen25_coder_lora/final")