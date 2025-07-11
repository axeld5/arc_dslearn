#!/usr/bin/env python3
import json
import random
import tiktoken

# Load the tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# Load the dataset
with open("train_split.json", "r") as f:
    data = json.load(f)

# Pick a random example
example = random.choice(data)

# Extract system and user prompts
system_prompt = example["system_prompt"]
user_prompt = example["user_prompt"]

print(f"=== RANDOM EXAMPLE ===")
print(f"System prompt: {system_prompt[:200]}...")
print(f"User prompt: {user_prompt[:200]}...")
print()

# Format as chat template
formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

print(f"=== FORMATTED PROMPT ===")
print(f"Formatted prompt: {formatted_prompt[:500]}...")
print()

# Count tokens
tokens = encoding.encode(formatted_prompt)
token_count = len(tokens)

print(f"=== TOKEN COUNT ===")
print(f"Total tokens: {token_count}")
print(f"System prompt tokens: {len(encoding.encode(system_prompt))}")
print(f"User prompt tokens: {len(encoding.encode(user_prompt))}")
print(f"Formatted prompt tokens: {token_count}")

# Show some token details
print(f"\nFirst 20 tokens: {tokens[:20]}")
print(f"Last 20 tokens: {tokens[-20:]}")