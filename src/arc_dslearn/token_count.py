"""Count the number of tokens in the system and user prompts of the train split."""

import json

import tiktoken

if __name__ == "__main__":
    # Load all samples from train_split.json
    with open("train_split.json", "r") as f:
        data = json.load(f)

    # Use tiktoken to count tokens for each sample's assistant_prompt
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # or use a different model if needed

    token_counts = []
    for i, sample in enumerate(data):
        full_prompt = sample.get("system_prompt", "") + sample.get("user_prompt", "")
        tokens = enc.encode(full_prompt)
        token_counts.append(len(tokens))
        print(f"Sample {i}: {len(tokens)} tokens")

    print(f"Average tokens: {sum(token_counts) / len(token_counts):.2f}")
    print(f"Max tokens: {max(token_counts)}")
    print(f"Min tokens: {min(token_counts)}")
