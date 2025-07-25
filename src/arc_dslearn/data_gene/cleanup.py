"""Cleanup the data to remove samples with more than token_limit tokens."""

import json

import tiktoken


def remove_long_token_samples(
    input_file, output_file, token_limit=8000, model_name="gpt-3.5-turbo"
):
    """Remove samples with more than token_limit tokens."""
    with open(input_file, "r") as f:
        data = json.load(f)

    enc = tiktoken.encoding_for_model(model_name)
    filtered = []
    removed = 0
    for i, sample in enumerate(data):
        system_prompt = sample.get("system_prompt", "")
        user_prompt = sample.get("user_prompt", "")
        full_prompt = system_prompt + user_prompt
        tokens = enc.encode(full_prompt)
        if len(tokens) <= token_limit:
            filtered.append(sample)
        else:
            print(f"Sample {i} of funcname {sample['name']} has {len(tokens)} tokens.")
            removed += 1

    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)
    print(
        f"{removed} samples removed from {input_file} (>{token_limit} tokens). {len(filtered)} remain. Saved to {output_file}."
    )


if __name__ == "__main__":
    # Clean train_set.json
    remove_long_token_samples("train_split.json", "train_split.json")
    # Clean eval_set.json
    remove_long_token_samples("eval_split.json", "eval_split.json")
