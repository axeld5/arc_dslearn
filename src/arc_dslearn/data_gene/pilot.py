"""Pilot script for generating training data for the RL script."""

from __future__ import annotations

import inspect
import json
from pprint import pformat
from typing import Any, Sequence

import src.arc_dslearn.arc_dsl.dsl as dsl

from .block_generation import make_block, should_skip_function
from .data_processing import create_train_eval_split, prepare_datasets_for_loading


def main() -> Sequence[dict[str, Any]]:
    """Generate blocks of code for DSL functions."""
    blocks = []
    for _ in range(20):
        for name, func in inspect.getmembers(dsl, inspect.isfunction):
            if name.startswith("_"):
                continue

            if should_skip_function(func):
                print(f"[info] skipped {name}: has callable parameter or returns callable")
                continue

            try:
                blocks.append(make_block(func))
            except Exception as err:
                # silenced: comment out the next line for verbose debugging
                print(f"[warn] skipped {name}: {err}")
                pass
    return blocks


if __name__ == "__main__":
    # Step 1: Generate training data
    print("Step 1: Generating training data...")
    training_blocks = main()
    with open("train_set.json", "w") as fp:
        json.dump(training_blocks, fp, indent=2, default=lambda o: pformat(o))
    print(f"✓ Wrote train_set.json with {len(training_blocks)} examples")

    # Step 2: Create train/eval split
    print("\nStep 2: Creating train/eval split...")
    train_data, eval_data = create_train_eval_split()

    # Step 3: Preprocess for dataset loading
    print("\nStep 3: Preprocessing for dataset loading...")
    prepare_datasets_for_loading()

    print("\n✓ Pipeline complete! Ready to use:")
    print(f"  - train_set.json ({len(training_blocks)} examples)")
    print(f"  - train_split.json ({len(train_data)} examples)")
    print(f"  - eval_split.json ({len(eval_data)} examples)")

    # Optional: Show how to load the datasets
    print("\nTo load the datasets:")
    print("from datasets import load_dataset")
    print("train_ds = load_dataset('json', data_files='train_split.json', split='train')")
    print("eval_ds = load_dataset('json', data_files='eval_split.json', split='train')")
