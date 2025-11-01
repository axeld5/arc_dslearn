"""Pilot script for generating training data for the RL script."""

from __future__ import annotations

import json

from src.arc_dslearn.data_gene_grid.block_generation import make_multiline_block
from src.arc_dslearn.data_gene_grid.data_processing import (
    create_train_eval_split,
    prepare_datasets_for_loading,
    remove_answer_overlap,
)


def main_generate_blocks(n: int = 200, max_lines: int = 5, seed: int = 1337):
    """Generate n multi-line Grid->Grid blocks using make_multiline_block."""
    blocks = []
    failed_count = 0

    for i in range(n):
        try:
            block = make_multiline_block(max_lines=max_lines, seed=seed + i, n_shots=3)
            blocks.append(block)
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{n} blocks (failed: {failed_count})")
        except Exception as e:
            failed_count += 1
            print(f"  Warning: Failed to generate block {i + 1} (seed={seed + i}): {e}")
            continue

    print(f"  Successfully generated {len(blocks)}/{n} blocks (failed: {failed_count})")
    return blocks


def run_pipeline():
    """Run the data generation pipeline."""
    # Step 1: Generate multi-line Grid→Grid training data
    print("Step 1: Generating multi-line Grid→Grid training data...")
    training_blocks = main_generate_blocks()

    if not training_blocks:
        print("✗ Error: No training blocks were generated successfully. Exiting.")
        return

    with open("train_set.json", "w") as fp:
        json.dump(training_blocks, fp, indent=2)
    print(f"✓ Wrote train_set.json with {len(training_blocks)} examples")

    # Step 2: Create train/eval split (use filtered data)
    print("\nStep 2: Creating train/eval split...")
    _, eval_data = create_train_eval_split(src_file="train_set.json")

    # Step 3: Remove answer overlaps from train split
    print("\nStep 3: Removing answer overlaps from train split...")
    filtered_train_data, train_overlap_stats = remove_answer_overlap(
        "train_split.json", "train_split.json"
    )
    print("✓ Saved filtered train data to train_split.json")

    # Combine stats for summary
    overlap_stats = {
        "blocks_removed": train_overlap_stats["blocks_removed"],
        "ambiguous_blocks": train_overlap_stats["ambiguous_blocks"],
        "unsolvable_blocks": train_overlap_stats["unsolvable_blocks"],
        "functions_affected": sorted(set(train_overlap_stats["functions_affected"])),
        "dsl_functions_tested": train_overlap_stats["dsl_functions_tested"],
        "criteria": train_overlap_stats["criteria"],
    }

    # Step 4: Prepare datasets for loading
    print("\nStep 4: Preparing datasets for loading...")
    prepare_datasets_for_loading()
    print("✓ Saved filtered data to train_split.json and eval_split.json")

    print("\n✓ Pipeline complete! Ready to use:")
    print(f"  - train_set.json ({len(training_blocks)} examples - original)")
    print(f"  - train_split.json ({len(filtered_train_data)} examples - filtered)")
    print(f"  - eval_split.json ({len(eval_data)} examples - original)")

    # Show overlap statistics
    if overlap_stats["blocks_removed"] > 0:
        print("\n📊 Overlap removal statistics:")
        print(f"  - Blocks removed: {overlap_stats['blocks_removed']}")
        print(f"  - Ambiguous blocks: {overlap_stats['ambiguous_blocks']}")
        print(f"  - Unsolvable blocks: {overlap_stats['unsolvable_blocks']}")
        print(f"  - Functions affected: {len(overlap_stats['functions_affected'])}")
        print(f"  - DSL functions tested: {overlap_stats['dsl_functions_tested']}")
        print(f"  - Criteria: {overlap_stats['criteria']}")

    # Optional: Show how to load the datasets
    print("\nTo load the datasets:")
    print("from datasets import load_dataset")
    print("train_ds = load_dataset('json', data_files='train_split.json', split='train')")
    print("eval_ds = load_dataset('json', data_files='eval_split.json', split='train')")


if __name__ == "__main__":
    run_pipeline()
