"""Pilot script for generating training data for the RL script."""

from __future__ import annotations

import gc
import json
from pathlib import Path

from tqdm import tqdm  # optional progress bar

from src.arc_dslearn.data_gene_callable.block_generation import (
    clear_internal_caches,
    make_multiline_block,
)


def main_generate_blocks(
    n: int = 60,
    max_lines: int = 20,
    seed: int = 1337,
    max_path_budget: int = 256,
    clear_every: int = 10,
):
    """Generate n multi-line Grid->Grid blocks using make_multiline_block.

    Returns
    -------
        dict with 'high_quality' and 'low_quality' lists

    """
    high_quality_blocks = []
    low_quality_blocks = []
    failed_count = 0

    for i in range(n):
        try:
            block = make_multiline_block(
                max_lines=max_lines, seed=seed + i, n_shots=3, path_budget=max_path_budget
            )

            # Separate by quality
            quality = block.get("quality", "high")
            if quality == "low":
                low_quality_blocks.append(block)
            else:
                high_quality_blocks.append(block)

            if (i + 1) % 20 == 0:
                print(
                    f"  Generated {i + 1}/{n} blocks (high: {len(high_quality_blocks)}, low: {len(low_quality_blocks)}, failed: {failed_count})"
                )
        except Exception as e:
            failed_count += 1
            print(f"  Warning: Failed to generate block {i + 1} (seed={seed + i}): {e}")
        finally:
            if (i + 1) % clear_every == 0:
                clear_internal_caches()
                gc.collect()

    print(
        f"  Successfully generated {len(high_quality_blocks) + len(low_quality_blocks)}/{n} blocks:"
    )
    print(f"    - High quality: {len(high_quality_blocks)}")
    print(f"    - Low quality: {len(low_quality_blocks)}")
    print(f"    - Failed: {failed_count}")

    return {
        "high_quality": high_quality_blocks,
        "low_quality": low_quality_blocks,
    }


def generate_blocks_to_jsonl(
    n_samples: int = 1000,
    out_path: str = "generated_blocks.jsonl",
    start_seed: int = 0,
    clear_every: int = 10,
    path_budget: int = 64,
    include_examples_str: bool = False,
):
    """Stream DSL-generated blocks to JSONL (one block per line, no large list kept in memory).

    Separates high-quality and low-quality blocks into different files.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Create low-quality output path
    low_quality_path = out_file.parent / f"{out_file.stem}_low_quality{out_file.suffix}"

    # Statistics tracking
    stats = {
        "high_quality": 0,
        "low_quality": 0,
        "failed": 0,
        "quality_reasons": {},
    }

    with (
        out_file.open("w", encoding="utf-8") as f_high,
        low_quality_path.open("w", encoding="utf-8") as f_low,
    ):
        for i in tqdm(range(n_samples), desc="Generating blocks"):
            seed = start_seed + i
            try:
                block = make_multiline_block(
                    max_lines=30,
                    path_budget=path_budget,
                    seed=seed,
                    include_examples_str=include_examples_str,  # toggle to save memory
                )

                # Check quality and route to appropriate file
                quality = block.get("quality", "high")
                if quality == "low":
                    f_low.write(json.dumps(block, ensure_ascii=False) + "\n")
                    stats["low_quality"] += 1
                    reason = block.get("quality_reason", "unknown")
                    stats["quality_reasons"][reason] = stats["quality_reasons"].get(reason, 0) + 1
                else:
                    f_high.write(json.dumps(block, ensure_ascii=False) + "\n")
                    stats["high_quality"] += 1

            except Exception as e:
                stats["failed"] += 1
                print(f"⚠️  Skipped sample {i} (seed={seed}): {e}")
            finally:
                # Free memory regularly (only after first item)
                if i and (i % clear_every == 0):
                    clear_internal_caches()
                    gc.collect()

    print("\n✅ Done! Generation statistics:")
    print(f"  - High quality blocks: {stats['high_quality']} → {out_path}")
    print(f"  - Low quality blocks: {stats['low_quality']} → {low_quality_path}")
    print(f"  - Failed generations: {stats['failed']}")

    if stats["quality_reasons"]:
        print("\n📊 Low quality reasons breakdown:")
        for reason, count in sorted(stats["quality_reasons"].items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")


def run_pipeline():
    """Run the data generation pipeline."""
    # Step 1: Generate multi-line Grid→Grid training data
    print("Step 1: Generating multi-line Grid→Grid training data...")
    generate_blocks_to_jsonl(
        n_samples=5000,
        out_path="data/blocks_200.jsonl",
        include_examples_str=False,  # disable long string field
        path_budget=256,
        start_seed=42,
    )

    """
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
    """


if __name__ == "__main__":
    run_pipeline()
