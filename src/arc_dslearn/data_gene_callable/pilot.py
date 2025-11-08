"""Pilot script for generating training data for the RL script."""

from __future__ import annotations

import gc
import json
import random
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
    max_lines: int = 30,
    include_examples_str: bool = False,
    append_mode: bool = False,
    batch_id: int | None = None,
):
    """Stream DSL-generated blocks to JSONL (one block per line, no large list kept in memory).

    Separates high-quality and low-quality blocks into different files.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    out_path : str
        Path to output JSONL file for high-quality blocks
    start_seed : int
        Starting seed for random number generation
    clear_every : int
        Clear caches every N samples to manage memory
    path_budget : int
        Budget for path exploration attempts
    max_lines : int
        Maximum number of lines in generated code
    include_examples_str : bool
        Whether to include example strings in output (memory intensive)
    append_mode : bool
        If True, append to existing files instead of overwriting
    batch_id : int, optional
        Batch identifier for progress tracking

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

    # Use append mode if requested
    mode = "a" if append_mode else "w"

    desc = f"Batch {batch_id}: Generating blocks" if batch_id is not None else "Generating blocks"

    with (
        out_file.open(mode, encoding="utf-8") as f_high,
        low_quality_path.open(mode, encoding="utf-8") as f_low,
    ):
        for i in tqdm(range(n_samples), desc=desc):
            seed = start_seed + i
            try:
                block = make_multiline_block(
                    max_lines=max_lines,
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

    batch_info = f" (Batch {batch_id})" if batch_id is not None else ""
    print(f"\n✅ Done{batch_info}! Generation statistics:")
    print(f"  - High quality blocks: {stats['high_quality']} → {out_path}")
    print(f"  - Low quality blocks: {stats['low_quality']} → {low_quality_path}")
    print(f"  - Failed generations: {stats['failed']}")

    if stats["quality_reasons"]:
        print("\n📊 Low quality reasons breakdown:")
        for reason, count in sorted(stats["quality_reasons"].items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")

    return stats


def run_pipeline():
    """Run the data generation pipeline in 10 batches of 100 samples each."""
    print("=" * 80)
    print("Starting data generation pipeline: 10 batches × 100 samples = 1000 total")
    print("=" * 80)

    out_path = "data/blocks_200.jsonl"
    num_batches = 10
    samples_per_batch = 100

    # Check if files already exist
    out_file = Path(out_path)
    low_quality_path = out_file.parent / f"{out_file.stem}_low_quality{out_file.suffix}"

    files_exist = out_file.exists() or low_quality_path.exists()
    if files_exist:
        print("\n📝 Existing files detected - will APPEND to:")
        if out_file.exists():
            print(f"  - {out_path}")
        if low_quality_path.exists():
            print(f"  - {low_quality_path}")
    else:
        print("\n📝 Creating new files:")
        print(f"  - {out_path}")
        print(f"  - {low_quality_path}")

    # Aggregate statistics across all batches
    total_stats = {
        "high_quality": 0,
        "low_quality": 0,
        "failed": 0,
        "quality_reasons": {},
    }

    for batch_num in range(1, num_batches + 1):
        print(f"\n{'=' * 80}")
        print(f"Batch {batch_num}/{num_batches}: Generating {samples_per_batch} samples")
        print(f"{'=' * 80}")

        # Generate random seed for this batch (more random)
        batch_seed = random.randint(1000, 999999)
        print(f"Using random seed: {batch_seed}")

        # First batch overwrites (if files exist from previous run), subsequent batches append
        append_mode = files_exist or (batch_num > 1)

        batch_stats = generate_blocks_to_jsonl(
            max_lines=30,
            n_samples=samples_per_batch,
            out_path=out_path,
            include_examples_str=False,  # disable long string field
            path_budget=128,
            start_seed=batch_seed,
            append_mode=append_mode,
            batch_id=batch_num,
        )

        # Aggregate statistics
        total_stats["high_quality"] += batch_stats["high_quality"]
        total_stats["low_quality"] += batch_stats["low_quality"]
        total_stats["failed"] += batch_stats["failed"]

        # Merge quality reasons
        for reason, count in batch_stats["quality_reasons"].items():
            total_stats["quality_reasons"][reason] = (
                total_stats["quality_reasons"].get(reason, 0) + count
            )

        # Clear caches between batches
        clear_internal_caches()
        gc.collect()

    # Final summary
    print(f"\n{'=' * 80}")
    print("🎉 PIPELINE COMPLETE - Final Statistics:")
    print(f"{'=' * 80}")
    print(f"  Total samples processed: {num_batches * samples_per_batch}")
    print(f"  ✓ High quality blocks: {total_stats['high_quality']}")
    print(f"  ⚠ Low quality blocks: {total_stats['low_quality']}")
    print(f"  ✗ Failed generations: {total_stats['failed']}")

    if total_stats["quality_reasons"]:
        print("\n📊 Overall low quality reasons breakdown:")
        for reason, count in sorted(total_stats["quality_reasons"].items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")

    print("\n📁 Output files:")
    print(f"  - High quality: {out_path}")
    print(f"  - Low quality: {low_quality_path}")
    print(f"{'=' * 80}")

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
