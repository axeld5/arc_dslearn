"""Test the remove_answer_overlap function for unique solutions."""

import json
import os
import tempfile

from src.arc_dslearn.data_gene.data_processing import remove_answer_overlap


def test_remove_answer_overlap_unique():
    """Test removing I/O pairs that don't have unique solutions."""
    # Create sample data with different types of overlaps
    sample_data = [
        {
            "name": "identity",
            "shots": [
                # This should be kept if only identity can solve it
                {"inputs": {"x": [1, 2, 3]}, "output": [1, 2, 3]},
                # This might be solvable by multiple functions (e.g., identity, add with 0)
                {"inputs": {"x": 42}, "output": 42},
            ],
        },
        {
            "name": "add",
            "shots": [
                # This should be unique to add function
                {"inputs": {"a": 5, "b": 3}, "output": 8},
                # This might overlap with other functions
                {"inputs": {"a": 0, "b": 42}, "output": 42},
            ],
        },
        {
            "name": "vlflip",
            "shots": [
                # Symmetric grid - might be solvable by multiple flip functions
                {"inputs": {"grid": ((1, 2, 1), (2, 3, 2))}, "output": ((2, 3, 2), (1, 2, 1))},
                # Asymmetric - should be unique
                {"inputs": {"grid": ((1, 2, 3), (4, 5, 6))}, "output": ((4, 5, 6), (1, 2, 3))},
            ],
        },
    ]

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f, indent=2)
        input_file = f.name

    try:
        # Test the function
        print("Testing remove_answer_overlap for unique solutions...")
        print("=" * 60)

        filtered_data, overlap_stats = remove_answer_overlap(
            input_file,
            min_solutions=1,  # At least one function should solve it
            max_solutions=1,  # But no more than one (unique solution)
        )

        print("\n📋 Original data:")
        total_original_shots = 0
        for block in sample_data:
            shots_count = len(block["shots"])
            total_original_shots += shots_count
            print(f"  {block['name']}: {shots_count} shots")
        print(f"  Total: {total_original_shots} shots")

        print("\n📋 Filtered data (unique solutions only):")
        total_filtered_shots = 0
        for block in filtered_data:
            shots_count = len(block["shots"])
            total_filtered_shots += shots_count
            print(f"  {block['name']}: {shots_count} shots")
        print(f"  Total: {total_filtered_shots} shots")

        print("\n📊 Statistics:")
        print(f"  DSL functions tested: {overlap_stats['dsl_functions_tested']}")
        print(f"  Total shots processed: {overlap_stats['total_shots']}")
        print(f"  Shots kept: {overlap_stats['shots_kept']}")
        print(f"  Shots removed: {overlap_stats['shots_removed']}")
        print(f"  Ambiguous pairs (>1 solution): {overlap_stats['ambiguous_pairs']}")
        print(f"  Unsolvable pairs (0 solutions): {overlap_stats['unsolvable_pairs']}")
        print(f"  Functions affected: {len(overlap_stats['functions_affected'])}")

        # Show some examples
        if overlap_stats["ambiguous_examples"]:
            print("\n🔍 Examples of ambiguous pairs (first few):")
            for i, example in enumerate(overlap_stats["ambiguous_examples"], 1):
                print(f"  {i}. Function: {example['function']}")
                print(f"     Input: {example['inputs']}")
                print(f"     Output: {example['expected_output']}")
                print(
                    f"     Can be solved by: {', '.join(example['solving_functions'][:5])}{'...' if len(example['solving_functions']) > 5 else ''}"
                )
                print()

        if overlap_stats["unsolvable_examples"]:
            print("\n⚠️  Examples of unsolvable pairs (first few):")
            for i, example in enumerate(overlap_stats["unsolvable_examples"], 1):
                print(f"  {i}. Function: {example['function']}")
                print(f"     Input: {example['inputs']}")
                print(f"     Output: {example['expected_output']}")
                print("     No DSL function can solve this!")
                print()

        print("\n✅ Test completed!")
        print("This function keeps only I/O pairs that have exactly one DSL function solution.")
        print(
            f"Removed {overlap_stats['shots_removed']} ambiguous/unsolvable shots out of {overlap_stats['total_shots']} total."
        )

    finally:
        # Clean up
        os.unlink(input_file)


if __name__ == "__main__":
    test_remove_answer_overlap_unique()
