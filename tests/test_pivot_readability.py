"""Tests for dataset readability as HuggingFace Dataset.

This module verifies that the generated training data can be properly
loaded and read using the HuggingFace datasets library.
"""

import json
import os
import sys
import tempfile

# Add the parent directory to the path so we can import arc_dslearn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arc_dslearn.data_gene.pilot import main


def test_data_generation():
    """Test that we can generate training blocks."""
    training_blocks = main(generation_seed=42)

    assert len(training_blocks) > 0, "Should generate at least one training block"
    assert isinstance(training_blocks, list), "Should return a list"
    assert all(
        isinstance(block, dict) for block in training_blocks
    ), "All blocks should be dictionaries"


def test_dataset_structure():
    """Test that dataset entries have expected structure."""
    training_blocks = main(generation_seed=42)

    if training_blocks:
        sample_block = training_blocks[0]
        assert "shots" in sample_block, "Block should have 'shots' field"
        assert "inputs" in sample_block["shots"][0], "Block should have 'inputs' field"
        assert "output" in sample_block["shots"][0], "Block should have 'output' field"
        assert "name" in sample_block, "Block should have 'name' field"
        assert "system_prompt" in sample_block, "Block should have 'system_prompt' field"
        assert "user_prompt" in sample_block, "Block should have 'user_prompt' field"
        assert "assistant_prompt" in sample_block, "Block should have 'assistant_prompt' field"


def test_huggingface_dataset_loading():
    """Test that generated data can be loaded as HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        import pytest

        pytest.skip("datasets library not available")

    training_blocks = main(generation_seed=42)
    assert len(training_blocks) > 0, "Need training blocks to test dataset loading"

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save training blocks to temporary file
        train_file = os.path.join(tmpdir, "train_set.json")
        with open(train_file, "w") as f:
            for record in training_blocks:
                if record["shots"]:
                    for shot in record["shots"]:
                        shot["inputs"] = json.dumps(shot["inputs"])
                        shot["output"] = json.dumps(shot["output"])
            with open(train_file, "w") as f:
                json.dump(training_blocks, f, indent=2)

        # Test that the JSON can be loaded as a dataset
        dataset = load_dataset("json", data_files=train_file, split="train")
        assert len(dataset) > 0, "Dataset should contain entries"
