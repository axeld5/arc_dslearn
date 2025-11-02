"""Tests for I/O validity using reward function.

This module verifies that all generated input/output pairs are valid
by using the reward function to check functional correctness.
"""

import json
import os
import sys

# Add the parent directory to the path so we can import arc_dslearn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arc_dslearn.data_gene_unary.pilot import main
from src.arc_dslearn.metrics_and_rewards.reward_fn import reward_function


def test_generated_blocks_structure() -> None:
    """Test that generated blocks have valid I/O structure."""
    training_blocks = main(generation_seed=42)
    assert len(training_blocks) > 0, "Should generate training blocks"

    sample_block = training_blocks[0]
    assert "shots" in sample_block, "Block should have 'shots' field"
    assert "inputs" in sample_block["shots"][0], "Block should have 'inputs' field"
    assert "output" in sample_block["shots"][0], "Block should have 'output' field"
    assert "name" in sample_block, "Block should have 'name' field"
    assert "system_prompt" in sample_block, "Block should have 'system_prompt' field"
    assert "user_prompt" in sample_block, "Block should have 'user_prompt' field"
    assert "assistant_prompt" in sample_block, "Block should have 'assistant_prompt' field"


def test_reward_function_compatibility() -> None:
    """Test reward function can process generated data."""
    training_blocks = main(generation_seed=42)
    assert len(training_blocks) > 0, "Need training blocks to test"

    # Take a small sample and test with reward function
    sample_blocks = training_blocks[:3]
    completions = [block.get("assistant_prompt", "") for block in sample_blocks]

    # Format shots for reward function
    shots = []
    for block in sample_blocks:
        block_shots = block.get("shots")
        if block_shots and len(block_shots) > 0:
            first_shot = block_shots[0]
            shot = [
                {
                    "inputs": json.dumps(first_shot["inputs"])
                    if isinstance(first_shot.get("inputs"), dict)
                    else str(first_shot.get("inputs", "{}")),
                    "output": json.dumps(first_shot["output"])
                    if isinstance(first_shot.get("output"), dict)
                    else str(first_shot.get("output", "{}")),
                }
            ]
            shots.append(shot)

    # Test that reward function runs without errors
    rewards = reward_function(completions, shots)

    assert isinstance(rewards, list), "Reward function should return a list"
    assert len(rewards) == len(completions), "Should return one reward per completion"
    assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"


def test_positive_rewards_achievable() -> None:
    """Test that some generated code achieves positive rewards."""
    training_blocks = main(generation_seed=42)
    assert len(training_blocks) > 0, "Need training blocks to test"

    # Test with more samples to find some good ones
    sample_blocks = training_blocks[: min(10, len(training_blocks))]
    completions = [block.get("assistant_prompt", "") for block in sample_blocks]

    shots = []
    for block in sample_blocks:
        block_shots = block.get("shots")
        if block_shots and len(block_shots) > 0:
            first_shot = block_shots[0]
            shot = [
                {
                    "inputs": json.dumps(first_shot["inputs"])
                    if isinstance(first_shot.get("inputs"), dict)
                    else str(first_shot.get("inputs", "{}")),
                    "output": json.dumps(first_shot["output"])
                    if isinstance(first_shot.get("output"), dict)
                    else str(first_shot.get("output", "{}")),
                }
            ]
            shots.append(shot)

    rewards = reward_function(completions, shots)

    # At least some rewards should be positive
    positive_rewards = [r for r in rewards if r > 0]
    assert len(positive_rewards) > 0, f"Expected some positive rewards, got: {rewards}"

    # Check that the best reward is reasonable
    max_reward = max(rewards)
    assert max_reward >= 0.1, f"Expected maximum reward >= 0.1, got {max_reward}"
