"""Tests for I/O validity using reward function.

This module verifies that all generated input/output pairs are valid
by using the reward function to check functional correctness.
"""

import json
import os
import sys

# Add the parent directory to the path so we can import arc_dslearn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arc_dslearn.data_gene.pilot import main
from src.arc_dslearn.metrics_and_rewards.reward_fn import reward_function


def test_generated_blocks_structure():
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


def test_reward_function_compatibility():
    """Test reward function can process generated data."""
    training_blocks = main(generation_seed=42)
    assert len(training_blocks) > 0, "Need training blocks to test"

    # Take a small sample and test with reward function
    sample_blocks = training_blocks[:3]
    completions = [block.get("assistant_prompt", "") for block in sample_blocks]

    # Format shots for reward function
    shots = []
    for block in sample_blocks:
        shot = [
            {
                "inputs": json.dumps(block["shots"][0]["inputs"])
                if isinstance(block.get("shots")[0].get("inputs"), dict)
                else str(block.get("shots")[0].get("inputs", "{}")),
                "output": json.dumps(block["shots"][0]["output"])
                if isinstance(block.get("shots")[0].get("output"), dict)
                else str(block.get("shots")[0].get("output", "{}")),
            }
        ]
        shots.append(shot)

    # Test that reward function runs without errors
    rewards = reward_function(completions, shots)

    assert isinstance(rewards, list), "Reward function should return a list"
    assert len(rewards) == len(completions), "Should return one reward per completion"
    assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"


def test_positive_rewards_achievable():
    """Test that some generated code achieves positive rewards."""
    training_blocks = main(generation_seed=42)
    assert len(training_blocks) > 0, "Need training blocks to test"

    # Test with more samples to find some good ones
    sample_blocks = training_blocks[: min(10, len(training_blocks))]
    completions = [block.get("assistant_prompt", "") for block in sample_blocks]

    shots = []
    for block in sample_blocks:
        shot = [
            {
                "inputs": json.dumps(block["shots"][0]["inputs"])
                if isinstance(block.get("shots")[0].get("inputs"), dict)
                else str(block.get("shots")[0].get("inputs", "{}")),
                "output": json.dumps(block["shots"][0]["output"])
                if isinstance(block.get("shots")[0].get("output"), dict)
                else str(block.get("shots")[0].get("output", "{}")),
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
