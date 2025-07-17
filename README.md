![CI](https://github.com/axeld5/arc_dslearn/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/arc-dslearn?color=blue)

# ARC DSL Learning

Training coder models to learn a Domain-Specific Language (DSL) for solving ARC (Abstraction and Reasoning Corpus) problems.

## Overview

This project implements a comprehensive DSL for ARC puzzle solving and trains Qwen2.5-Coder-1.5B models using supervised fine-tuning (SFT) and reinforcement learning (RL) approaches.

## Architecture

- **Base Model**: Qwen2.5-Coder-1.5B-Instruct
- **Fine-tuning**: LoRA adapters
- **DSL**: 100+ functions for grid operations, object manipulation, mathematical operations, and logical reasoning. DSL was made by Michael Hodel and taken from the https://github.com/michaelhodel/arc-dsl repo.
- **Training**: SFT followed by GRPO (Group Relative Policy Optimization) for RL

## Performance (Functional Accuracy on 274 eval tasks)

| Model | Accuracy |
|-------|----------|
| Base  | 0.7%     |
| SFT   | 77.4%    |
| RL    | 73.7%    |

## Key Components

- `arc_dsl/`: Domain-specific language implementation
  - `dsl.py`: Core DSL functions (identity, arithmetic, grid operations, etc.)
  - `arc_types.py`: Type definitions for grids, objects, patches
  - `solvers.py`: Problem-solving utilities
- `finetuning_script.py`: Supervised fine-tuning with LoRA
- `rl_script.py`: Reinforcement learning with GRPO on the SFT'd model
- `evaluate_models.py`: Model evaluation and accuracy measurement
- `reward_fn.py`: Reward function for RL training

## Usage

1. **Install dependencies**: `pip install -r requirements.txt`
2. **SFT Training**: `python finetuning_script.py`
3. **RL Training**: `python rl_script.py`
4. **Evaluation**: `python evaluate_models.py`

## Next Steps

- **Increase Datapoint Information**: Right now grids are restricted to at most 5x5, which is not realistic given ARC standards. They need to be much higher.
- **Function Composition**: Implement higher-order function combinations
- **Model Scaling**: Experiment with 7B parameter models
- **Advanced Features**: Additional capabilities and optimizations

## Requirements

- Python ≥3.11
- PyTorch, Transformers, PEFT
- HuggingFace Hub access
- 16GB+ GPU memory recommended

## License

This project is under the MIT License.
