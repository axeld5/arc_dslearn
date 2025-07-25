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

## Setup

### Prerequisites

- Python ≥3.11
- [UV](https://github.com/astral-sh/uv) - Ultra-fast Python package manager

### Installation

1. **Install UV** (if not already installed):
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or with pip
   pip install uv
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/axeld5/arc_dslearn.git
   cd arc_dslearn
   ```

3. **Install the project and dependencies**:
   ```bash
   # Install the project in development mode with all dependencies
   uv sync
   
   # Or install with development dependencies
   uv sync --extra dev
   ```

4. **Activate the virtual environment**:
   ```bash
   # UV automatically creates and manages a virtual environment
   # To run commands in the environment, use:
   uv run <command>
   
   # Or activate the environment manually
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

## Usage

All scripts should be executed using `uv run` to ensure proper environment and dependency management.

### Core Scripts

1. **Generate Training Data**: 
   ```bash
   uv run python -m src.arc_dslearn.data_gene.pilot
   ```

2. **SFT Training**: 
   ```bash
   uv run torchrun --nproc_per_node <n_gpus> -m arc_dslearn.model_tuning.finetuning_script
   ```

3. **RL Training**: 
   ```bash
   uv run torchrun --nproc_per_node <n_gpus> -m arc_dslearn.model_tuning.rl_script
   ```

4. **Model Evaluation**: 
   ```bash
   uv run python src/arc_dslearn/model_eval/evaluate_main.py
   ```

5. **Run Tests**: 
   ```bash
   uv run pytest
   ```

### Important Notes

- **Module Execution**: Some scripts (like the pilot script) should be run as modules using `python -m` to ensure proper import resolution
- **Environment**: Always use `uv run` to execute scripts to maintain consistent dependency management
- **GPU Requirements**: Training scripts require CUDA-compatible GPU with 16GB+ memory

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
