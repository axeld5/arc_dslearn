#!/bin/bash

# ARC DSL Learning Setup and Training Script
# Usage: ./setup_and_train.sh <n_gpus>

set -e  # Exit on any error

# Check if number of GPUs is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <n_gpus>"
    echo "Example: $0 2"
    exit 1
fi

N_GPUS=$1

# Validate that n_gpus is a positive integer
if ! [[ "$N_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: n_gpus must be a positive integer"
    exit 1
fi

echo "=================================================="
echo "ARC DSL Learning Setup and Training"
echo "Number of GPUs: $N_GPUS"
echo "=================================================="

# Step 1: Install astral-uv
echo "Step 1: Installing astral-uv..."
sudo snap install astral-uv --classic
echo "✓ astral-uv installed successfully"

# Step 2: Sync dependencies
echo "Step 2: Syncing dependencies..."
uv sync
echo "✓ Dependencies synced successfully"

# Step 3: Generate training data
echo "Step 3: Generating training data..."
uv run python -m src.arc_dslearn.data_gene.pilot
echo "✓ Training data generated successfully"

# Step 4: Run fine-tuning
echo "Step 4: Starting fine-tuning with $N_GPUS GPUs..."
uv run torchrun --nproc_per_node $N_GPUS --standalone src/arc_dslearn/model_tuning/finetuning_script.py
echo "✓ Fine-tuning completed successfully"

echo "=================================================="
echo "Setup and training completed successfully!"
echo "==================================================" 