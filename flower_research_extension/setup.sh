#!/bin/bash
set -e

# Step 1: Go to the root of the repository (2 levels up from current file location)
cd "$(dirname "$0")/../"

echo "Current working directory: $(pwd)"

# Step 2: Set up virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing core Flower simulation dependencies..."
pip install --upgrade pip
pip install -e framework[simulation]
pip install -e ./datasets

echo "Installing PyTorch (CUDA 12.1)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Installing other dependencies..."
pip install wandb scikit-learn

#echo "Running experiment..."
#python -m flower_research_extension.experiments.run_experiment
