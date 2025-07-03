#!/usr/bin/env bash
# hyperparam_runs.sh

# Path to the Python interpreter in your virtual environment
# Go to the root of the project
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$(pwd)
VENV_PYTHON="$HOME/pycharm/flower/venv/bin/python"


# Ensure the venv Python exists
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Error: Python interpreter not found at $VENV_PYTHON"
  exit 1
fi

# Common logging args (handled internally by run_experiment.py)
COMMON_ARGS=(
  --wandb_dir "results/wandb"
  --csv_log_dir "results/logs"
  --wandb_project "flower-federated"
)

# Run 1: low fraction
"$VENV_PYTHON" ./flower_research_extension/experiments/run_experiment.py \
  --fraction_fit 0.1 \
  --min_fit_clients 2 \
  --num_rounds 10 \
  --wandb_run_name "sweep_frac0.1" \
  "${COMMON_ARGS[@]}"

# Run 2: medium fraction
"$VENV_PYTHON" ./flower_research_extension/experiments/run_experiment.py \
  --fraction_fit 0.25 \
  --min_fit_clients 3 \
  --num_rounds 15 \
  --wandb_run_name "sweep_frac0.25" \
  "${COMMON_ARGS[@]}"

# Run 3: high fraction
"$VENV_PYTHON" ./flower_research_extension/experiments/run_experiment.py \
  --fraction_fit 0.5 \
  --min_fit_clients 5 \
  --num_rounds 20 \
  --wandb_run_name "sweep_frac0.5" \
  "${COMMON_ARGS[@]}"

echo "All runs launched "
