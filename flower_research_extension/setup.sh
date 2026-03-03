#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_ROOT="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$EXT_ROOT/.venv"

echo "Detected OS: $(uname -s)"
echo "Extension root: $EXT_ROOT"

if [[ "$(uname -s)" == "Windows_NT" ]]; then
  echo "Use setup.ps1 on Windows."
  exit 1
fi

echo "Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ -d "$REPO_ROOT/framework" ]]; then
  echo "Installing local Flower framework from $REPO_ROOT/framework"
  python -m pip install -e "$REPO_ROOT/framework[simulation]"
else
  echo "Installing Flower from PyPI"
  python -m pip install "flwr[simulation]>=1.5.0"
fi

python -m pip install -r "$EXT_ROOT/requirements.txt"
python -m pip install -e "$EXT_ROOT"

echo "Setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run with: python -m flower_research_extension.experiments.run_experiment"

