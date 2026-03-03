#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXT_ROOT/.." && pwd)"

usage() {
  cat <<'EOF'
Run a small hyperparameter sweep over client participation fraction.

Usage:
  bash flower_research_extension/experiments/hyperparam_runs.sh [options] [-- extra run_experiment args]

Options:
  --dry-run        Append --dry_run to each run.
  --with-wandb     Enable W&B logging (default: disabled).
  --only NAME      Run a single sweep case: low | medium | high.
  -h, --help       Show this help.

Examples:
  bash flower_research_extension/experiments/hyperparam_runs.sh --dry-run
  bash flower_research_extension/experiments/hyperparam_runs.sh --only medium --dry-run
  bash flower_research_extension/experiments/hyperparam_runs.sh --dry-run -- --dataset cifar10 --model resnet18
EOF
}

detect_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi

  if [[ -x "$EXT_ROOT/.venv/bin/python" ]]; then
    echo "$EXT_ROOT/.venv/bin/python"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi

  echo "Error: no Python interpreter found (set PYTHON_BIN to override)." >&2
  exit 1
}

print_cmd() {
  local quoted=()
  for arg in "$@"; do
    quoted+=("$(printf '%q' "$arg")")
  done
  printf '%s\n' "${quoted[*]}"
}

DRY_RUN=0
WITH_WANDB=0
ONLY_CASE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --with-wandb)
      WITH_WANDB=1
      shift
      ;;
    --only)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --only requires one of: low|medium|high" >&2
        exit 1
      fi
      ONLY_CASE="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$ONLY_CASE" && "$ONLY_CASE" != "low" && "$ONLY_CASE" != "medium" && "$ONLY_CASE" != "high" ]]; then
  echo "Error: --only must be one of: low | medium | high" >&2
  exit 1
fi

PYTHON_BIN="$(detect_python)"
echo "Using Python: $PYTHON_BIN"
echo "Repo root: $REPO_ROOT"

BASE_CMD=(
  "$PYTHON_BIN"
  -m
  flower_research_extension.experiments.run_experiment
  --dataset_root
  flower_research_extension/data
  --csv_log_dir
  flower_research_extension/results/logs
  --wandb_dir
  flower_research_extension/results/wandb
  --wandb_project
  flower-federated
  --num_partitions
  10
  --batch_size
  64
  --local_epochs
  5
  --lr
  0.01
  --momentum
  0.9
  --seed
  42
  --client_cpu
  1
  --client_gpu
  0.01
)

run_case() {
  local name="$1"
  local fraction="$2"
  local min_fit="$3"
  local rounds="$4"

  local run_name="sweep_${name}_ff${fraction}"
  local cmd=(
    "${BASE_CMD[@]}"
    --fraction_fit "$fraction"
    --min_fit_clients "$min_fit"
    --min_evaluate_clients "$min_fit"
    --num_rounds "$rounds"
    --wandb_run_name "$run_name"
  )

  if [[ "$WITH_WANDB" -eq 0 ]]; then
    cmd+=(--disable_wandb)
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    cmd+=(--dry_run)
  fi

  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo
  echo "=== Running case: $name (fraction_fit=$fraction, min_fit_clients=$min_fit, num_rounds=$rounds) ==="
  print_cmd "${cmd[@]}"
  (cd "$REPO_ROOT" && "${cmd[@]}")
}

if [[ -z "$ONLY_CASE" || "$ONLY_CASE" == "low" ]]; then
  run_case "low" "0.1" "2" "10"
fi
if [[ -z "$ONLY_CASE" || "$ONLY_CASE" == "medium" ]]; then
  run_case "medium" "0.25" "3" "15"
fi
if [[ -z "$ONLY_CASE" || "$ONLY_CASE" == "high" ]]; then
  run_case "high" "0.5" "5" "20"
fi

echo
echo "Sweep completed."
