# Flower Research Extension

Research-oriented extension for [Flower](https://flower.dev) with:

- provider-based dataset loading
- configurable experiment runs from CLI or YAML
- IID and non-IID client partitioning
- model selection policies per dataset
- CSV and W&B logging plugins
- smoke/validation helpers for local checks

## Project Layout

```text
flower_research_extension/
├── client.py
├── model.py
├── training.py
├── data_files/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   └── *_provider.py
├── experiments/
│   ├── catalog.py
│   ├── example.yaml
│   ├── experiment_setup.py
│   ├── hyperparam_runs.sh
│   ├── run_commands.py
│   ├── run_experiment.py
│   ├── run_experiment_cli.py
│   ├── validation_suite.py
│   └── RUN_COMMANDS.md
├── plugins/
│   ├── base.py
│   ├── csv_logger.py
│   └── wandb_logger.py
├── strategies/
│   ├── custom_fedavg.py
│   ├── hooked_strategy.py
│   └── round_timer.py
└── tests/
```

## Setup

From the extension directory on Linux/macOS:

```bash
bash setup.sh
source .venv/bin/activate
```

`setup.sh`:

- creates `.venv`
- installs the local Flower framework in editable mode when `../framework` exists
- otherwise installs Flower from PyPI
- installs `requirements.txt`
- installs `flower_research_extension` in editable mode

On Windows, use `setup.ps1`.

## Running Experiments

From the extension directory:

```bash
python -m experiments.run_experiment --dry_run --disable_wandb
```

From the repository root:

```bash
python -m flower_research_extension.experiments.run_experiment --dry_run --disable_wandb
```

Useful commands:

- Print resolved configuration:
  ```bash
  python -m experiments.run_experiment --dry_run
  ```
- List datasets, distributions, models, and policies:
  ```bash
  python -m experiments.run_experiment --list_capabilities
  ```
- Run with YAML config:
  ```bash
  python -m experiments.run_experiment --config experiments/example.yaml
  ```
- Override config values from CLI:
  ```bash
  python -m experiments.run_experiment --config experiments/example.yaml --lr 0.01 --disable_wandb
  ```

## Configuration

YAML config files are supported through `--config`. CLI flags override YAML values.

Example fields in `experiments/example.yaml`:

- dataset/model selection
- federated settings such as rounds, partitions, and fit fraction
- optimizer settings such as `local_epochs`, `lr`, and `momentum`
- distribution settings such as `distribution` and `dirichlet_alpha`
- logging destinations and W&B toggles

## Distributions

Supported client distributions include:

- `iid`
- `dirichlet`
- `label_skew`
- `pathological`
- `shard`
- `linear`
- `square`
- `exponential`
- `size`
- `inner_dirichlet`
- `distribution`

Examples:

```bash
python -m experiments.run_experiment --distribution dirichlet --dirichlet_alpha 0.3
python -m experiments.run_experiment --distribution shard --shard_num_shards_per_partition 2
python -m experiments.run_experiment --distribution size --size_partition_weights "1,2,3,4,5,6,7,8,9,10"
```

## Logging

Each run folder under `results/logs/` contains:

- `round_metrics_<timestamp>.csv`
- `client_metrics_<timestamp>.csv`
- `run_config.json`
- `round_metrics.jsonl`
- `run_summary.json`

Round CSVs include dynamic fit/eval timing fields, including metrics such as:

- `fit_elapsed_time`
- `fit_aggregation_time`
- `fit_client_count`
- `fit_failure_count`
- `server_eval_time`
- `fit_to_eval_gap_time`
- `round_total_time`

When W&B is enabled, the logger records the resolved run configuration plus fit/eval metrics with explicit round-based steps.

## Validation And Tests

Quick local test run:

```bash
python -m unittest discover -s tests -v
```

Validation suite examples:

```bash
python -m experiments.validation_suite --mode smoke --num_rounds 1
python -m experiments.validation_suite --mode dry-only
```

Additional runnable scenarios are listed in `experiments/RUN_COMMANDS.md`.
