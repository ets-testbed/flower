# Flower Research Extension

Research-oriented extension for [Flower](https://flower.dev) with:
- plugin-based metrics logging,
- strategy wrappers,
- dataset-provider registry,
- configurable experiment runner.

## Project Layout

```
flower_research_extension/
├── client.py
├── model.py
├── training.py
├── data_files/
│   ├── base.py
│   ├── registry.py
│   ├── *_provider.py
│   └── cifar10.py
├── strategies/
├── plugins/
├── experiments/
│   ├── run_experiment.py        # minimal entrypoint
│   ├── run_experiment_cli.py    # parser/validation/normalization
│   ├── catalog.py               # model/dataset/distribution metadata
│   ├── run_commands.py          # direct scenario runner
│   └── RUN_COMMANDS.md          # scenario command guide
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── setup.sh
├── setup.ps1
└── README_CHANGES.md
```

## Setup

From `C:\pycharm\flower\flower_research_extension` (or equivalent path on Linux):

Linux/macOS:
```bash
bash setup.sh
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

Both setup scripts:
- create `.venv`,
- install local Flower framework in editable mode when `../framework` exists (otherwise install from PyPI),
- install extension requirements,
- install this package in editable mode.

## Run

`run_experiment` supports two invocation modes:
- From `flower_research_extension/`: `py -m experiments.run_experiment ...`
- From repo root (`C:\pycharm\flower`): `py -m flower_research_extension.experiments.run_experiment ...`

Note: use `py` on Windows PowerShell and `python` on Linux/macOS shells.

Quick smoke check (from `flower_research_extension/`):

```powershell
py -m experiments.run_experiment --dry_run --disable_wandb
```

Equivalent smoke check (from repo root):

```bash
py -m flower_research_extension.experiments.run_experiment --dry_run --disable_wandb
```

Copy-paste command catalog:
- `experiments/RUN_COMMANDS.md`
- Direct runner:
  - from `flower_research_extension/`: `py -m experiments.run_commands --list`
  - from repo root: `py -m flower_research_extension.experiments.run_commands --list`

Hyperparameter sweep script (Linux/macOS bash):
- `bash flower_research_extension/experiments/hyperparam_runs.sh --dry-run`
- `bash flower_research_extension/experiments/hyperparam_runs.sh --only medium --dry-run -- --dataset cifar10 --model resnet18`

Example:
```bash
py -m experiments.run_experiment --dataset mnist --model resnet18 --num_rounds 5
```

Print resolved config without running:
```bash
py -m experiments.run_experiment --dry_run
```

Disable W&B and tune local optimizer settings:
```bash
py -m experiments.run_experiment --disable_wandb --local_epochs 3 --lr 0.005 --momentum 0.8
```

With W&B enabled, the run now captures an expanded start-of-run configuration payload, including:
- resolved CLI args
- dataset/model/distribution setup
- optimizer and federated settings
- client resource settings and runtime environment metadata.

Without W&B, the CSV logger now stores the same experiment context and round-level results under each run folder:
- `run_config.json`: full resolved run configuration snapshot
- `round_metrics.jsonl`: chronological fit/eval/failure events per round
- `run_summary.json`: final summary with key counters, last metrics, and artifact paths
- `global_metrics_*.csv` and `client_metrics_*.csv`: compact tabular metrics.

Use dataset-specific automatic model selection:
```bash
py -m experiments.run_experiment --dataset cifar100 --model auto
```

Model resource profile metadata (informational):
- `light`: `net`, `mobilenet_v2`, `shufflenet_v2_x1_0`, `squeezenet1_1`
- `medium`: `resnet18`, `resnet34`, `densenet121`, `efficientnet_b0`
- `heavy`: `resnet50`, `resnext50_32x4d`, `wide_resnet50_2`, `convnext_tiny`

List all supported datasets, distributions, models, and dataset-model policies:
```bash
py -m experiments.run_experiment --list_capabilities
```

Dataset-model compatibility policy:
- `mnist`: default `net`, allowed `net,resnet18,resnet34,mobilenet_v2,shufflenet_v2_x1_0,squeezenet1_1`
- `fashionmnist`: default `net`, allowed `net,resnet18,resnet34,mobilenet_v2,shufflenet_v2_x1_0,squeezenet1_1`
- `emnist_balanced`: default `resnet18`, allowed `net,resnet18,resnet34,mobilenet_v2,shufflenet_v2_x1_0,squeezenet1_1,densenet121,efficientnet_b0`
- `cifar10`: default `resnet18`, allowed `net,resnet18,resnet34,mobilenet_v2,shufflenet_v2_x1_0,squeezenet1_1,densenet121,efficientnet_b0,resnet50,resnext50_32x4d,wide_resnet50_2,convnext_tiny`
- `svhn`: default `resnet18`, allowed `net,resnet18,resnet34,mobilenet_v2,shufflenet_v2_x1_0,squeezenet1_1,densenet121,efficientnet_b0,resnet50,resnext50_32x4d,wide_resnet50_2,convnext_tiny`
- `cifar100`: default `densenet121`, allowed `resnet18,resnet34,mobilenet_v2,shufflenet_v2_x1_0,squeezenet1_1,densenet121,efficientnet_b0,resnet50,resnext50_32x4d,wide_resnet50_2,convnext_tiny`

Choose client data distribution:
```bash
py -m experiments.run_experiment --distribution dirichlet --dirichlet_alpha 0.3
py -m experiments.run_experiment --distribution label_skew --label_skew_classes 2
py -m experiments.run_experiment --distribution shard --shard_num_shards_per_partition 2
py -m experiments.run_experiment --distribution size --size_partition_weights "1,2,3,4,5,6,7,8,9,10"
py -m experiments.run_experiment --distribution inner_dirichlet --inner_dirichlet_alpha 0.5 --size_partition_weights "1,1,1,1,1,1,1,1,1,1"
py -m experiments.run_experiment --distribution distribution --distribution_matrix_json partition_matrix.json
```

Distribution quick guide:
- `iid`: uniform random split, closest to classical balanced FL.
- `dirichlet`: random class proportions per client, controlled by `--dirichlet_alpha` (`smaller => more skew`).
- `inner_dirichlet`: similar to Dirichlet but also biased by client size weights.
- `distribution`: explicit class-probability matrix per client (most controlled/custom mode).
- `label_skew`: each client only sees a small subset of classes.
- `pathological`: alias of `label_skew` for standard pathological non-IID studies.
- `shard`: label-sorted shards assigned to clients; creates sharp class concentration.
- `linear`: client sizes grow linearly from client 0 to last client.
- `square`: client sizes grow quadratically, stronger imbalance than linear.
- `exponential`: client sizes grow exponentially, strongest built-in size imbalance.
- `size`: manual client-size weights through `--size_partition_weights`.

Windows simulation notes:
- Flower currently prints a Ray experimental-support warning on Windows; this is expected.
- You might also see Ray/PyTorch deprecation warnings in client logs; these do not stop training.
- A real failure will end with a Python traceback and non-zero exit status.

`distribution_matrix_json` notes:
- matrix rows must equal `--num_partitions`
- matrix columns must be at least the dataset class count (for example, at least 10 columns for `cifar10`)

## Reproducibility

The extension now applies deterministic seeding at:
- experiment startup,
- per-client startup,
- data partition split and data-loader workers.

Use `--seed` in `run_experiment.py` to control run-to-run determinism.

## Plugin Extension

1. Subclass `MetricsPlugin` in `plugins/base.py`.
2. Implement needed hooks (`on_client_result`, `on_round_end`, `on_server_evaluate`, etc.).
3. Register plugin creation in `experiments/experiment_setup.py`.
