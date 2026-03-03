from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
MATRIX_REL_PATH = Path("flower_research_extension") / "experiments" / "matrix_mnist_10x10.json"


def _base_args() -> list[str]:
    return [
        "--dataset_root",
        "flower_research_extension/data",
        "--num_rounds",
        "10",
        "--num_partitions",
        "10",
        "--fraction_fit",
        "0.25",
        "--min_fit_clients",
        "3",
        "--min_evaluate_clients",
        "3",
        "--batch_size",
        "64",
        "--local_epochs",
        "5",
        "--lr",
        "0.01",
        "--momentum",
        "0.9",
        "--seed",
        "42",
        "--client_cpu",
        "1",
        "--client_gpu",
        "0.01",
        "--csv_log_dir",
        "flower_research_extension/results/logs",
        "--wandb_dir",
        "flower_research_extension/results/wandb",
        "--wandb_project",
        "flower-federated",
        "--wandb_run_name",
        "auto",
        "--disable_wandb",
    ]


SCENARIOS: dict[str, list[str]] = {
    "mnist_iid": [
        "--dataset",
        "mnist",
        "--model",
        "resnet18",
        "--distribution",
        "iid",
    ],
    "mnist_dirichlet": [
        "--dataset",
        "mnist",
        "--model",
        "resnet18",
        "--distribution",
        "dirichlet",
        "--dirichlet_alpha",
        "0.3",
    ],
    "mnist_label_skew": [
        "--dataset",
        "mnist",
        "--model",
        "resnet18",
        "--distribution",
        "label_skew",
        "--label_skew_classes",
        "2",
    ],
    "mnist_shard": [
        "--dataset",
        "mnist",
        "--model",
        "resnet18",
        "--distribution",
        "shard",
        "--shard_num_shards_per_partition",
        "2",
    ],
    "mnist_inner_dirichlet": [
        "--dataset",
        "mnist",
        "--model",
        "resnet18",
        "--distribution",
        "inner_dirichlet",
        "--inner_dirichlet_alpha",
        "0.5",
        "--size_partition_weights",
        "1,1,1,1,1,1,1,1,1,1",
    ],
    "mnist_size": [
        "--dataset",
        "mnist",
        "--model",
        "resnet18",
        "--distribution",
        "size",
        "--size_partition_weights",
        "1,2,3,4,5,6,7,8,9,10",
    ],
    "mnist_distribution": [
        "--dataset",
        "mnist",
        "--model",
        "resnet18",
        "--distribution",
        "distribution",
        "--distribution_matrix_json",
        str(MATRIX_REL_PATH),
    ],
    "cifar10_iid": [
        "--dataset",
        "cifar10",
        "--model",
        "resnet18",
        "--distribution",
        "iid",
    ],
    "svhn_dirichlet": [
        "--dataset",
        "svhn",
        "--model",
        "mobilenet_v2",
        "--distribution",
        "dirichlet",
        "--dirichlet_alpha",
        "0.3",
    ],
    "cifar100_iid": [
        "--dataset",
        "cifar100",
        "--model",
        "densenet121",
        "--distribution",
        "iid",
    ],
}


def _ensure_distribution_matrix() -> None:
    matrix_path = REPO_ROOT / MATRIX_REL_PATH
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    if matrix_path.exists():
        return
    matrix = [[1.0] * 10 for _ in range(10)]
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")


def _build_command(scenario: str, dry_run: bool) -> list[str]:
    if scenario == "mnist_distribution":
        _ensure_distribution_matrix()
    cmd = [sys.executable, "-m", "flower_research_extension.experiments.run_experiment"]
    cmd.extend(_base_args())
    cmd.extend(SCENARIOS[scenario])
    if dry_run:
        cmd.append("--dry_run")
    return cmd


def _run_command(cmd: list[str], print_only: bool) -> None:
    pretty = " ".join(cmd)
    print(pretty, flush=True)
    if print_only:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _list_scenarios() -> None:
    print("Available scenarios:")
    for name in SCENARIOS:
        print(f"- {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run predefined important experiment commands.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    parser.add_argument("--scenario", type=str, default="", help="Scenario name to execute.")
    parser.add_argument("--all", action="store_true", help="Run all predefined scenarios.")
    parser.add_argument("--dry_run", action="store_true", help="Append --dry_run to each command.")
    parser.add_argument("--print_only", action="store_true", help="Print commands without executing.")

    args = parser.parse_args(argv)

    if args.list:
        _list_scenarios()
        return 0

    if args.all:
        for scenario in SCENARIOS:
            cmd = _build_command(scenario, dry_run=args.dry_run)
            _run_command(cmd, print_only=args.print_only)
        return 0

    if not args.scenario:
        parser.error("Provide --scenario <name>, or use --all, or --list")
    if args.scenario not in SCENARIOS:
        parser.error(f"Unknown scenario '{args.scenario}'. Use --list to see options.")

    cmd = _build_command(args.scenario, dry_run=args.dry_run)
    _run_command(cmd, print_only=args.print_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
