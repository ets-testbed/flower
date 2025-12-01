import argparse
import re

from flower_research_extension.experiments.experiment_setup import run_experiment
from flower_research_extension.data_files import REGISTRY as _REGISTRY


def _sanitize(name: str) -> str:
    # keep letters, numbers, dash, underscore, dot
    name = name.strip()
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", name)


def _auto_run_name(dataset: str, num_rounds: int, num_partitions: int, batch_size: int, fraction_fit: float) -> str:
    base = f"{dataset}_fedavg"
    # keep compact, readable; fraction_fit with up to 2 decimals
    ff = f"{fraction_fit:.2f}".rstrip("0").rstrip(".")
    suffix = f"r{num_rounds}_C{num_partitions}_b{batch_size}_ff{ff}"
    return _sanitize(f"{base}_{suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a Flower federated learning experiment with configurable parameters."
    )
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=_REGISTRY.available(),
                        help="Dataset to use")
    parser.add_argument("--dataset_root", type=str, default="data",
                        help="Dataset root directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per client")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for partitioning")

    parser.add_argument("--num_rounds", type=int, default=10, help="Total number of federated rounds")
    parser.add_argument("--num_partitions", type=int, default=10, help="Number of simulated clients")
    parser.add_argument("--fraction_fit", type=float, default=0.25, help="Fraction of clients used for training each round")
    parser.add_argument("--min_fit_clients", type=int, default=3, help="Minimum number of clients to sample for training")
    parser.add_argument("--min_evaluate_clients", type=int, default=3,
                        help="Minimum number of clients to sample for evaluation")
    parser.add_argument("--client_cpu", type=int, default=1, help="Number of CPUs per client for simulation backend")
    parser.add_argument("--client_gpu", type=float, default=0.01,
                        help="Fraction of one GPU per client for simulation backend")
    parser.add_argument("--csv_log_dir", type=str, default="results/logs", help="Directory for CSV logs")
    parser.add_argument("--wandb_dir", type=str, default="results/wandb", help="Directory for Weights & Biases logs")
    parser.add_argument("--wandb_project", type=str, default="flower-federated", help="Weights & Biases project name")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="auto",
        help="Weights & Biases run name. Use 'auto' to name by dataset/rounds/clients/batch/fraction_fit.",
    )

    args = parser.parse_args()

    # Adaptive W&B run name
    if args.wandb_run_name in {"auto", "", None}:
        args.wandb_run_name = _auto_run_name(
            dataset=args.dataset,
            num_rounds=args.num_rounds,
            num_partitions=args.num_partitions,
            batch_size=args.batch_size,
            fraction_fit=args.fraction_fit,
        )

    run_experiment(args)


if __name__ == "__main__":
    main()
