import argparse

from flower_research_extension.experiments.experiment_setup import run_experiment

# ─── Argument Parser ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run a Flower federated learning experiment with configurable parameters."
)
parser.add_argument("--num_rounds", type=int, default=100, help="Total number of federated rounds")
parser.add_argument("--num_partitions", type=int, default=20, help="Number of simulated clients")
parser.add_argument("--fraction_fit", type=float, default=0.25, help="Fraction of clients used for training each round")
parser.add_argument("--min_fit_clients", type=int, default=3, help="Minimum number of clients to sample for training")
parser.add_argument("--min_evaluate_clients", type=int, default=3,
                    help="Minimum number of clients to sample for evaluation")
parser.add_argument("--client_cpu", type=int, default=2, help="Number of CPUs per client for simulation backend")
parser.add_argument("--client_gpu", type=float, default=0.01,
                    help="Fraction of one GPU per client for simulation backend")
parser.add_argument("--csv_log_dir", type=str, default="results/logs", help="Directory for CSV logs")
parser.add_argument("--wandb_dir", type=str, default="results/wandb", help="Directory for Weights & Biases logs")
parser.add_argument("--wandb_project", type=str, default="flower-federated", help="Weights & Biases project name")
parser.add_argument("--wandb_run_name", type=str, default="cifar10_fedavg", help="Weights & Biases run name")
args = parser.parse_args()

run_experiment(args)
