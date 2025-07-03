import warnings
import datasets
from datasets import logging as hf_logging
import torch
from typing import List, Tuple, Dict

from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.client import ClientApp
from flwr.common import ndarrays_to_parameters, Context
from flwr.server.strategy import FedAvg

from flower_research_extension.strategies.hooked_strategy import HookedStrategy
from flower_research_extension.strategies.round_timer import RoundTimerStrategy
from flower_research_extension.plugins.wandb_logger import WandBLogger
from flower_research_extension.plugins.csv_logger import CSVLogger
from flower_research_extension.model import Net, get_parameters
from flower_research_extension.client import client_fn
from flower_research_extension.training import evaluate, fit_config


# ─── Suppress Specific Warnings ────────────────────────────────────────────────
def suppress_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="datasets")
    datasets.logging.set_verbosity_error()
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# ─── Aggregation Functions ────────────────────────────────────────────────────
def aggregate_fit_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    return {
        "accuracy": sum(m["accuracy"] * n for n, m in metrics if "accuracy" in m) / total,
        "loss": sum(m["loss"] * n for n, m in metrics if "loss" in m) / total,
    }


def aggregate_evaluate_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    keys = metrics[0][1].keys()
    return {k: sum(m[k] * n for n, m in metrics if k in m) / total for k in keys}


# ─── Experiment Builder ───────────────────────────────────────────────────────
def build_experiment(args):
    # Suppress logs
    suppress_warnings()

    # Device & global params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_params = ndarrays_to_parameters(get_parameters(Net()))

    # Client App
    client_app = ClientApp(client_fn=client_fn)

    # Plugins
    wandb_logger = WandBLogger(
        exp_dir=args.wandb_dir,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
    )
    csv_logger = CSVLogger(log_dir=args.csv_log_dir)
    plugins = [wandb_logger, csv_logger]

    # Disable summary printout
    ServerApp._log_summary = lambda *a, **k: None

    # Base strategy
    base = FedAvg(
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.num_partitions,
        initial_parameters=init_params,
        evaluate_fn=lambda r, p, c={}: evaluate(p, device=device),
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )
    hooked = HookedStrategy(base_strategy=base, plugins=plugins)
    final_strat = RoundTimerStrategy(base_strategy=hooked, plugins=plugins)

    # ServerApp
    def server_fn(ctx: Context) -> ServerAppComponents:
        return ServerAppComponents(
            strategy=final_strat,
            config=ServerConfig(num_rounds=args.num_rounds)
        )

    server_app = ServerApp(server_fn=server_fn)

    # Backend config
    backend = {}
    if device.type == "cuda":
        backend = {"client_resources": {"num_cpus": args.client_cpu, "num_gpus": args.client_gpu}}

    return client_app, server_app, plugins, backend


# ─── Runner ───────────────────────────────────────────────────────
def run_experiment(args):
    client_app, server_app, plugins, backend = build_experiment(args)
    run_simulation(
        client_app=client_app,
        server_app=server_app,
        num_supernodes=args.num_partitions,
        backend_config=backend,
    )
    # Finalize plugins
    for plugin in plugins:
        plugin.finalize()
