import warnings
from typing import List, Tuple, Dict, Any
import inspect

import datasets
from datasets import logging as hf_logging
import torch

from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.client import ClientApp
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Context
from flwr.server.strategy import FedAvg

from flower_research_extension.strategies.hooked_strategy import HookedStrategy
from flower_research_extension.strategies.round_timer import RoundTimerStrategy
from flower_research_extension.plugins.wandb_logger import WandBLogger
from flower_research_extension.plugins.base import MetricsPlugin
from flower_research_extension.model import Net, get_parameters
from flower_research_extension.client import build_client_fn
from flower_research_extension.data_files import REGISTRY as DATASETS

# Use your original fit_config, and the new provider-aware evaluate
from flower_research_extension.training import fit_config as training_fit_config, evaluate_with_provider


def _make_csv_logger(path: str):
    """Be tolerant of constructor parameter names across versions."""
    try:
        try:
            from flower_research_extension.plugins.csv_logger import CsvLogger as _Csv
        except ImportError:
            from flower_research_extension.plugins.csv_logger import CSVLogger as _Csv  # type: ignore

        params = list(inspect.signature(_Csv.__init__).parameters.keys())
        if "exp_dir" in params:
            return _Csv(exp_dir=path)
        if "log_dir" in params:
            return _Csv(log_dir=path)
        if "out_dir" in params:
            return _Csv(out_dir=path)
        try:
            return _Csv(path)
        except TypeError:
            return _Csv()
    except Exception:
        class _Noop(MetricsPlugin):
            def finalize(self):  # prevent finalize errors
                pass
        return _Noop()


def suppress_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="datasets")
    datasets.logging.set_verbosity_error()
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def aggregate_fit_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    return {
        "accuracy": sum(m.get("accuracy", 0.0) * n for n, m in metrics) / max(1, total),
        "loss": sum(m.get("loss", 0.0) * n for n, m in metrics) / max(1, total),
    }


def aggregate_evaluate_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    keys = set().union(*(m.keys() for _, m in metrics))
    return {k: sum(m.get(k, 0.0) * n for n, m in metrics) / max(1, total) for k in keys}


def _on_evaluate_config(server_round: int) -> Dict:
    return {"server_round": server_round}


def _evaluate_fn_factory(
    *,
    device: torch.device,
    provider,
    dataset_root: str,
    batch_size: int,
    seed: int,
):
    # Accept either a Parameters proto or already-converted ndarrays.
    def evaluate_fn(server_round: int, parameters: Any, config: Dict) -> Tuple[float, Dict]:
        if isinstance(parameters, list):
            nds = parameters
        else:
            nds = parameters_to_ndarrays(parameters)
        return evaluate_with_provider(
            nds,
            provider=provider,
            dataset_root=dataset_root,
            device=device,
            batch_size=batch_size,
            seed=seed,
        )
    return evaluate_fn


def build_experiment(args):
    suppress_warnings()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_params = ndarrays_to_parameters(get_parameters(Net()))

    provider = DATASETS.get(getattr(args, "dataset", "cifar10"))
    dataset_root = getattr(args, "dataset_root", "data")
    batch_size = getattr(args, "batch_size", 64)
    seed = getattr(args, "seed", 42)

    client_app = ClientApp(
        client_fn=build_client_fn(
            provider=provider,
            dataset_root=dataset_root,
            num_partitions=args.num_partitions,
            device=device,
            batch_size=batch_size,
            seed=seed,
        )
    )

    wandb_logger = WandBLogger(
        exp_dir=args.wandb_dir,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
    )
    csv_logger = _make_csv_logger(getattr(args, "csv_log_dir", "results/logs"))
    plugins: List[MetricsPlugin] = [wandb_logger, csv_logger]

    min_avail = max(args.min_fit_clients, args.min_evaluate_clients)

    base = FedAvg(
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=min_avail,
        initial_parameters=init_params,
        on_fit_config_fn=training_fit_config,  # your original hook
        on_evaluate_config_fn=_on_evaluate_config,
        evaluate_fn=_evaluate_fn_factory(
            device=device,
            provider=provider,
            dataset_root=dataset_root,
            batch_size=batch_size,
            seed=seed,
        ),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    hooked = HookedStrategy(base_strategy=base, plugins=plugins)
    final_strat = RoundTimerStrategy(base_strategy=hooked, plugins=plugins)

    def server_fn(ctx: Context) -> ServerAppComponents:
        return ServerAppComponents(
            strategy=final_strat,
            config=ServerConfig(num_rounds=args.num_rounds),
        )

    server_app = ServerApp(server_fn=server_fn)

    backend = {
        "client_resources": {
            "num_cpus": getattr(args, "client_cpu", 1),
            "num_gpus": getattr(args, "client_gpu", 0.0),
        }
    }

    # Make runs visible immediately
    for p in plugins:
        try:
            p.on_training_start({"dataset": provider.name, "num_partitions": args.num_partitions})
        except Exception:
            pass

    return client_app, server_app, plugins, backend


def run_experiment(args):
    client_app, server_app, plugins, backend = build_experiment(args)
    run_simulation(
        client_app=client_app,
        server_app=server_app,
        num_supernodes=args.num_partitions,
        backend_config=backend,
    )
    for plugin in plugins:
        plugin.finalize()
