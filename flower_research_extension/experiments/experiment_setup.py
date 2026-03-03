import warnings
from typing import List, Tuple, Dict, Any
import inspect
import importlib
import logging

import torch

from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.client import ClientApp
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Context
from flwr.server.strategy import FedAvg

from flower_research_extension.strategies.hooked_strategy import HookedStrategy
from flower_research_extension.strategies.round_timer import RoundTimerStrategy
from flower_research_extension.plugins.base import MetricsPlugin
from flower_research_extension.model import get_parameters
from flower_research_extension.client import build_client_fn
from flower_research_extension.data_files import REGISTRY as DATASETS
from flower_research_extension.utils.reproducibility import seed_everything

# Use your original fit_config, and the new provider-aware evaluate
from flower_research_extension.training import evaluate_with_provider

logger = logging.getLogger(__name__)


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


def _make_wandb_logger(args):
    if getattr(args, "disable_wandb", False):
        return None
    try:
        from flower_research_extension.plugins.wandb_logger import WandBLogger

        return WandBLogger(
            exp_dir=args.wandb_dir,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
        )
    except Exception:
        logger.exception("Failed to initialize W&B logger; continuing without W&B")
        return None


def suppress_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="datasets")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # In monorepo layouts, "datasets" may resolve to a non-HuggingFace package.
    # Keep this optional so startup never fails due to import path collisions.
    try:
        hf_datasets = importlib.import_module("datasets")
        hf_logging = getattr(hf_datasets, "logging", None)
        if hf_logging is not None and hasattr(hf_logging, "set_verbosity_error"):
            hf_logging.set_verbosity_error()
    except Exception as exc:
        logger.debug("Skipping HuggingFace datasets logging suppression: %s", exc)


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


def _make_fit_config_fn(*, local_epochs: int, lr: float, momentum: float):
    def _fit_config(server_round: int) -> Dict:
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "lr": lr,
            "momentum": momentum,
        }

    return _fit_config


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

    provider = DATASETS.get(getattr(args, "dataset", "cifar10"))
    dataset_root = getattr(args, "dataset_root", "data")
    batch_size = getattr(args, "batch_size", 64)
    seed = getattr(args, "seed", 42)
    seed_everything(seed)

    num_classes = int(getattr(provider, "num_classes", 10))
    model = args.model_builder(num_classes).to(device)
    init_params = ndarrays_to_parameters(get_parameters(model))

    client_app = ClientApp(
        client_fn=build_client_fn(
            provider=provider,
            dataset_root=dataset_root,
            num_partitions=args.num_partitions,
            device=device,
            batch_size=batch_size,
            seed=seed,
            distribution=str(getattr(args, "distribution", "iid")),
            dirichlet_alpha=float(getattr(args, "dirichlet_alpha", 0.5)),
            label_skew_classes=int(getattr(args, "label_skew_classes", 2)),
            shard_num_shards_per_partition=int(getattr(args, "shard_num_shards_per_partition", 2)),
            inner_dirichlet_alpha=float(getattr(args, "inner_dirichlet_alpha", 0.5)),
            size_partition_weights=getattr(args, "size_partition_weights", None),
            distribution_matrix=getattr(args, "distribution_matrix", None),
            model_builder=args.model_builder
        )
    )

    csv_logger = _make_csv_logger(getattr(args, "csv_log_dir", "results/logs"))
    plugins: List[MetricsPlugin] = []
    wandb_logger = _make_wandb_logger(args)
    if wandb_logger is not None:
        plugins.append(wandb_logger)
    plugins.append(csv_logger)

    min_avail = max(args.min_fit_clients, args.min_evaluate_clients)

    base = FedAvg(
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=min_avail,
        initial_parameters=init_params,
        on_fit_config_fn=_make_fit_config_fn(
            local_epochs=int(getattr(args, "local_epochs", 5)),
            lr=float(getattr(args, "lr", 0.01)),
            momentum=float(getattr(args, "momentum", 0.9)),
        ),
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

    timed = RoundTimerStrategy(base_strategy=base)
    final_strat = HookedStrategy(base_strategy=timed, plugins=plugins)

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
            logger.exception(
                "Plugin %s failed in on_training_start; continuing",
                p.__class__.__name__,
            )

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
        try:
            plugin.finalize()
        except Exception:
            logger.exception(
                "Plugin %s failed in finalize; continuing",
                plugin.__class__.__name__,
            )
