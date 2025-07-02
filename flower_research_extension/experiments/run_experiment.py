from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.client import ClientApp
from flwr.common import ndarrays_to_parameters, Context

from flower_research_extension.strategies.hooked_strategy import HookedStrategy
from flower_research_extension.strategies.round_timer import RoundTimerStrategy
from flower_research_extension.plugins.wandb_logger import WandBLogger
from flower_research_extension.plugins.csv_logger import CSVLogger

from flower_research_extension.model import Net, get_parameters
from flower_research_extension.client import client_fn
from flower_research_extension.training import evaluate, fit_config

import torch

# ─── Constants ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PARTITIONS = 20
NUM_ROUNDS = 5
INITIAL_PARAMETERS = ndarrays_to_parameters(get_parameters(Net()))

# ─── Client App ───────────────────────────────────────────────────────────────
client_app = ClientApp(client_fn=client_fn)

# ─── Plugins ──────────────────────────────────────────────────────────────────
wandb_logger = WandBLogger(
    exp_dir="results/wandb",
    project="flower-federated",
    run_name="cifar10_fedavg"
)

csv_logger = CSVLogger(
    log_dir="results/logs"
)

plugins = [wandb_logger, csv_logger]

# ─── Strategy Setup ───────────────────────────────────────────────────────────
from flwr.server.strategy import FedAvg

base_strategy = FedAvg(
    fraction_fit=0.25,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=NUM_PARTITIONS,
    initial_parameters=INITIAL_PARAMETERS,
    evaluate_fn=lambda r, p, c={}: evaluate(r, p, c, device=DEVICE),
    on_fit_config_fn=fit_config,
)

hooked_strategy = HookedStrategy(base_strategy=base_strategy, plugins=plugins)
final_strategy = RoundTimerStrategy(base_strategy=hooked_strategy, plugins=plugins)

# ─── Server Setup ─────────────────────────────────────────────────────────────
def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=final_strategy, config=config)

server_app = ServerApp(server_fn=server_fn)

# ─── Run the Simulation ───────────────────────────────────────────────────────
backend_config = {}
if DEVICE.type == "cuda":
    backend_config = {
        "client_resources": {
            "num_cpus": 2,
            "num_gpus": 0.1  # allow 10 clients to share the GPU
        }
    }

run_simulation(
    client_app=client_app,
    server_app=server_app,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)

# ─── Cleanup ──────────────────────────────────────────────────────────────────
for plugin in plugins:
    plugin.finalize()
