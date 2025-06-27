# ─── Suppress Specific Warnings ────────────────────────────────────────────────
import dev.suppress_warnings  # must come before any other imports!

_ = dev.suppress_warnings



# ─── Standard Library Imports ─────────────────────────────────────────────────
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# ─── Third-Party Imports ───────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
import wandb
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

from flwr_datasets import FederatedDataset

from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

# ─── Device & Global Constants ────────────────────────────────────────────────
DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
NUM_PARTITIONS = 100  # Number of federated clients
BATCH_SIZE = 32  # Mini-batch size per client
NUM_ROUNDS = 100

# Diagnostics
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")


# ─── Data Loading Utility ─────────────────────────────────────────────────────
def load_datasets(partition_id: int, num_partitions: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    # 80% train / 20% test split on this node’s data
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)

    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloader, valloader, testloader


# ─── Model Definition ─────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ─── Model Parameter Conversion Utilities ───────────────────────────────────────────
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# ─── Training & Evaluation Routines ───────────────────────────────────────────
def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# ─── Flower Client Implementation ─────────────────────────────────────────────
class FlowerClient(NumPyClient):
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.pid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        print(f"[Client {self.pid}, round {server_round}] fit, config: {config}")

        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=local_epochs)

        # Compute local accuracy on your val‐split
        loss, accuracy = test(self.net, self.valloader)
        print(f"[Client {self.pid}] local eval loss {loss:.4f}, acc {accuracy:.4f}")

        # **Return** the accuracy so aggregate_fit can see it
        return get_parameters(self.net), len(self.trainloader), {
            "accuracy": accuracy,
            "loss": loss,
        }
    def evaluate(self, parameters, config):
        print(f"[Client {self.pid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
    net = Net().to(DEVICE)  # initialize the model and move it to the selected device (CPU or GPU)
    pid = context.node_config["partition-id"]  # get this client's partition ID
    nparts = context.node_config["num-partitions"]  # get the total number of partitions (clients)
    trainloader, valloader, _ = load_datasets(pid, nparts)  # load this client's dataset partition
    return FlowerClient(pid, net, trainloader, valloader).to_client()  # return wrapped client instance


# Build ClientApp
client = ClientApp(client_fn=client_fn)

# ─── Fit Config & Server-Side Eval ────────────────────────────────────────────
params = get_parameters(Net())


def fit_config(server_round: int):
    """Return training configuration for each round."""
    return {
        "server_round": server_round,
        "local_epochs": 1 if server_round < 2 else 2,
    }


def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    net = Net().to(DEVICE)
    _, _, testloader = load_datasets(0, NUM_PARTITIONS)
    set_parameters(net, parameters)
    loss, accuracy = test(net, testloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


# ─── CustomFedAvg Implementation ─────────────────────────────────────────────
class CustomFedAvg(FedAvg):
    """FedAvg + per-experiment folder + configurable checkpointing."""

    def __init__(self, *args, save_every: int = 50, num_rounds: int = None, **kwargs):
        # Extract our custom args
        self.save_every = save_every
        self.num_rounds = num_rounds

        # In‐memory metrics store
        self.results_to_save = {}

        # Prepare a unique experiment directory
        base_dir = Path(__file__).parent / "results"
        base_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = base_dir / f"exp_{ts}"
        self.exp_dir.mkdir()

        # ←── Add this line so evaluate() has a valid path
        self.json_path = self.exp_dir / "results.json"

        import wandb
        # Single, server‐side run
        wandb.init(
            project="flower-simulation-tutorial",
            name=f"flower-sim-{ts}",
            dir=str(self.exp_dir / "wandb"),
            job_type="server",
        )

        wandb.config.update({
            "device": str(DEVICE),
            "batch_size": BATCH_SIZE,
            "num_partitions": NUM_PARTITIONS,
            "num_rounds": NUM_ROUNDS,
            "save_every": save_every,
        })

        # model_for_watch = Net().to(DEVICE)
        # wandb.watch(model_for_watch, log="all", log_freq=100)

        # Finally, call the parent constructor
        super().__init__(*args, **kwargs)

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[tuple[ClientProxy, Exception]],
    ) -> tuple[Parameters, dict]:

        # 1. Perform standard FedAvg aggregation
        # ─── START TIMING THIS ROUND ─────────────────────────────────────
        round_start = time.time()
        # 1. Perform standard FedAvg aggregation
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        # If no clients succeeded, parameters_aggregated will be None
        if parameters_aggregated is None:
            # simply return the last global params unchanged
            return self.initial_parameters, {}

        # 2. Save checkpoint if needed
        is_final = (self.num_rounds is not None and server_round == self.num_rounds)
        if (server_round % self.save_every == 0) or is_final:
            ndarrays = parameters_to_ndarrays(parameters_aggregated)
            model = Net().to(DEVICE)
            set_parameters(model, ndarrays)
            ckpt_path = self.exp_dir / f"global_model_round_{server_round}.pt"
            torch.save(model.state_dict(), ckpt_path)

        # 3. Build a table of per-client metrics
        import wandb
        rows = []
        for client_proxy, fit_res in results:
            rows.append({
                "round": server_round,
                "client_id": client_proxy.cid,
                "num_examples": fit_res.num_examples,
                **fit_res.metrics,
            })
        client_table = wandb.Table(
            data=rows,
            columns=sorted(rows[0].keys()) if rows else ["round", "client_id", "num_examples"]
        )

        # ─── CLIENT ACCURACY SPREAD ───────────────────────────────────
        client_accs = [r["accuracy"] for r in rows if "accuracy" in r]

        if client_accs:
            acc_std = float(np.std(client_accs))
            acc_min = float(np.min(client_accs))
            acc_max = float(np.max(client_accs))
        else:
            acc_std = acc_min = acc_max = 0.0

        # ─── ROUND DURATION ───────────────────────────────────────────
        round_time = time.time() - round_start

        # ─── COUNT CLIENTS ─────────────────────────────────────────
        num_selected = len(results) + len(failures)
        num_successful = len(results)
        num_failed = len(failures)

        # 4. Log **one** entry per round: global + client table
        wandb.log({
            # basic
            "round": server_round,
            "global_loss": metrics_aggregated.get("loss"),
            "global_accuracy": metrics_aggregated.get("accuracy"),

            # client‐accuracy spread
            "client_acc_std": acc_std,
            "client_acc_min": acc_min,
            "client_acc_max": acc_max,

            # round duration
            "round_time": round_time,

            "clients/selected": num_selected,
            "clients/succeeded": num_successful,
            "clients/failed": num_failed,

            # always log this table last
            "client_metrics": client_table,
        })

        return parameters_aggregated, metrics_aggregated

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model, then save metrics to local JSON and to W&B."""
        # Call the default behaviour from FedAvg
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None
        loss, metrics = result

        # 2) Build y_true / y_pred on your test set
        net = Net().to(DEVICE)
        ndarrays = parameters_to_ndarrays(parameters)
        set_parameters(net, ndarrays)
        _, _, testloader = load_datasets(0, NUM_PARTITIONS)

        y_true, y_pred = [], []
        net.eval()
        with torch.no_grad():
            for batch in testloader:
                imgs = batch["img"].to(DEVICE)
                lbls = batch["label"].to(DEVICE)
                logits = net(imgs)
                y_true.extend(lbls.cpu().numpy())
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())

        # 3) Compute macro‐averaged precision, recall, F1
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Store metrics as dictionary
        my_results = {
            "loss": loss,
            "accuracy": metrics["accuracy"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        self.results_to_save[server_round] = my_results

        # Save metrics as JSON in this experiment’s folder
        with open(self.json_path, "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log({
            "loss": loss,
            "accuracy": metrics["accuracy"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "round": server_round,
        })

        # Return the expected outputs for `evaluate`
        return loss, metrics


# ─── Flower Server Implementation ─────────────────────────────────────────────
def aggregate_fit_metrics(
        metrics: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Compute a weighted average of client training accuracy."""
    # metrics is a list of (num_examples, metrics_dict) tuples
    weighted_acc = sum(n * m["accuracy"] for n, m in metrics) / sum(n for n, _ in metrics)
    return {"accuracy": float(weighted_acc)}


def server_fn(context: Context) -> ServerAppComponents:
    strategy = CustomFedAvg(
        fraction_fit=0.25,  # fraction of clients for training each round
        fraction_evaluate=0.05,  # fraction of clients for evaluation each round
        min_fit_clients=3,  # minimum clients required to train
        min_evaluate_clients=3,  # minimum clients required to evaluate
        min_available_clients=NUM_PARTITIONS,  # total clients that must be available
        initial_parameters=ndarrays_to_parameters(params),  # global model init params
        evaluate_fn=evaluate,  # server-side evaluation callback
        on_fit_config_fn=fit_config,  # callback to supply per-round train config
        save_every=10,  # checkpoint every 10 rounds
        num_rounds=NUM_ROUNDS,  # total federated rounds to run
        # fit_metrics_aggregation_fn=aggregate_fit_metrics,  # clients train aggregated metrics

    )
    config = ServerConfig(num_rounds=NUM_ROUNDS)  # server run configuration
    return ServerAppComponents(strategy=strategy, config=config)  # return strategy+config


# Build ServerApp
server = ServerApp(server_fn=server_fn)

# ─── Simulation Launch ────────────────────────────────────────────────────────
# Default resources: 2 CPUs, 0 GPUs per client
backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
    backend_config = {
        "client_resources": {
            "num_cpus": 1,  # or however many cores each client should assume
            "num_gpus": 1,  # or a fraction like 0.25 if you want multiple per GPU
        }
    }

run_simulation(
    server_app=server,  # the server logic to execute each round
    client_app=client,  # the client logic to simulate on each node
    num_supernodes=NUM_PARTITIONS,  # number of simulated client nodes (partitions)
    backend_config=backend_config,  # hardware resource configuration for each client
)
