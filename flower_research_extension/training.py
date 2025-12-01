from typing import Dict, Tuple, Any
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from flower_research_extension.model import Net, set_parameters
from flower_research_extension.data_files.base import PartitionSpec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PARTITIONS = 20  # kept for backward-compat with your original evaluate()


def fit_config(server_round: int) -> Dict:
    return {
        "server_round": server_round,
        "local_epochs": 5,
    }


def _iter_batches(dl: DataLoader):
    """Handle dict or tuple batches uniformly."""
    for batch in dl:
        if isinstance(batch, dict):
            images, labels = batch["img"], batch["label"]
        else:
            images, labels = batch
        yield images, labels


def _compute_classification_metrics(
    model: torch.nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict]:
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    batch_accuracies = []

    with torch.no_grad():
        for images, labels in _iter_batches(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            batch_accuracy = (preds == labels).sum().item() / len(labels)
            batch_accuracies.append(batch_accuracy)

    # why: average CE per-sample (matches your earlier intent)
    loss = total_loss / max(1, len(testloader.dataset))
    accuracy = float(np.mean(np.array(y_pred) == np.array(y_true)))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "accuracy_min": float(np.min(batch_accuracies)) if batch_accuracies else 0.0,
        "accuracy_max": float(np.max(batch_accuracies)) if batch_accuracies else 0.0,
        "accuracy_std": float(np.std(batch_accuracies)) if batch_accuracies else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return float(loss), metrics


# ----- Original CIFAR-10–specific evaluate (kept intact) -----------------------
from flower_research_extension.data_files.cifar10 import load_cifar10_partition  # noqa: E402


def evaluate(parameters, device=DEVICE) -> Tuple[float, Dict]:
    model = Net().to(device)
    set_parameters(model, parameters)
    _, testloader = load_cifar10_partition(0, NUM_PARTITIONS)

    return _compute_classification_metrics(model, testloader, device)


# ----- New: provider-aware evaluate (works for any registered dataset) ---------
def evaluate_with_provider(
    parameters_ndarrays: Any,
    *,
    provider,
    dataset_root: str | Path,
    device: torch.device = DEVICE,
    batch_size: int = 128,
    seed: int = 42,
) -> Tuple[float, Dict]:
    """Evaluate global params on the provider's test set (any dataset)."""
    model = Net().to(device)
    set_parameters(model, parameters_ndarrays)

    # Single logical partition for evaluation; no shuffling
    spec = PartitionSpec(
        partition_id=0,
        num_partitions=1,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )
    _, _, testloader = provider.partition(Path(dataset_root), spec)

    return _compute_classification_metrics(model, testloader, device)
