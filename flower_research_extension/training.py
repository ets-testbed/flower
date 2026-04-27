from typing import Dict, Tuple, Any
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from typing import Callable, Optional
import torch.nn as nn

from flower_research_extension.model import Net, set_parameters
from flower_research_extension.data_files.base import PartitionSpec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PARTITIONS = 20  # kept for backward-compat with your original evaluate()




def _build_model(
    *,
    provider=None,
    model_builder: Optional[Callable[[int], nn.Module]] = None,
) -> nn.Module:
    num_classes = int(getattr(provider, "num_classes", 10)) if provider is not None else 10
    if model_builder is not None:
        return model_builder(num_classes)
    return Net(num_classes=num_classes)


def make_fit_config_fn(
    *,
    local_epochs: int = 5,
    lr: float = 0.01,
    momentum: float = 0.9,
) -> Callable[[int], Dict]:
    def fit_config(server_round: int) -> Dict:
        return {
            "server_round": server_round,
            "local_epochs": int(local_epochs),
            "lr": float(lr),
            "momentum": float(momentum),
        }

    return fit_config


def fit_config(server_round: int) -> Dict:
    return make_fit_config_fn()(server_round)


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
            loss_batch = criterion(outputs, labels)  # mean over batch by default
            total_loss += float(loss_batch.item()) * labels.size(0)

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


# ----- New: provider-aware evaluate (works for any registered dataset) ---------
def evaluate_with_provider(
    parameters_ndarrays: Any,
    *,
    provider,
    dataset_root: str | Path,
    device: torch.device = DEVICE,
    batch_size: int = 128,
    seed: int = 42,
    model_builder=None,
) -> Tuple[float, Dict]:
    model = _build_model(provider=provider, model_builder=model_builder).to(device)
    set_parameters(model, parameters_ndarrays)

    spec = PartitionSpec(
        partition_id=0,
        num_partitions=1,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )
    _, _, testloader = provider.partition(Path(dataset_root), spec)

    return _compute_classification_metrics(model, testloader, device)
