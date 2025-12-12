from typing import Callable, Tuple, Dict, Any, Optional
from pathlib import Path
import zlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from flwr.client import NumPyClient, Client

from flower_research_extension.model import Net, get_parameters, set_parameters
from flower_research_extension.data_files.base import PartitionSpec


class _DLClient(NumPyClient):
    """NumPyClient using provided DataLoaders."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loaders: Tuple[DataLoader, DataLoader, DataLoader],
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_dl, self.val_dl, self.test_dl = loaders
        self.criterion = nn.CrossEntropyLoss()  # classification default

    def get_parameters(self, config: Dict[str, Any]):
        return get_parameters(self.model)

    def fit(self, parameters, config: Dict[str, Any]):
        set_parameters(self.model, parameters)
        lr = float(config.get("lr", 0.01))
        momentum = float(config.get("momentum", 0.9))
        local_epochs = int(config.get("local_epochs", 1))

        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.model.train()

        total_loss, total_correct, total_seen = 0.0, 0, 0
        for _ in range(local_epochs):
            for xb, yb in self.train_dl:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_loss += float(loss.item()) * yb.size(0)
                    preds = logits.argmax(dim=1)
                    total_correct += int((preds == yb).sum().item())
                    total_seen += int(yb.size(0))

        avg_loss = total_loss / max(1, total_seen)
        avg_acc = total_correct / max(1, total_seen)
        metrics = {"loss": avg_loss, "accuracy": avg_acc, "local_epochs": local_epochs}
        return get_parameters(self.model), total_seen, metrics

    def evaluate(self, parameters, config: Dict[str, Any]):
        set_parameters(self.model, parameters)
        self.model.eval()

        total_loss, total_correct, total_seen = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in self.test_dl:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                total_loss += float(loss.item()) * yb.size(0)
                preds = logits.argmax(dim=1)
                total_correct += int((preds == yb).sum().item())
                total_seen += int(yb.size(0))

        avg_loss = total_loss / max(1, total_seen)
        avg_acc = total_correct / max(1, total_seen)
        metrics = {"loss": avg_loss, "accuracy": avg_acc}
        return float(avg_loss), total_seen, metrics


def _extract_cid_from_context(context: Any) -> Optional[str]:
    """Try multiple fields to get CID across Flower versions."""
    cid = getattr(context, "node_id", None)
    if cid is not None:
        return str(cid)
    props = getattr(context, "properties", None)
    if isinstance(props, dict):
        for key in ("node_id", "cid", "client_id", "partition_id"):
            if key in props:
                return str(props[key])
    return None


def _cid_to_partition(cid_str: str, num_active: int) -> int:
    """Deterministically map arbitrary CID into [0, num_active)."""
    if num_active <= 0:
        return 0
    try:
        cid_int = int(cid_str)
        if 0 <= cid_int < num_active:
            return cid_int
    except Exception:
        pass
    return int(zlib.crc32(cid_str.encode("utf-8")) % num_active)


def _patch_num_classes(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the final classification layer to match num_classes when possible."""

    # Your custom Net: fc3
    if hasattr(model, "fc3") and isinstance(getattr(model, "fc3"), nn.Linear):
        if model.fc3.out_features != num_classes:
            model.fc3 = nn.Linear(model.fc3.in_features, num_classes)
        return model

    # Common: .fc (ResNet-like)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        if model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # Common: .classifier as Linear
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        if model.classifier.out_features != num_classes:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    # Common: .classifier as Sequential -> replace last Linear
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for i in reversed(range(len(model.classifier))):
            if isinstance(model.classifier[i], nn.Linear):
                if model.classifier[i].out_features != num_classes:
                    model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
                return model

    return model



def build_client_fn(
    *,
    provider,
    dataset_root: str | Path,
    num_partitions: int,
    device: torch.device,
    batch_size: int = 64,
    seed: int = 42,
    model_builder: Optional[Callable[[int], nn.Module]] = None,
) -> Callable[..., Client]:
    """
    Client factory compatible with both `client_fn(context)` and legacy `client_fn(cid)`.
    Ensures non-empty shards and returns a true `Client` instance.
    """
    dataset_root = Path(dataset_root)
    provider.prepare(dataset_root)

    # Inspect train size once to guarantee non-empty partitions
    train_ds, _, _ = provider.raw_datasets(dataset_root)
    train_size = len(train_ds)
    active_partitions = max(1, min(num_partitions, train_size))

    def _make_client(pid: int) -> Client:
        spec = PartitionSpec(
            partition_id=pid,
            num_partitions=active_partitions,
            batch_size=batch_size,
            seed=seed,
        )
        loaders = provider.partition(dataset_root, spec)

        num_classes = int(getattr(provider, "num_classes", 10))

        if model_builder:
            model = model_builder(num_classes)
        else:
            # Preferred: build correctly from the start
            try:
                model = Net(num_classes=num_classes)
            except TypeError:
                # Backward compatibility if Net signature wasn't updated
                model = Net()
                model = _patch_num_classes(model, num_classes)

        return _DLClient(model=model, device=device, loaders=loaders).to_client()

    def client_fn(context_or_cid):
        if isinstance(context_or_cid, str):
            cid_str = context_or_cid
        else:
            cid_str = _extract_cid_from_context(context_or_cid) or "0"
        pid = _cid_to_partition(cid_str, active_partitions)
        return _make_client(pid)

    return client_fn
