# flower_research_extension/data_files/cifar10.py

import os
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data._utils.collate import default_collate

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
BATCH_SIZE     = 32
USE_PIN_MEMORY = torch.cuda.is_available()
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")

# ------------------------------------------------------------------------------
# ONE-TIME DOWNLOAD
# ------------------------------------------------------------------------------
os.makedirs(DATA_DIR, exist_ok=True)
# Download only the train split (we’ll shard/train-val split it ourselves)
CIFAR10(root=DATA_DIR, train=True, download=True)

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def _collate_dict(batch):
    """Convert [(img, label), …] into {'img': Tensor, 'label': Tensor}."""
    imgs, labels = zip(*batch)
    return {
        "img":   default_collate(imgs),
        "label": default_collate(labels),
    }

# ------------------------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------------------------
def load_cifar10_partition(partition_id: int, num_partitions: int):
    """
    Exactly as before: give me train/val loaders for client partition `partition_id`.
    Batches come out as dicts with keys "img" and "label".
    """
    # 1) standard CIFAR10 transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 2) load full train set from cache
    full = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform,
    )

    # 3) carve out this client’s shard
    total      = len(full)                 # 50 000
    per_client = total // num_partitions
    start      = partition_id * per_client
    end        = start + per_client if partition_id < num_partitions - 1 else total
    shard      = Subset(full, list(range(start, end)))

    # 4) 80/20 train/val split *within* the shard
    train_size = int(len(shard) * 0.8)
    val_size   = len(shard) - train_size
    train_ds, val_ds = random_split(
        shard,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # 5) DataLoaders with dict batches
    trainloader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=USE_PIN_MEMORY,
        collate_fn=_collate_dict,
    )
    valloader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=USE_PIN_MEMORY,
        collate_fn=_collate_dict,
    )

    return trainloader, valloader
