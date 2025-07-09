import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data._utils.collate import default_collate
from flwr_datasets import FederatedDataset
from datasets import logging as hf_logging
from huggingface_hub import login
from torchvision.datasets import CIFAR10

hf_logging.set_verbosity_error()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
if HF_TOKEN:
    login(HF_TOKEN)

BATCH_SIZE     = 32
USE_PIN_MEMORY = torch.cuda.is_available()
CACHE_DIR      = "/tmp/huggingface_cache"

# Global sentinel to avoid retrying HF every time
_hf_federated_failed = False

def collate_dict(batch):
    imgs, labels = zip(*batch)
    return {
        "img":   default_collate(imgs),
        "label": default_collate(labels),
    }

def load_cifar10_partition(partition_id: int, num_partitions: int):
    """Load CIFAR‑10 (federated or fallback) exactly once per process."""
    global _hf_federated_failed

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    # 1️⃣ Try once to use HuggingFace FederatedDataset
    if not _hf_federated_failed:
        try:
            fds = FederatedDataset(
                dataset="cifar10",
                partitioners={"train": num_partitions},
                cache_dir=CACHE_DIR,
            )
            partition = fds.load_partition(partition_id)
            train_test = partition.train_test_split(test_size=0.2, seed=42)

            trainloader = DataLoader(
                train_test["train"],
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=USE_PIN_MEMORY,
            )
            valloader = DataLoader(
                train_test["test"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=USE_PIN_MEMORY,
            )
            return trainloader, valloader

        except Exception as e:
            print(f"❌ HF federated load failed: {e}")
            print("↪️ Falling back to torchvision.CIFAR10 …")
            _hf_federated_failed = True  # never try HF again

    # 2️⃣ Always use torchvision fallback from now on
    full = CIFAR10(
        root=CACHE_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    # split full dataset into equal client partitions
    total = len(full)
    per_client = total // num_partitions
    start = partition_id * per_client
    end   = start + per_client if partition_id < num_partitions - 1 else total
    subset = Subset(full, list(range(start, end)))

    # 80/20 train/val split
    train_size = int(len(subset) * 0.8)
    val_size   = len(subset) - train_size
    train_ds, val_ds = random_split(
        subset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=USE_PIN_MEMORY,
        collate_fn=collate_dict,
    )
    valloader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=USE_PIN_MEMORY,
        collate_fn=collate_dict,
    )

    return trainloader, valloader
