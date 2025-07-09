import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data._utils.collate import default_collate
from flwr_datasets import FederatedDataset
from datasets import logging as hf_logging
from torchvision.datasets import CIFAR10

hf_logging.set_verbosity_error()

BATCH_SIZE     = 32
USE_PIN_MEMORY = torch.cuda.is_available()
CACHE_DIR      = "/tmp/huggingface_cache"

# Once HF fails, never retry
_hf_federated_failed = False

# Our standard CIFAR-10 preprocess
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),
])

def _apply_transforms(example):
    # example is a dict {"img": PIL.Image, "label": int}
    example["img"] = _transform(example["img"])
    return example

def _collate_dict(batch):
    # batch is a list of (tensor, int)
    imgs, labels = zip(*batch)
    return {
        "img":   default_collate(imgs),
        "label": default_collate(labels),
    }

def load_cifar10_partition(partition_id: int, num_partitions: int):
    """
    Return (trainloader, valloader) for client `partition_id` out of `num_partitions`.
    Tries HF FederatedDataset once, then falls back to torchvision.CIFAR10.
    """
    global _hf_federated_failed

    # 1️⃣ Try HF FederatedDataset (once per process)
    if not _hf_federated_failed:
        try:
            fds = FederatedDataset(
                dataset="cifar10",
                partitioners={"train": num_partitions},
                cache_dir=CACHE_DIR,
            )
            partition = fds.load_partition(partition_id)
            train_test = partition.train_test_split(test_size=0.2, seed=42)

            # apply our tensor-transform to each example
            train_test = train_test.with_transform(_apply_transforms)

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
            _hf_federated_failed = True

    # 2️⃣ torchvision fallback
    # load entire CIFAR-10 with our transform baked in
    full = CIFAR10(
        root=CACHE_DIR,
        train=True,
        download=True,
        transform=_transform,
    )

    # carve out this client's shard
    total     = len(full)
    per_client = total // num_partitions
    start      = partition_id * per_client
    end        = start + per_client if partition_id < num_partitions - 1 else total
    shard      = Subset(full, list(range(start, end)))

    # 80/20 train/val split within the shard
    train_size = int(len(shard) * 0.8)
    val_size   = len(shard) - train_size
    train_ds, val_ds = random_split(
        shard,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

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
