import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset

from datasets import logging as hf_logging

hf_logging.set_verbosity_error()

BATCH_SIZE = 32


def load_cifar10_partition(partition_id: int, num_partitions: int):
    """Load partitioned CIFAR-10 dataset with 80/20 train/val split."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def apply_transforms(batch):
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                             pin_memory=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE, num_workers=4,
                           pin_memory=True)

    return trainloader, valloader
