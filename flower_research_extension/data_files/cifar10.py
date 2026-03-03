from pathlib import Path
from typing import Tuple, Union

from torch.utils.data import DataLoader

from flower_research_extension.data_files.base import PartitionSpec
from flower_research_extension.data_files.cifar10_provider import CIFAR10Provider


def load_cifar10_partition(
    partition_id: int,
    num_partitions: int,
    *,
    batch_size: int = 32,
    seed: int = 42,
    root: Union[str, Path, None] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Backward-compatible helper for CIFAR-10 partition loading.

    This wrapper intentionally avoids any import-time download or filesystem side effects.
    Dataset preparation is performed only when this function is called.
    """
    provider = CIFAR10Provider()
    dataset_root = Path(root) if root is not None else Path(__file__).resolve().parent / "data"
    provider.prepare(dataset_root)

    spec = PartitionSpec(
        partition_id=partition_id,
        num_partitions=max(1, num_partitions),
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )
    trainloader, valloader, _ = provider.partition(dataset_root, spec)
    return trainloader, valloader
