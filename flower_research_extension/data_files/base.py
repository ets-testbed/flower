# =========================================
# file: flower_research_extension/data_files/base.py
# =========================================
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import math
import random

@dataclass(frozen=True)
class PartitionSpec:
    partition_id: int
    num_partitions: int
    batch_size: int = 64
    seed: int = 42
    shuffle: bool = True
    num_workers: int = 2
    drop_last: bool = False

class DatasetProvider(ABC):
    """Uniform dataset interface so strategies/clients stay dataset-agnostic."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @abstractmethod
    def prepare(self, root: Path) -> None:
        """One-time dataset preparation (download, cache, etc.)."""

    @abstractmethod
    def raw_datasets(self, root: Path) -> Tuple[Dataset, Optional[Dataset], Dataset]:
        """Return (train, val, test). `val` can be None if you want to split from train."""

    def _iid_indices(self, n: int, spec: PartitionSpec) -> range:
        """Deterministic IID split into `num_partitions` shards (equal-sized; last shard may be larger)."""
        g = random.Random(spec.seed)
        idxs = list(range(n))
        g.shuffle(idxs)
        per = math.ceil(n / spec.num_partitions)
        start = spec.partition_id * per
        end = min(start + per, n)
        return idxs[start:end]

    def partition(
        self,
        root: Path,
        spec: PartitionSpec,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for a user/client partition."""
        train_ds, val_ds, test_ds = self.raw_datasets(root)
        # If no explicit val set, carve a 10% split from train deterministically
        if val_ds is None:
            total = len(train_ds)
            val_count = max(1, int(0.1 * total))
            val_idxs = list(range(val_count))
            train_idxs = list(range(val_count, total))
            val_ds = Subset(train_ds, val_idxs)
            train_ds = Subset(train_ds, train_idxs)

        # Partition train_ds across clients
        if isinstance(train_ds, Subset):
            base_len = len(train_ds)
            base_idxs = train_ds.indices
            part_rel = self._iid_indices(base_len, spec)
            part_abs = [base_idxs[i] for i in part_rel]
            train_part = Subset(train_ds.dataset, part_abs)
        else:
            part_rel = self._iid_indices(len(train_ds), spec)
            train_part = Subset(train_ds, part_rel)

        train_dl = DataLoader(
            train_part,
            batch_size=spec.batch_size,
            shuffle=spec.shuffle,
            num_workers=spec.num_workers,
            drop_last=spec.drop_last,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=spec.num_workers,
            drop_last=False,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=spec.num_workers,
            drop_last=False,
        )
        return train_dl, val_dl, test_dl
