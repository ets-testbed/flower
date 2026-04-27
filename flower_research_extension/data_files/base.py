# =========================================
# file: flower_research_extension/data_files/base.py
# =========================================
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
import math
import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from flower_research_extension.utils.reproducibility import make_torch_generator


def _seed_worker_with_base(worker_id: int, *, base_seed: int) -> None:
    """Top-level worker init function so multiprocessing can pickle it."""
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)

@dataclass(frozen=True)
class PartitionSpec:
    partition_id: int
    num_partitions: int
    batch_size: int = 64
    seed: int = 42
    shuffle: bool = True
    num_workers: int = 2
    drop_last: bool = False
    distribution: str = "iid"
    dirichlet_alpha: float = 0.5
    label_skew_classes: int = 2
    shard_num_shards_per_partition: int = 2
    inner_dirichlet_alpha: float = 0.5
    size_partition_weights: Optional[Tuple[float, ...]] = None
    distribution_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None

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

    def _partition_seed(self, spec: PartitionSpec, offset: int = 0) -> int:
        """Compute a stable seed per partition to keep runs repeatable."""
        return int(spec.seed) + int(spec.partition_id) * 1000 + int(offset)

    def _iid_indices(self, n: int, spec: PartitionSpec) -> list[int]:
        """Deterministic IID split into `num_partitions` shards (equal-sized; last shard may be larger)."""
        g = random.Random(self._partition_seed(spec, offset=17))
        idxs = list(range(n))
        g.shuffle(idxs)
        per = math.ceil(n / spec.num_partitions)
        start = spec.partition_id * per
        end = min(start + per, n)
        return idxs[start:end]

    def _split_train_val(self, train_ds: Dataset, spec: PartitionSpec) -> Tuple[Dataset, Dataset]:
        """Create a deterministic 90/10 train/val split when no explicit val set exists."""
        total = len(train_ds)
        val_count = max(1, int(0.1 * total))
        indices = list(range(total))
        random.Random(self._partition_seed(spec, offset=31)).shuffle(indices)
        val_idxs = indices[:val_count]
        train_idxs = indices[val_count:]
        return Subset(train_ds, train_idxs), Subset(train_ds, val_idxs)

    def _make_worker_init_fn(self, base_seed: int) -> Callable[[int], None]:
        # Use a top-level function + partial so DataLoader workers are pickle-safe
        # on Windows (spawn) and newer Python multiprocessing defaults.
        return partial(_seed_worker_with_base, base_seed=int(base_seed))

    def _dataset_labels(self, dataset: Dataset) -> list[int]:
        """Extract integer class labels from common torchvision-style datasets."""
        if isinstance(dataset, Subset):
            base_labels = self._dataset_labels(dataset.dataset)
            return [int(base_labels[i]) for i in dataset.indices]

        if isinstance(dataset, ConcatDataset):
            labels: list[int] = []
            for ds in dataset.datasets:
                labels.extend(self._dataset_labels(ds))
            return labels

        targets = getattr(dataset, "targets", None)
        if targets is None:
            targets = getattr(dataset, "labels", None)
        if targets is not None:
            if hasattr(targets, "tolist"):
                return [int(v) for v in targets.tolist()]
            return [int(v) for v in targets]

        tensors = getattr(dataset, "tensors", None)
        if tensors is not None and len(tensors) >= 2:
            labels_tensor = tensors[1]
            if hasattr(labels_tensor, "tolist"):
                return [int(v) for v in labels_tensor.tolist()]

        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, dict):
                label = sample.get("label", sample.get("target"))
            else:
                label = sample[1]
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(int(label))
        return labels

    def _weights_to_counts(self, weights: list[float], n: int) -> list[int]:
        """Convert positive weights into deterministic sample counts summing to n."""
        if n <= 0:
            return [0 for _ in weights]
        arr = np.array(weights, dtype=np.float64)
        if np.any(arr < 0):
            raise ValueError("Partition weights must be non-negative")
        if np.all(arr == 0):
            arr = np.ones_like(arr)
        arr = arr / arr.sum()
        raw = arr * n
        counts = np.floor(raw).astype(int)

        remainder = int(n - int(counts.sum()))
        if remainder > 0:
            frac = raw - counts
            order = np.lexsort((np.arange(len(arr)), -frac))
            for i in range(remainder):
                counts[order[i % len(order)]] += 1

        # Guarantee at least one sample per partition when feasible.
        if n >= len(counts):
            empties = [i for i, c in enumerate(counts) if c == 0]
            for empty in empties:
                donor_candidates = [i for i, c in enumerate(counts) if c > 1]
                if not donor_candidates:
                    break
                donor = max(donor_candidates, key=lambda i: counts[i])
                counts[donor] -= 1
                counts[empty] += 1

        return [int(v) for v in counts.tolist()]

    def _partition_map_from_counts(self, n: int, counts: list[int], seed: int) -> list[list[int]]:
        indices = list(range(n))
        random.Random(seed).shuffle(indices)
        mapping: list[list[int]] = []
        cursor = 0
        for cnt in counts:
            end = min(cursor + int(cnt), n)
            mapping.append(indices[cursor:end])
            cursor = end
        # If rounding/edge effects left remaining points, append to last partition.
        if cursor < n and mapping:
            mapping[-1].extend(indices[cursor:])
        return mapping

    def _size_weights(self, spec: PartitionSpec, mode: str) -> list[float]:
        num = int(spec.num_partitions)
        if mode == "size":
            if spec.size_partition_weights is not None and len(spec.size_partition_weights) > 0:
                if len(spec.size_partition_weights) != num:
                    raise ValueError(
                        f"size_partition_weights length ({len(spec.size_partition_weights)}) must equal num_partitions ({num})"
                    )
                return [float(v) for v in spec.size_partition_weights]
            return [1.0 for _ in range(num)]
        if mode == "linear":
            return [float(i + 1) for i in range(num)]
        if mode == "square":
            return [float((i + 1) ** 2) for i in range(num)]
        if mode == "exponential":
            return [float(2**i) for i in range(num)]
        raise ValueError(f"Unsupported size mode '{mode}'")

    def _compute_partition_map_iid(self, n: int, spec: PartitionSpec) -> list[list[int]]:
        indices = list(range(n))
        random.Random(int(spec.seed) + 17).shuffle(indices)
        per = math.ceil(n / spec.num_partitions)
        mapping: list[list[int]] = []
        for pid in range(spec.num_partitions):
            start = pid * per
            end = min(start + per, n)
            mapping.append(indices[start:end])
        return mapping

    def _compute_partition_map_size_mode(self, n: int, spec: PartitionSpec, mode: str) -> list[list[int]]:
        weights = self._size_weights(spec, mode)
        counts = self._weights_to_counts(weights, n)
        mapping = self._partition_map_from_counts(n, counts, seed=int(spec.seed) + 43)
        return self._ensure_non_empty_partitions(mapping, n, seed=int(spec.seed) + 47)

    def _compute_partition_map_shard(self, labels: list[int], spec: PartitionSpec, n: int) -> list[list[int]]:
        num_shards = max(spec.num_partitions, spec.num_partitions * int(spec.shard_num_shards_per_partition))
        sorted_indices = sorted(range(n), key=lambda idx: int(labels[idx]))
        shards = [list(chunk) for chunk in np.array_split(np.array(sorted_indices, dtype=np.int64), num_shards)]
        shards = [list(map(int, shard.tolist() if hasattr(shard, "tolist") else shard)) for shard in shards]
        shards = [s for s in shards if len(s) > 0]
        if not shards:
            return [[] for _ in range(spec.num_partitions)]

        rng = random.Random(int(spec.seed) + 53)
        shard_ids = list(range(len(shards)))
        rng.shuffle(shard_ids)
        mapping: list[list[int]] = [[] for _ in range(spec.num_partitions)]
        for i, shard_id in enumerate(shard_ids):
            pid = i % spec.num_partitions
            mapping[pid].extend(shards[shard_id])

        return self._ensure_non_empty_partitions(mapping, n, seed=int(spec.seed) + 59)

    def _compute_partition_map_inner_dirichlet(
        self, labels: list[int], spec: PartitionSpec, n: int
    ) -> list[list[int]]:
        weights = self._size_weights(spec, "size")
        capacities = self._weights_to_counts(weights, n)
        remaining = capacities[:]
        mapping: list[list[int]] = [[] for _ in range(spec.num_partitions)]
        rng_np = np.random.default_rng(int(spec.seed) + 61)
        unique_classes = sorted(set(int(lbl) for lbl in labels))

        for cls in unique_classes:
            cls_indices = [i for i, lbl in enumerate(labels) if int(lbl) == cls]
            rng_np.shuffle(cls_indices)
            for idx in cls_indices:
                eligible = [pid for pid, cap in enumerate(remaining) if cap > 0]
                if not eligible:
                    break
                probs = rng_np.dirichlet(
                    np.full(shape=len(eligible), fill_value=float(spec.inner_dirichlet_alpha), dtype=np.float64)
                )
                pick_rel = int(rng_np.choice(len(eligible), p=probs))
                pick_pid = eligible[pick_rel]
                mapping[pick_pid].append(int(idx))
                remaining[pick_pid] -= 1

        # Fill any remaining capacity with yet-unassigned indices.
        assigned = set(i for part in mapping for i in part)
        leftovers = [i for i in range(n) if i not in assigned]
        rng = random.Random(int(spec.seed) + 67)
        rng.shuffle(leftovers)
        for pid in range(spec.num_partitions):
            while remaining[pid] > 0 and leftovers:
                mapping[pid].append(leftovers.pop())
                remaining[pid] -= 1

        return self._ensure_non_empty_partitions(mapping, n, seed=int(spec.seed) + 71)

    def _compute_partition_map_distribution_matrix(
        self, labels: list[int], spec: PartitionSpec, n: int
    ) -> list[list[int]]:
        if spec.distribution_matrix is None:
            raise ValueError("distribution_matrix must be provided for distribution mode")
        matrix = np.array(spec.distribution_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != spec.num_partitions:
            raise ValueError(
                "distribution_matrix must be a 2D array with shape [num_partitions, num_classes]"
            )

        unique_classes = sorted(set(int(lbl) for lbl in labels))
        class_to_col = {cls: i for i, cls in enumerate(unique_classes)}
        if matrix.shape[1] < len(unique_classes):
            raise ValueError(
                f"distribution_matrix has {matrix.shape[1]} columns but requires at least {len(unique_classes)}"
            )

        mapping: list[list[int]] = [[] for _ in range(spec.num_partitions)]
        rng_np = np.random.default_rng(int(spec.seed) + 73)
        for cls in unique_classes:
            cls_indices = [i for i, lbl in enumerate(labels) if int(lbl) == cls]
            if not cls_indices:
                continue
            rng_np.shuffle(cls_indices)
            col = class_to_col[cls]
            probs = matrix[:, col].astype(np.float64)
            if np.any(probs < 0):
                raise ValueError("distribution_matrix entries must be non-negative")
            if float(probs.sum()) <= 0:
                probs = np.ones_like(probs, dtype=np.float64)
            probs = probs / probs.sum()
            counts = rng_np.multinomial(len(cls_indices), probs)
            cursor = 0
            for pid, cnt in enumerate(counts.tolist()):
                cnt_i = int(cnt)
                if cnt_i <= 0:
                    continue
                end = cursor + cnt_i
                mapping[pid].extend(cls_indices[cursor:end])
                cursor = end

        return self._ensure_non_empty_partitions(mapping, n, seed=int(spec.seed) + 79)

    def _compute_partition_map_dirichlet(
        self, labels: list[int], spec: PartitionSpec, n: int
    ) -> list[list[int]]:
        mapping: list[list[int]] = [[] for _ in range(spec.num_partitions)]
        rng = np.random.default_rng(int(spec.seed) + 23)
        unique_classes = sorted(set(int(lbl) for lbl in labels))

        for cls in unique_classes:
            cls_indices = [i for i, lbl in enumerate(labels) if int(lbl) == cls]
            if not cls_indices:
                continue
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(
                np.full(shape=spec.num_partitions, fill_value=float(spec.dirichlet_alpha), dtype=np.float64)
            )
            counts = rng.multinomial(len(cls_indices), proportions)
            start = 0
            for pid, cnt in enumerate(counts):
                end = start + int(cnt)
                if end > start:
                    mapping[pid].extend(cls_indices[start:end])
                start = end

        return self._ensure_non_empty_partitions(mapping, n, seed=int(spec.seed) + 29)

    def _compute_partition_map_label_skew(
        self, labels: list[int], spec: PartitionSpec, n: int
    ) -> list[list[int]]:
        mapping: list[list[int]] = [[] for _ in range(spec.num_partitions)]
        unique_classes = sorted(set(int(lbl) for lbl in labels))
        if not unique_classes:
            return mapping

        k = max(1, min(int(spec.label_skew_classes), len(unique_classes)))
        class_to_partitions: dict[int, list[int]] = {}
        for cls_idx, cls in enumerate(unique_classes):
            eligible: list[int] = []
            for pid in range(spec.num_partitions):
                start = pid % len(unique_classes)
                allowed = [unique_classes[(start + offset) % len(unique_classes)] for offset in range(k)]
                if cls in allowed:
                    eligible.append(pid)
            if not eligible:
                eligible = [cls_idx % spec.num_partitions]
            class_to_partitions[cls] = eligible

        rng = random.Random(int(spec.seed) + 37)
        for cls in unique_classes:
            cls_indices = [i for i, lbl in enumerate(labels) if int(lbl) == cls]
            rng.shuffle(cls_indices)
            targets = class_to_partitions[cls]
            for i, idx in enumerate(cls_indices):
                pid = targets[i % len(targets)]
                mapping[pid].append(idx)

        return self._ensure_non_empty_partitions(mapping, n, seed=int(spec.seed) + 41)

    def _ensure_non_empty_partitions(
        self, mapping: list[list[int]], n: int, *, seed: int
    ) -> list[list[int]]:
        """Move a sample from largest partitions to any empty partition."""
        rng = random.Random(seed)
        all_indices = set(range(n))
        assigned_indices = set(i for part in mapping for i in part)
        unassigned = list(all_indices - assigned_indices)

        # First, fill empties from unassigned samples (if any).
        for pid, part in enumerate(mapping):
            if part:
                continue
            if unassigned:
                idx = unassigned.pop()
                part.append(idx)

        # Then rebalance from largest partitions if empties remain.
        for pid, part in enumerate(mapping):
            if part:
                continue
            donors = [d for d, vals in enumerate(mapping) if len(vals) > 1]
            if not donors:
                continue
            donor = max(donors, key=lambda d: len(mapping[d]))
            pick = rng.randrange(len(mapping[donor]))
            part.append(mapping[donor].pop(pick))

        # Keep deterministic order inside partitions.
        for part in mapping:
            part.sort()
        return mapping

    def _partition_indices(self, train_ds: Dataset, spec: PartitionSpec) -> list[int]:
        n = len(train_ds)
        if n == 0:
            return []
        if spec.distribution == "iid":
            partition_map = self._compute_partition_map_iid(n, spec)
            return partition_map[spec.partition_id]
        if spec.distribution in {"linear", "square", "exponential", "size"}:
            partition_map = self._compute_partition_map_size_mode(n, spec, mode=spec.distribution)
            return partition_map[spec.partition_id]

        labels = self._dataset_labels(train_ds)
        if len(labels) != n:
            raise ValueError("Label extraction failed: labels length does not match dataset length")

        if spec.distribution == "dirichlet":
            partition_map = self._compute_partition_map_dirichlet(labels, spec, n)
            return partition_map[spec.partition_id]
        if spec.distribution in {"label_skew", "pathological"}:
            partition_map = self._compute_partition_map_label_skew(labels, spec, n)
            return partition_map[spec.partition_id]
        if spec.distribution == "shard":
            partition_map = self._compute_partition_map_shard(labels, spec, n)
            return partition_map[spec.partition_id]
        if spec.distribution == "inner_dirichlet":
            partition_map = self._compute_partition_map_inner_dirichlet(labels, spec, n)
            return partition_map[spec.partition_id]
        if spec.distribution == "distribution":
            partition_map = self._compute_partition_map_distribution_matrix(labels, spec, n)
            return partition_map[spec.partition_id]
        raise ValueError(f"Unsupported distribution '{spec.distribution}'")

    def partition(
        self,
        root: Path,
        spec: PartitionSpec,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for a user/client partition."""
        train_ds, val_ds, test_ds = self.raw_datasets(root)

        # If no explicit val set, carve a deterministic 10% split from train.
        if val_ds is None:
            train_ds, val_ds = self._split_train_val(train_ds, spec)

        # Partition train_ds across clients according to the selected distribution.
        part_rel = self._partition_indices(train_ds, spec)
        if isinstance(train_ds, Subset):
            base_idxs = list(train_ds.indices)
            part_abs = [base_idxs[i] for i in part_rel]
            train_part = Subset(train_ds.dataset, part_abs)
        else:
            train_part = Subset(train_ds, part_rel)

        base_seed = self._partition_seed(spec, offset=101)
        # Flower+Ray on Windows can fail when DataLoader uses multiprocessing
        # workers (spawn/Pipe handle errors). Force single-process loading there.
        effective_num_workers = int(spec.num_workers)
        if os.name == "nt":
            effective_num_workers = 0

        loader_common_kwargs = {
            "num_workers": effective_num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": self._make_worker_init_fn(base_seed),
            "generator": make_torch_generator(base_seed),
        }

        train_dl = DataLoader(
            train_part,
            batch_size=spec.batch_size,
            shuffle=spec.shuffle,
            drop_last=spec.drop_last,
            **loader_common_kwargs,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=spec.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_common_kwargs,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=spec.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_common_kwargs,
        )
        return train_dl, val_dl, test_dl
