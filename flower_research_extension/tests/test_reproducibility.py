from pathlib import Path
import pickle
import random
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from flower_research_extension.data_files.base import DatasetProvider, PartitionSpec
from flower_research_extension.utils.reproducibility import seed_everything


class _DummyProvider(DatasetProvider):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def num_classes(self) -> int:
        return 2

    def prepare(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)

    def raw_datasets(self, root: Path) -> tuple[Dataset, Dataset | None, Dataset]:
        x = torch.arange(0, 100, dtype=torch.float32).unsqueeze(1)
        y = (torch.arange(0, 100) % 2).to(torch.long)
        train = TensorDataset(x, y)
        test = TensorDataset(x.clone(), y.clone())
        return train, None, test


class TestReproducibility(unittest.TestCase):
    def _partition_sizes(
        self,
        provider: DatasetProvider,
        root: Path,
        *,
        num_partitions: int,
        seed: int,
        distribution: str,
        **extra,
    ) -> list[int]:
        sizes: list[int] = []
        for pid in range(num_partitions):
            spec = PartitionSpec(
                partition_id=pid,
                num_partitions=num_partitions,
                batch_size=8,
                seed=seed,
                shuffle=True,
                num_workers=0,
                distribution=distribution,
                **extra,
            )
            train_dl, _, _ = provider.partition(root, spec)
            sizes.append(len(train_dl.dataset.indices))
        return sizes

    def test_seed_everything_is_repeatable(self) -> None:
        seed_everything(1234)
        a = random.random()
        b = float(np.random.rand())
        c = float(torch.rand(1).item())

        seed_everything(1234)
        self.assertEqual(a, random.random())
        self.assertEqual(b, float(np.random.rand()))
        self.assertEqual(c, float(torch.rand(1).item()))

    def test_worker_init_fn_is_pickle_safe(self) -> None:
        provider = _DummyProvider()
        spec = PartitionSpec(partition_id=0, num_partitions=2, seed=123, num_workers=2)
        init_fn = provider._make_worker_init_fn(provider._partition_seed(spec, offset=101))
        blob = pickle.dumps(init_fn)
        restored = pickle.loads(blob)
        self.assertTrue(callable(restored))

    def test_partition_and_val_split_are_deterministic(self) -> None:
        provider = _DummyProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider.prepare(root)

            spec = PartitionSpec(
                partition_id=0,
                num_partitions=5,
                batch_size=8,
                seed=77,
                shuffle=True,
                num_workers=0,
            )
            train_1, val_1, _ = provider.partition(root, spec)
            train_2, val_2, _ = provider.partition(root, spec)

            self.assertEqual(train_1.dataset.indices, train_2.dataset.indices)
            self.assertEqual(val_1.dataset.indices, val_2.dataset.indices)

            changed_spec = PartitionSpec(
                partition_id=0,
                num_partitions=5,
                batch_size=8,
                seed=78,
                shuffle=True,
                num_workers=0,
            )
            train_3, _, _ = provider.partition(root, changed_spec)
            self.assertNotEqual(train_1.dataset.indices, train_3.dataset.indices)

    def test_dirichlet_partition_is_deterministic(self) -> None:
        provider = _DummyProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider.prepare(root)

            spec = PartitionSpec(
                partition_id=1,
                num_partitions=5,
                batch_size=8,
                seed=99,
                shuffle=True,
                num_workers=0,
                distribution="dirichlet",
                dirichlet_alpha=0.3,
            )
            train_1, _, _ = provider.partition(root, spec)
            train_2, _, _ = provider.partition(root, spec)
            self.assertEqual(train_1.dataset.indices, train_2.dataset.indices)

    def test_label_skew_restricts_classes_per_client(self) -> None:
        provider = _DummyProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider.prepare(root)

            spec = PartitionSpec(
                partition_id=0,
                num_partitions=4,
                batch_size=8,
                seed=11,
                shuffle=True,
                num_workers=0,
                distribution="label_skew",
                label_skew_classes=1,
            )
            train_dl, _, _ = provider.partition(root, spec)

            labels = []
            for _, y in train_dl.dataset:
                labels.append(int(y.item() if isinstance(y, torch.Tensor) else y))
            self.assertLessEqual(len(set(labels)), 1)

    def test_shard_partition_is_deterministic(self) -> None:
        provider = _DummyProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider.prepare(root)

            spec = PartitionSpec(
                partition_id=2,
                num_partitions=5,
                batch_size=8,
                seed=15,
                shuffle=True,
                num_workers=0,
                distribution="shard",
                shard_num_shards_per_partition=2,
            )
            train_1, _, _ = provider.partition(root, spec)
            train_2, _, _ = provider.partition(root, spec)
            self.assertEqual(train_1.dataset.indices, train_2.dataset.indices)

    def test_size_weights_change_partition_size(self) -> None:
        provider = _DummyProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider.prepare(root)

            spec_small = PartitionSpec(
                partition_id=0,
                num_partitions=3,
                batch_size=8,
                seed=22,
                shuffle=True,
                num_workers=0,
                distribution="size",
                size_partition_weights=(1.0, 1.0, 1.0),
            )
            spec_large = PartitionSpec(
                partition_id=0,
                num_partitions=3,
                batch_size=8,
                seed=22,
                shuffle=True,
                num_workers=0,
                distribution="size",
                size_partition_weights=(5.0, 1.0, 1.0),
            )
            train_small, _, _ = provider.partition(root, spec_small)
            train_large, _, _ = provider.partition(root, spec_large)
            self.assertGreater(len(train_large.dataset.indices), len(train_small.dataset.indices))

    def test_size_distribution_with_zero_weights_is_stable(self) -> None:
        provider = _DummyProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider.prepare(root)
            sizes = self._partition_sizes(
                provider,
                root,
                num_partitions=6,
                seed=25,
                distribution="size",
                size_partition_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
            self.assertTrue(all(size > 0 for size in sizes))
            self.assertEqual(sum(sizes), 90)

    def test_linear_square_exponential_modes_are_monotonic(self) -> None:
        provider = _DummyProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider.prepare(root)
            for mode in ("linear", "square", "exponential"):
                sizes = self._partition_sizes(
                    provider,
                    root,
                    num_partitions=5,
                    seed=26,
                    distribution=mode,
                )
                self.assertEqual(sorted(sizes), sizes)
                self.assertEqual(sum(sizes), 90)

    def test_inner_dirichlet_respects_size_bias(self) -> None:
        provider = _DummyProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider.prepare(root)
            sizes = self._partition_sizes(
                provider,
                root,
                num_partitions=4,
                seed=27,
                distribution="inner_dirichlet",
                inner_dirichlet_alpha=0.4,
                size_partition_weights=(10.0, 1.0, 1.0, 1.0),
            )
            self.assertGreater(sizes[0], sizes[-1])
            self.assertEqual(sum(sizes), 90)

    def test_distribution_matrix_mode_is_deterministic(self) -> None:
        provider = _DummyProvider()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider.prepare(root)
            kwargs = {
                "num_partitions": 4,
                "seed": 28,
                "distribution": "distribution",
                "distribution_matrix": (
                    (0.7, 0.3),
                    (0.6, 0.4),
                    (0.4, 0.6),
                    (0.3, 0.7),
                ),
            }
            first = self._partition_sizes(provider, root, **kwargs)
            second = self._partition_sizes(provider, root, **kwargs)
            self.assertEqual(first, second)
            self.assertEqual(sum(first), 90)


if __name__ == "__main__":
    unittest.main()
