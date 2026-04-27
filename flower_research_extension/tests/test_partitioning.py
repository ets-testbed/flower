import sys
import unittest
from pathlib import Path

from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flower_research_extension.data_files.base import DatasetProvider, PartitionSpec


class DummyDataset(Dataset):
    def __init__(self) -> None:
        self.targets = [i // 10 for i in range(40)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        return idx, self.targets[idx]


class DummyProvider(DatasetProvider):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def num_classes(self) -> int:
        return 4

    def prepare(self, root: Path) -> None:
        return None

    def raw_datasets(self, root: Path):
        ds = DummyDataset()
        return ds, None, ds


class PartitioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = DummyProvider()
        self.root = Path(".")

    def _indices(self, mode: str, partition_id: int, **kwargs):
        spec = PartitionSpec(
            partition_id=partition_id,
            num_partitions=4,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            distribution=mode,
            **kwargs,
        )
        train_dl, _, _ = self.provider.partition(self.root, spec)
        return list(train_dl.dataset.indices)

    def test_iid_partition_sizes(self) -> None:
        sizes = [len(self._indices("iid", pid)) for pid in range(4)]
        self.assertEqual(sizes, [9, 9, 9, 9])

    def test_dirichlet_partition_is_deterministic_and_nonempty(self) -> None:
        first = [self._indices("dirichlet", pid, dirichlet_alpha=0.3) for pid in range(4)]
        second = [self._indices("dirichlet", pid, dirichlet_alpha=0.3) for pid in range(4)]

        self.assertEqual(first, second)
        self.assertTrue(all(first))

    def test_shard_partition_skews_labels(self) -> None:
        label_sets = []
        for pid in range(4):
            part = self._indices("shard", pid, shard_num_shards_per_partition=2)
            labels = [self.provider.raw_datasets(self.root)[0].targets[i] for i in part]
            label_sets.append(set(labels))

        self.assertTrue(all(label_sets))
        self.assertTrue(any(len(labels) <= 2 for labels in label_sets))


if __name__ == "__main__":
    unittest.main()
