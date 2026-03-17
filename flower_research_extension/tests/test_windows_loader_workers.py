from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from flower_research_extension.data_files.base import DatasetProvider, PartitionSpec


class _DummyProvider(DatasetProvider):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def num_classes(self) -> int:
        return 2

    def prepare(self, root: Path) -> None:
        return None

    def raw_datasets(self, root: Path):
        x = torch.randn(40, 1, 4, 4)
        y = torch.tensor([i % 2 for i in range(40)], dtype=torch.long)
        ds = TensorDataset(x, y)
        return ds, None, ds


class WindowsLoaderWorkersTest(unittest.TestCase):
    def test_windows_forces_single_worker_dataloaders(self) -> None:
        provider = _DummyProvider()
        spec = PartitionSpec(partition_id=0, num_partitions=4, batch_size=8, num_workers=2)

        with patch("flower_research_extension.data_files.base.os.name", "nt"):
            train_dl, val_dl, test_dl = provider.partition(root=Path("."), spec=spec)

        self.assertEqual(train_dl.num_workers, 0)
        self.assertEqual(val_dl.num_workers, 0)
        self.assertEqual(test_dl.num_workers, 0)


if __name__ == "__main__":
    unittest.main()
