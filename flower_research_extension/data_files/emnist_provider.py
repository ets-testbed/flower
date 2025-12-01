from pathlib import Path
from typing import Tuple, Optional
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .base import DatasetProvider

class EMNISTBalancedProvider(DatasetProvider):
    @property
    def name(self) -> str:
        return "emnist_balanced"

    @property
    def num_classes(self) -> int:
        return 47  # EMNIST Balanced split

    def _transforms(self):
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        test_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        return train_tf, test_tf

    def prepare(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        datasets.EMNIST(root=str(root), split="balanced", train=True, download=True)
        datasets.EMNIST(root=str(root), split="balanced", train=False, download=True)

    def raw_datasets(self, root: Path) -> Tuple[Dataset, Optional[Dataset], Dataset]:
        tr_tf, te_tf = self._transforms()
        train = datasets.EMNIST(root=str(root), split="balanced", train=True, transform=tr_tf, download=False)
        test = datasets.EMNIST(root=str(root), split="balanced", train=False, transform=te_tf, download=False)
        return train, None, test
