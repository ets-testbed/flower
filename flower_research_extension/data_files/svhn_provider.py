from pathlib import Path
from typing import Tuple, Optional
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
from .base import DatasetProvider

class SVHNProvider(DatasetProvider):
    @property
    def name(self) -> str:
        return "svhn"

    @property
    def num_classes(self) -> int:
        return 10

    def _transforms(self):
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
        ])
        return train_tf, test_tf

    def prepare(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        datasets.SVHN(root=str(root), split="train", download=True)
        datasets.SVHN(root=str(root), split="extra", download=True)
        datasets.SVHN(root=str(root), split="test", download=True)

    def raw_datasets(self, root: Path) -> Tuple[Dataset, Optional[Dataset], Dataset]:
        tr_tf, te_tf = self._transforms()
        train = datasets.SVHN(root=str(root), split="train", transform=tr_tf, download=False)
        extra = datasets.SVHN(root=str(root), split="extra", transform=tr_tf, download=False)
        train = ConcatDataset([train, extra])
        test = datasets.SVHN(root=str(root), split="test", transform=te_tf, download=False)
        return train, None, test
