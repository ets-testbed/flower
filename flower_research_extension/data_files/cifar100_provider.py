from pathlib import Path
from typing import Tuple, Optional
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .base import DatasetProvider

class CIFAR100Provider(DatasetProvider):
    @property
    def name(self) -> str:
        return "cifar100"

    @property
    def num_classes(self) -> int:
        return 100

    def _transforms(self):
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
        ])
        return train_tf, test_tf

    def prepare(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        datasets.CIFAR100(root=str(root), train=True, download=True)
        datasets.CIFAR100(root=str(root), train=False, download=True)

    def raw_datasets(self, root: Path) -> Tuple[Dataset, Optional[Dataset], Dataset]:
        tr_tf, te_tf = self._transforms()
        train = datasets.CIFAR100(root=str(root), train=True, transform=tr_tf, download=False)
        test = datasets.CIFAR100(root=str(root), train=False, transform=te_tf, download=False)
        return train, None, test
