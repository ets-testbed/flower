from pathlib import Path
from typing import Tuple, Optional
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .base import DatasetProvider

class MNISTProvider(DatasetProvider):
    @property
    def name(self) -> str:
        return "mnist"

    @property
    def num_classes(self) -> int:
        return 10

    def _transforms(self):
        # why: model expects 3x32x32 (CIFAR-like); adapt MNIST
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 1->3 channels
            transforms.Resize(32),                        # 28->32
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
        _ = datasets.MNIST(root=str(root), train=True, download=True)
        _ = datasets.MNIST(root=str(root), train=False, download=True)

    def raw_datasets(self, root: Path) -> Tuple[Dataset, Optional[Dataset], Dataset]:
        tr_tf, te_tf = self._transforms()
        train = datasets.MNIST(root=str(root), train=True, transform=tr_tf, download=False)
        test = datasets.MNIST(root=str(root), train=False, transform=te_tf, download=False)
        return train, None, test
