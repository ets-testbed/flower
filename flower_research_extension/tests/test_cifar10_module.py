import importlib
import sys
import unittest
from unittest import mock


class TestCifar10LegacyModule(unittest.TestCase):
    def test_import_has_no_dataset_download_side_effect(self) -> None:
        module_name = "flower_research_extension.data_files.cifar10"
        sys.modules.pop(module_name, None)

        with mock.patch("torchvision.datasets.CIFAR10", side_effect=AssertionError("Unexpected download call")):
            module = importlib.import_module(module_name)

        self.assertTrue(hasattr(module, "load_cifar10_partition"))


if __name__ == "__main__":
    unittest.main()

