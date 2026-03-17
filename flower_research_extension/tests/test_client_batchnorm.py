import unittest
import sys
import types

import torch
from torch.utils.data import DataLoader, TensorDataset

# Keep this test runnable in lightweight environments where Flower may be absent.
if "flwr" not in sys.modules:
    flwr_mod = types.ModuleType("flwr")
    flwr_client_mod = types.ModuleType("flwr.client")

    class _NumPyClient:
        def to_client(self):
            return self

    class _Client:
        pass

    flwr_client_mod.NumPyClient = _NumPyClient
    flwr_client_mod.Client = _Client
    flwr_mod.client = flwr_client_mod
    sys.modules["flwr"] = flwr_mod
    sys.modules["flwr.client"] = flwr_client_mod

from flower_research_extension.client import _DLClient
from flower_research_extension.experiments.run_experiment import get_model_builder
from flower_research_extension.model import get_parameters


class TestClientBatchNormSafety(unittest.TestCase):
    def test_fit_handles_single_sample_batches_for_bn_models(self) -> None:
        model = get_model_builder("resnet18")(10)
        x = torch.randn(1, 3, 32, 32)
        y = torch.tensor([0], dtype=torch.long)
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        client = _DLClient(
            model=model,
            device=torch.device("cpu"),
            loaders=(loader, loader, loader),
        )
        initial = get_parameters(model)
        updated, num_examples, metrics = client.fit(
            initial, {"local_epochs": 1, "lr": 0.01, "momentum": 0.9}
        )

        self.assertEqual(num_examples, 1)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertEqual(len(updated), len(initial))


if __name__ == "__main__":
    unittest.main()
