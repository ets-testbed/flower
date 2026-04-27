import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flower_research_extension.experiments.run_experiment import get_model_builder, parse_args
from flower_research_extension.training import make_fit_config_fn


class ExperimentConfigTests(unittest.TestCase):
    def test_yaml_defaults_and_cli_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset: mnist",
                        "local_epochs: 3",
                        "lr: 0.05",
                        "momentum: 0.7",
                        "disable_wandb: true",
                    ]
                ),
                encoding="utf-8",
            )

            args = parse_args(["--config", str(config_path), "--lr", "0.2"])

        self.assertEqual(args.dataset, "mnist")
        self.assertEqual(args.local_epochs, 3)
        self.assertEqual(args.lr, 0.2)
        self.assertEqual(args.momentum, 0.7)
        self.assertTrue(args.disable_wandb)

        fit_config = make_fit_config_fn(
            local_epochs=args.local_epochs,
            lr=args.lr,
            momentum=args.momentum,
        )(server_round=2)
        self.assertEqual(
            fit_config,
            {
                "server_round": 2,
                "local_epochs": 3,
                "lr": 0.2,
                "momentum": 0.7,
            },
        )

    def test_model_builders_forward(self) -> None:
        net = get_model_builder("net")(10)
        resnet = get_model_builder("resnet18")(7)

        self.assertEqual(tuple(net(torch.randn(2, 3, 32, 32)).shape), (2, 10))
        self.assertEqual(tuple(resnet(torch.randn(2, 3, 32, 32)).shape), (2, 7))


if __name__ == "__main__":
    unittest.main()
