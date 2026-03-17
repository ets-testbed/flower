from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from flower_research_extension.plugins.wandb_logger import WandBLogger


class WandBRunConfigTest(unittest.TestCase):
    @patch("flower_research_extension.plugins.wandb_logger.wandb")
    def test_wandb_logger_updates_flattened_config(self, wandb_mock: MagicMock) -> None:
        wandb_mock.config = MagicMock()

        logger = WandBLogger(
            exp_dir="results/wandb",
            project="flower-federated",
            run_name="test-run-config",
        )

        logger.on_training_start(
            {
                "dataset": {"name": "mnist", "num_classes": 10},
                "federated": {"num_rounds": 5},
                "resolved_args": {"model_builder": "resnet18"},
            }
        )

        wandb_mock.config.update.assert_called_once()
        args, kwargs = wandb_mock.config.update.call_args
        flat = args[0]
        self.assertEqual(flat["dataset.name"], "mnist")
        self.assertEqual(flat["dataset.num_classes"], 10)
        self.assertEqual(flat["federated.num_rounds"], 5)
        self.assertEqual(flat["resolved_args.model_builder"], "resnet18")
        self.assertTrue(kwargs["allow_val_change"])


if __name__ == "__main__":
    unittest.main()
