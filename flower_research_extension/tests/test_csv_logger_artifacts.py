from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from flower_research_extension.plugins.csv_logger import CSVLogger


class CSVLoggerArtifactsTest(unittest.TestCase):
    def test_persists_run_config_round_metrics_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(log_dir=tmpdir)

            logger.on_training_start(
                {
                    "dataset": {"name": "mnist", "num_classes": 10},
                    "model": {"resolved": "resnet18"},
                    "federated": {"num_rounds": 2},
                }
            )
            logger.on_round_end(1, {"loss": 1.23, "accuracy": 0.45, "round_time": 0.1})
            logger.on_server_evaluate(1, {"loss": 0.9, "accuracy": 0.6, "f1": 0.55})
            logger.on_client_result(1, "client-1", {"loss": 1.2, "accuracy": 0.4})
            logger.finalize()

            run_folder = Path(logger.log_folder)
            self.assertTrue((run_folder / "run_config.json").exists())
            self.assertTrue((run_folder / "round_metrics.jsonl").exists())
            self.assertTrue((run_folder / "run_summary.json").exists())

            run_config = json.loads((run_folder / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["dataset"]["name"], "mnist")
            self.assertEqual(run_config["model"]["resolved"], "resnet18")

            summary = json.loads((run_folder / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["fit_rounds"], 1)
            self.assertEqual(summary["eval_rounds"], 1)
            self.assertEqual(summary["client_result_count"], 1)
            self.assertAlmostEqual(summary["last_eval_metrics"]["accuracy"], 0.6)

            lines = (run_folder / "round_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["phase"], "fit")
            second = json.loads(lines[1])
            self.assertEqual(second["phase"], "eval")


if __name__ == "__main__":
    unittest.main()
