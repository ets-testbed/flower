import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from flower_research_extension.experiments import run_experiment as cli


class TestExperimentMatrixSmoke(unittest.TestCase):
    def _run_dry(self, argv: list[str]) -> dict:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = cli.main(["--dry_run", *argv])
        self.assertEqual(code, 0)
        return json.loads(out.getvalue())

    def test_dry_run_matrix_common_combinations(self) -> None:
        cases = [
            ["--dataset", "mnist", "--model", "auto", "--distribution", "iid"],
            [
                "--dataset",
                "cifar10",
                "--model",
                "resnet50",
                "--distribution",
                "dirichlet",
                "--dirichlet_alpha",
                "0.3",
            ],
            [
                "--dataset",
                "svhn",
                "--model",
                "mobilenet_v2",
                "--distribution",
                "inner_dirichlet",
                "--num_partitions",
                "4",
                "--size_partition_weights",
                "1,1,1,1",
                "--inner_dirichlet_alpha",
                "0.4",
            ],
            [
                "--dataset",
                "cifar100",
                "--model",
                "auto",
                "--distribution",
                "shard",
                "--shard_num_shards_per_partition",
                "3",
            ],
        ]
        for argv in cases:
            cfg = self._run_dry(argv)
            self.assertEqual(cfg["dry_run"], True)
            self.assertIn(cfg["model"], cli.MODEL_BUILDERS)

    def test_dry_run_distribution_matrix_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "matrix.json"
            matrix_path.write_text(
                json.dumps(
                    [
                        [1.0] * 10,
                        [1.0] * 10,
                        [1.0] * 10,
                        [1.0] * 10,
                    ]
                ),
                encoding="utf-8",
            )
            cfg = self._run_dry(
                [
                    "--dataset",
                    "cifar10",
                    "--model",
                    "resnet18",
                    "--distribution",
                    "distribution",
                    "--num_partitions",
                    "4",
                    "--distribution_matrix_json",
                    str(matrix_path),
                ]
            )
            self.assertEqual(cfg["distribution"], "distribution")
            self.assertEqual(len(cfg["distribution_matrix"]), 4)


if __name__ == "__main__":
    unittest.main()
