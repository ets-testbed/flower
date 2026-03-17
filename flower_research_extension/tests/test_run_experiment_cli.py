import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from flower_research_extension.experiments import run_experiment as cli


class TestRunExperimentCli(unittest.TestCase):
    def test_model_choices_match_model_registry(self) -> None:
        parser = cli.build_parser()
        model_action = parser._option_string_actions["--model"]
        self.assertEqual(sorted(model_action.choices), sorted(["auto", *cli.MODEL_BUILDERS.keys()]))

    def test_validate_rejects_invalid_fraction_fit(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--fraction_fit", "0"])
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli._validate_args(parser, args)

    def test_validate_rejects_too_many_fit_clients(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--num_partitions", "2", "--min_fit_clients", "3"])
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli._validate_args(parser, args)

    def test_auto_run_name_includes_model_and_seed(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "--wandb_run_name",
                "auto",
                "--dataset",
                "mnist",
                "--model",
                "auto",
                "--seed",
                "7",
                "--distribution",
                "dirichlet",
            ]
        )
        cli._validate_args(parser, args)
        normalized = cli._normalize_args(args)
        self.assertIn("mnet", normalized.wandb_run_name)
        self.assertIn("ddirichlet", normalized.wandb_run_name)
        self.assertIn("_s7", normalized.wandb_run_name)

    def test_auto_model_resolution_uses_dataset_default(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--dataset", "cifar100", "--model", "auto"])
        cli._validate_args(parser, args)
        normalized = cli._normalize_args(args)
        self.assertEqual(normalized.model, "densenet121")
        self.assertEqual(normalized.requested_model, "auto")

    def test_model_fit_profile_is_exposed_in_normalized_config(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "cifar10",
                "--model",
                "resnet50",
            ]
        )
        cli._validate_args(parser, args)
        normalized = cli._normalize_args(args)
        self.assertEqual(normalized.model_fit_profile, "heavy")

    def test_validate_rejects_incompatible_model_for_dataset(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--dataset", "cifar100", "--model", "net"])
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli._validate_args(parser, args)

    def test_validate_rejects_invalid_dirichlet_alpha(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--distribution", "dirichlet", "--dirichlet_alpha", "0"])
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli._validate_args(parser, args)

    def test_validate_rejects_invalid_label_skew_classes(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--distribution", "label_skew", "--label_skew_classes", "0"])
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli._validate_args(parser, args)

    def test_validate_rejects_invalid_shard_count(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--distribution", "shard", "--shard_num_shards_per_partition", "0"])
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli._validate_args(parser, args)

    def test_validate_rejects_invalid_inner_dirichlet_alpha(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--distribution", "inner_dirichlet", "--inner_dirichlet_alpha", "0"])
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli._validate_args(parser, args)

    def test_validate_rejects_invalid_size_weights_len(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            ["--distribution", "size", "--num_partitions", "3", "--size_partition_weights", "1,2"]
        )
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli._validate_args(parser, args)

    def test_validate_rejects_missing_distribution_matrix(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--distribution", "distribution"])
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli._validate_args(parser, args)

    def test_validate_accepts_distribution_matrix(self) -> None:
        parser = cli.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "matrix.json"
            p.write_text(
                json.dumps(
                    [
                        [0.1] * 10,
                        [0.2] * 10,
                        [0.3] * 10,
                    ]
                ),
                encoding="utf-8",
            )
            args = parser.parse_args(
                ["--distribution", "distribution", "--num_partitions", "3", "--distribution_matrix_json", str(p)]
            )
            cli._validate_args(parser, args)

    def test_validate_rejects_distribution_matrix_with_too_few_columns(self) -> None:
        parser = cli.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "matrix.json"
            p.write_text(json.dumps([[0.7, 0.3], [0.3, 0.7], [0.5, 0.5]]), encoding="utf-8")
            args = parser.parse_args(
                ["--dataset", "cifar10", "--distribution", "distribution", "--num_partitions", "3", "--distribution_matrix_json", str(p)]
            )
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli._validate_args(parser, args)

    def test_main_dry_run_prints_json_config(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(io.StringIO()):
            code = cli.main(["--dry_run", "--dataset", "mnist", "--model", "auto"])
        self.assertEqual(code, 0)
        cfg = json.loads(buffer.getvalue())
        self.assertTrue(cfg["dry_run"])
        self.assertEqual(cfg["dataset"], "mnist")
        self.assertEqual(cfg["requested_model"], "auto")
        self.assertEqual(cfg["model"], "net")
        self.assertEqual(cfg["model_fit_profile"], "light")
        self.assertEqual(cfg["model_builder"], "net")

    def test_main_list_capabilities_prints_json(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(io.StringIO()):
            code = cli.main(["--list_capabilities"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertIn("datasets", payload)
        self.assertIn("distributions", payload)
        self.assertIn("distribution_descriptions", payload)
        self.assertIn("models", payload)
        self.assertIn("model_fit_profile", payload)
        self.assertIn("dataset_model_policies", payload)
        self.assertIn("cifar100", payload["dataset_model_policies"])
        self.assertIn("resnet50", payload["models"])


if __name__ == "__main__":
    unittest.main()
