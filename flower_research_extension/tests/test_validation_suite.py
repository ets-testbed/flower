from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from flower_research_extension.data_files import REGISTRY
from flower_research_extension.experiments.catalog import DATASET_MODEL_POLICIES, DISTRIBUTIONS
from flower_research_extension.experiments.validation_suite import (
    SMOKE_REAL_RUNTIME_DATASETS,
    ValidationCase,
    _distribution_args,
    build_validation_cases,
    run_validation_suite,
)


_TMP_ROOT = Path(__file__).resolve().parents[2] / ".tmp_validation_tests"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class ValidationSuiteTest(unittest.TestCase):
    def _workspace_case_dir(self, name: str) -> Path:
        path = _TMP_ROOT / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_distribution_args_include_required_matrix_file(self) -> None:
        output_dir = self._workspace_case_dir("distribution_args")
        provider = REGISTRY.get("mnist")
        args = _distribution_args(
            distribution="distribution",
            dataset="mnist",
            num_classes=int(provider.num_classes),
            num_partitions=4,
            output_dir=output_dir,
        )
        self.assertIn("--distribution_matrix_json", args)
        self.assertTrue(Path(args[-1]).exists())

    def test_smoke_mode_builds_dataset_distribution_cases_and_representative_real_runs(self) -> None:
        output_dir = self._workspace_case_dir("smoke_case_coverage")
        cases = build_validation_cases(mode="smoke", num_rounds=1, output_dir=output_dir)

        case_names = {case.name for case in cases}
        self.assertIn("unit_tests", case_names)
        self.assertIn("run_commands_list", case_names)

        expected_dry = len(REGISTRY.available()) * len(DISTRIBUTIONS)
        actual_dry = sum(1 for case in cases if case.name.startswith("dry_distribution_"))
        self.assertEqual(actual_dry, expected_dry)

        actual_real_dataset_runs = sum(1 for case in cases if case.name.startswith("real_iid_"))
        self.assertEqual(actual_real_dataset_runs, len(SMOKE_REAL_RUNTIME_DATASETS))

    def test_smoke_mode_real_cases_require_ray_and_use_explicit_models(self) -> None:
        output_dir = self._workspace_case_dir("smoke_real_cases")
        cases = build_validation_cases(mode="smoke", num_rounds=1, output_dir=output_dir)

        real_cases = [case for case in cases if case.name.startswith("real_iid_")]
        self.assertEqual(len(real_cases), len(SMOKE_REAL_RUNTIME_DATASETS))

        for case in real_cases:
            self.assertTrue(case.requires_ray)
            dataset = case.name.removeprefix("real_iid_")
            model_index = case.command.index("--model") + 1
            model = case.command[model_index]
            self.assertNotEqual(model, "auto")
            self.assertIn(model, DATASET_MODEL_POLICIES[dataset].allowed_models)

    def test_full_mode_adds_all_dataset_model_dry_runs_and_full_runtime_dataset_coverage(self) -> None:
        output_dir = self._workspace_case_dir("full_mode_case_coverage")
        smoke_cases = build_validation_cases(mode="smoke", num_rounds=1, output_dir=output_dir)
        full_cases = build_validation_cases(mode="full", num_rounds=1, output_dir=output_dir)

        expected_model_cases = sum(len(policy.allowed_models) for policy in DATASET_MODEL_POLICIES.values())
        actual_model_cases = sum(1 for case in full_cases if case.name.startswith("dry_model_"))
        self.assertEqual(actual_model_cases, expected_model_cases)

        full_real_dataset_runs = sum(1 for case in full_cases if case.name.startswith("real_iid_"))
        self.assertEqual(full_real_dataset_runs, len(REGISTRY.available()))
        self.assertGreater(len(full_cases), len(smoke_cases))

    def test_run_validation_suite_skips_ray_cases_when_ray_is_missing(self) -> None:
        workspace_dir = self._workspace_case_dir("ray_skip_case")
        output_dir = workspace_dir / "validation"
        cases = [
            ValidationCase(
                name="needs_ray",
                description="Should be skipped when ray is missing",
                command=["this-command-should-not-run"],
                cwd=workspace_dir,
                requires_ray=True,
            )
        ]

        stdout = StringIO()
        with patch("flower_research_extension.experiments.validation_suite._ray_available", return_value=False):
            with patch("sys.stdout", stdout):
                exit_code = run_validation_suite(
                    cases=cases,
                    output_dir=output_dir,
                    stop_on_failure=False,
                )

        self.assertEqual(exit_code, 0)
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["skipped"], 1)
        self.assertIn("[SKIP] needs_ray", stdout.getvalue())
        self.assertIn("\x1b[33m", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
