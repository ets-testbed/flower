from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    repo_root_str = str(_REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from flower_research_extension.data_files import REGISTRY
from flower_research_extension.experiments.catalog import DATASET_MODEL_POLICIES, DISTRIBUTIONS
from flower_research_extension.experiments.run_commands import SCENARIOS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
SMOKE_REAL_RUNTIME_DATASETS = ("mnist", "cifar10", "cifar100")


@dataclass(frozen=True)
class ValidationCase:
    name: str
    description: str
    command: list[str]
    cwd: Path
    expects_artifacts: bool = False
    requires_ray: bool = False
    timeout_sec: int = 900


@dataclass
class ValidationResult:
    name: str
    description: str
    status: str
    returncode: int
    duration_sec: float
    log_path: str
    command: list[str]
    cwd: str
    artifact_dir: str = ""
    details: str = ""


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


RESET = "\x1b[0m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"


def _colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _validate_rounds(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--num_rounds must be an integer") from exc
    if parsed < 1 or parsed > 3:
        raise argparse.ArgumentTypeError("--num_rounds must be between 1 and 3")
    return parsed


def _matrix_path(output_dir: Path, *, dataset: str, num_classes: int, num_partitions: int) -> Path:
    path = output_dir / f"distribution_matrix_{dataset}_{num_partitions}x{num_classes}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        matrix = [[1.0] * num_classes for _ in range(num_partitions)]
        path.write_text(json.dumps(matrix), encoding="utf-8")
    return path


def _distribution_args(
    *,
    distribution: str,
    dataset: str,
    num_classes: int,
    num_partitions: int,
    output_dir: Path,
) -> list[str]:
    if distribution == "iid":
        return []
    if distribution == "dirichlet":
        return ["--dirichlet_alpha", "0.3"]
    if distribution in {"label_skew", "pathological"}:
        return ["--label_skew_classes", "2"]
    if distribution == "shard":
        return ["--shard_num_shards_per_partition", "2"]
    if distribution == "inner_dirichlet":
        weights = ",".join("1" for _ in range(num_partitions))
        return ["--inner_dirichlet_alpha", "0.5", "--size_partition_weights", weights]
    if distribution == "size":
        weights = ",".join(str(i + 1) for i in range(num_partitions))
        return ["--size_partition_weights", weights]
    if distribution == "linear":
        return []
    if distribution == "square":
        return []
    if distribution == "exponential":
        return []
    if distribution == "distribution":
        matrix = _matrix_path(
            output_dir,
            dataset=dataset,
            num_classes=num_classes,
            num_partitions=num_partitions,
        )
        return ["--distribution_matrix_json", str(matrix)]
    raise ValueError(f"Unsupported distribution: {distribution}")


def _base_run_args(*, num_rounds: int, num_partitions: int, output_dir: Path) -> list[str]:
    return [
        "--dataset_root",
        str(PACKAGE_ROOT / "data"),
        "--num_rounds",
        str(num_rounds),
        "--num_partitions",
        str(num_partitions),
        "--fraction_fit",
        "1",
        "--min_fit_clients",
        str(min(2, num_partitions)),
        "--min_evaluate_clients",
        str(min(2, num_partitions)),
        "--batch_size",
        "16",
        "--local_epochs",
        "1",
        "--lr",
        "0.01",
        "--momentum",
        "0.9",
        "--seed",
        "42",
        "--client_cpu",
        "1",
        "--client_gpu",
        "0",
        "--csv_log_dir",
        str(PACKAGE_ROOT / "results" / "logs"),
        "--wandb_dir",
        str(PACKAGE_ROOT / "results" / "wandb"),
        "--wandb_project",
        "flower-federated",
        "--wandb_run_name",
        "auto",
        "--disable_wandb",
    ]


def _run_experiment_module_args(*, from_package_dir: bool) -> list[str]:
    if from_package_dir:
        return ["-m", "experiments.run_experiment"]
    return ["-m", "flower_research_extension.experiments.run_experiment"]


def _run_commands_module_args(*, from_package_dir: bool) -> list[str]:
    if from_package_dir:
        return ["-m", "experiments.run_commands"]
    return ["-m", "flower_research_extension.experiments.run_commands"]


def _real_runtime_model(dataset: str) -> str:
    policy = DATASET_MODEL_POLICIES[dataset]
    for candidate in ("net", "squeezenet1_1", "shufflenet_v2_x1_0", "mobilenet_v2"):
        if candidate in policy.allowed_models:
            return candidate
    return policy.default_model


def _ray_available() -> bool:
    return importlib.util.find_spec("ray") is not None


def _real_runtime_datasets(mode: str) -> tuple[str, ...]:
    if mode == "full":
        return tuple(REGISTRY.available())
    return SMOKE_REAL_RUNTIME_DATASETS


def build_validation_cases(
    *,
    mode: str,
    num_rounds: int,
    output_dir: Path,
) -> list[ValidationCase]:
    dry_num_partitions = 4
    real_num_partitions = 1
    cases: list[ValidationCase] = []

    cases.append(
        ValidationCase(
            name="unit_tests",
            description="Regression unit tests",
            command=[
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "flower_research_extension/tests",
                "-p",
                "test_*.py",
            ],
            cwd=REPO_ROOT,
            timeout_sec=1200,
        )
    )

    cases.append(
        ValidationCase(
            name="capabilities_package_dir",
            description="Capabilities listing from package directory",
            command=[
                sys.executable,
                *_run_experiment_module_args(from_package_dir=True),
                "--list_capabilities",
            ],
            cwd=PACKAGE_ROOT,
        )
    )
    cases.append(
        ValidationCase(
            name="dry_run_package_dir",
            description="Basic dry run from package directory",
            command=[
                sys.executable,
                *_run_experiment_module_args(from_package_dir=True),
                *(_base_run_args(num_rounds=num_rounds, num_partitions=dry_num_partitions, output_dir=output_dir)),
                "--dry_run",
            ],
            cwd=PACKAGE_ROOT,
        )
    )
    cases.append(
        ValidationCase(
            name="dry_run_repo_root",
            description="Basic dry run from repo root",
            command=[
                sys.executable,
                *_run_experiment_module_args(from_package_dir=False),
                *(_base_run_args(num_rounds=num_rounds, num_partitions=dry_num_partitions, output_dir=output_dir)),
                "--dry_run",
            ],
            cwd=REPO_ROOT,
        )
    )
    cases.append(
        ValidationCase(
            name="run_commands_list",
            description="Scenario runner listing from package directory",
            command=[sys.executable, *_run_commands_module_args(from_package_dir=True), "--list"],
            cwd=PACKAGE_ROOT,
        )
    )

    for scenario in sorted(SCENARIOS):
        cases.append(
            ValidationCase(
                name=f"scenario_{scenario}_dry_run",
                description=f"Scenario dry run: {scenario}",
                command=[
                    sys.executable,
                    *_run_commands_module_args(from_package_dir=True),
                    "--scenario",
                    scenario,
                    "--dry_run",
                ],
                cwd=PACKAGE_ROOT,
            )
        )

    for dataset in REGISTRY.available():
        provider = REGISTRY.get(dataset)
        policy = DATASET_MODEL_POLICIES[dataset]

        for distribution in DISTRIBUTIONS:
            cases.append(
                ValidationCase(
                    name=f"dry_distribution_{dataset}_{distribution}",
                    description=f"Dry-run distribution coverage: {dataset} / {distribution}",
                    command=[
                        sys.executable,
                        *_run_experiment_module_args(from_package_dir=True),
                        *(_base_run_args(num_rounds=num_rounds, num_partitions=dry_num_partitions, output_dir=output_dir)),
                        "--dataset",
                        dataset,
                        "--model",
                        "auto",
                        "--distribution",
                        distribution,
                        *(
                            _distribution_args(
                                distribution=distribution,
                                dataset=dataset,
                                num_classes=int(getattr(provider, "num_classes", 10)),
                                num_partitions=dry_num_partitions,
                                output_dir=output_dir,
                            )
                        ),
                        "--dry_run",
                    ],
                    cwd=PACKAGE_ROOT,
                )
            )

        if mode == "full":
            for model in policy.allowed_models:
                cases.append(
                    ValidationCase(
                        name=f"dry_model_{dataset}_{model}",
                        description=f"Dry-run dataset/model coverage: {dataset} / {model}",
                        command=[
                            sys.executable,
                            *_run_experiment_module_args(from_package_dir=True),
                            *(_base_run_args(num_rounds=num_rounds, num_partitions=dry_num_partitions, output_dir=output_dir)),
                            "--dataset",
                            dataset,
                            "--model",
                            model,
                            "--distribution",
                            "iid",
                            "--dry_run",
                        ],
                        cwd=PACKAGE_ROOT,
                    )
                )

    if mode != "dry-only":
        for dataset in _real_runtime_datasets(mode):
            cases.append(
                ValidationCase(
                    name=f"real_iid_{dataset}",
                    description=f"Real runtime smoke: {dataset} / {_real_runtime_model(dataset)} / iid",
                    command=[
                        sys.executable,
                        *_run_experiment_module_args(from_package_dir=True),
                        *(_base_run_args(num_rounds=num_rounds, num_partitions=real_num_partitions, output_dir=output_dir)),
                        "--dataset",
                        dataset,
                        "--model",
                        _real_runtime_model(dataset),
                        "--distribution",
                        "iid",
                    ],
                    cwd=PACKAGE_ROOT,
                    expects_artifacts=True,
                    requires_ray=True,
                    timeout_sec=900,
                )
            )

        for distribution in ("dirichlet", "shard", "size", "distribution"):
            cases.append(
                ValidationCase(
                    name=f"real_partition_mnist_{distribution}",
                    description=f"Real partition smoke: mnist / net / {distribution}",
                    command=[
                        sys.executable,
                        *_run_experiment_module_args(from_package_dir=True),
                        *(_base_run_args(num_rounds=num_rounds, num_partitions=real_num_partitions, output_dir=output_dir)),
                        "--dataset",
                        "mnist",
                        "--model",
                        "net",
                        "--distribution",
                        distribution,
                        *(
                            _distribution_args(
                                distribution=distribution,
                                dataset="mnist",
                                num_classes=int(getattr(REGISTRY.get("mnist"), "num_classes", 10)),
                                num_partitions=real_num_partitions,
                                output_dir=output_dir,
                            )
                        ),
                    ],
                    cwd=PACKAGE_ROOT,
                    expects_artifacts=True,
                    requires_ray=True,
                    timeout_sec=900,
                )
            )

    return cases


def _latest_run_dirs() -> set[Path]:
    logs_dir = PACKAGE_ROOT / "results" / "logs"
    if not logs_dir.exists():
        return set()
    return {path for path in logs_dir.iterdir() if path.is_dir()}


def _artifact_check(before: set[Path], after: set[Path]) -> tuple[bool, str, str]:
    new_dirs = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if not new_dirs:
        return False, "", "No new run folder created under results/logs"

    latest = new_dirs[-1]
    required = {
        "run_config.json",
        "round_metrics.jsonl",
        "run_summary.json",
    }
    existing = {path.name for path in latest.iterdir()}
    missing = sorted(required - existing)
    if missing:
        return False, str(latest), f"Missing expected artifacts: {', '.join(missing)}"
    return True, str(latest), ""


def _write_log(log_path: Path, *, case: ValidationCase, completed: subprocess.CompletedProcess[str], elapsed: float) -> None:
    text = [
        f"CASE: {case.name}",
        f"DESCRIPTION: {case.description}",
        f"CWD: {case.cwd}",
        f"COMMAND: {' '.join(case.command)}",
        f"RETURN_CODE: {completed.returncode}",
        f"DURATION_SEC: {elapsed:.2f}",
        "",
        "STDOUT:",
        completed.stdout or "",
        "",
        "STDERR:",
        completed.stderr or "",
    ]
    log_path.write_text("\n".join(text), encoding="utf-8")


def _extract_failure_details(completed: subprocess.CompletedProcess[str]) -> str:
    stderr_lines = [line.strip() for line in (completed.stderr or "").splitlines() if line.strip()]
    stdout_lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if stderr_lines:
        return stderr_lines[-1]
    if stdout_lines:
        return stdout_lines[-1]
    return "No output captured"


def run_validation_suite(
    *,
    cases: Iterable[ValidationCase],
    output_dir: Path,
    stop_on_failure: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "case_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cases = list(cases)
    results: list[ValidationResult] = []

    print(f"Validation output: {output_dir}")
    print(f"Cases planned: {len(cases)}")
    ray_available = _ray_available()
    if not ray_available and any(case.requires_ray for case in cases):
        print(_colorize("Ray not found in the active interpreter. Real runtime cases will be marked as SKIP.", YELLOW), flush=True)

    for index, case in enumerate(cases, start=1):
        print(_colorize(f"[{index:03d}/{len(cases):03d}] RUN {case.name} :: {case.description}", CYAN), flush=True)
        if case.requires_ray and not ray_available:
            result = ValidationResult(
                name=case.name,
                description=case.description,
                status="SKIP",
                returncode=0,
                duration_sec=0.0,
                log_path="",
                command=case.command,
                cwd=str(case.cwd),
                details="Skipped because 'ray' is not installed in the active interpreter.",
            )
            results.append(result)
            print(_colorize(f"[SKIP] {case.name} (0.0s)", YELLOW), flush=True)
            print(f"      {result.details}", flush=True)
            continue

        before_dirs = _latest_run_dirs() if case.expects_artifacts else set()
        started = datetime.now().timestamp()
        try:
            completed = subprocess.run(
                case.command,
                cwd=str(case.cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=case.timeout_sec,
                check=False,
            )
            elapsed = datetime.now().timestamp() - started
            log_path = logs_dir / f"{index:03d}_{_safe_name(case.name)}.log"
            _write_log(log_path, case=case, completed=completed, elapsed=elapsed)

            artifact_dir = ""
            details = ""
            status = "PASS"

            if completed.returncode != 0:
                status = "FAIL"
                details = _extract_failure_details(completed)
            elif case.expects_artifacts:
                ok, artifact_dir, artifact_details = _artifact_check(before_dirs, _latest_run_dirs())
                if not ok:
                    status = "FAIL"
                    details = artifact_details

            result = ValidationResult(
                name=case.name,
                description=case.description,
                status=status,
                returncode=int(completed.returncode),
                duration_sec=elapsed,
                log_path=str(log_path),
                command=case.command,
                cwd=str(case.cwd),
                artifact_dir=artifact_dir,
                details=details,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = datetime.now().timestamp() - started
            log_path = logs_dir / f"{index:03d}_{_safe_name(case.name)}.log"
            log_path.write_text(
                "\n".join(
                    [
                        f"CASE: {case.name}",
                        f"DESCRIPTION: {case.description}",
                        f"CWD: {case.cwd}",
                        f"COMMAND: {' '.join(case.command)}",
                        f"TIMEOUT_SEC: {case.timeout_sec}",
                        "",
                        "STDOUT:",
                        exc.stdout or "",
                        "",
                        "STDERR:",
                        exc.stderr or "",
                    ]
                ),
                encoding="utf-8",
            )
            result = ValidationResult(
                name=case.name,
                description=case.description,
                status="FAIL",
                returncode=-1,
                duration_sec=elapsed,
                log_path=str(log_path),
                command=case.command,
                cwd=str(case.cwd),
                details=f"Timed out after {case.timeout_sec}s",
            )

        results.append(result)
        status_color = GREEN if result.status == "PASS" else RED
        print(_colorize(f"[{result.status}] {case.name} ({result.duration_sec:.1f}s)", status_color), flush=True)
        if result.details:
            print(f"      {result.details}", flush=True)
        if result.artifact_dir:
            print(f"      artifacts: {result.artifact_dir}", flush=True)

        if stop_on_failure and result.status == "FAIL":
            break

    passed = sum(1 for result in results if result.status == "PASS")
    skipped = sum(1 for result in results if result.status == "SKIP")
    failed = sum(1 for result in results if result.status == "FAIL")
    summary = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "total_cases": len(results),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "results": [asdict(result) for result in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"Validation output: {output_dir}",
        f"Total cases: {len(results)}",
        f"Passed: {passed}",
        f"Failed: {failed}",
        f"Skipped: {skipped}",
        "",
    ]
    for result in results:
        line = f"{result.status} | {result.name} | {result.description} | {result.duration_sec:.1f}s"
        if result.details:
            line += f" | {result.details}"
        if result.artifact_dir:
            line += f" | artifacts={result.artifact_dir}"
        lines.append(line)
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print(f"Finished. Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    print(f"Summary text: {output_dir / 'summary.txt'}")
    print(f"Summary json: {output_dir / 'summary.json'}")

    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simple pass/fail smoke suite for recent experiment updates.")
    parser.add_argument(
        "--mode",
        choices=("smoke", "full", "dry-only"),
        default="smoke",
        help="smoke: full dry-run coverage + representative real runs, full: add all dataset/model dry-runs and full real runtime dataset coverage, dry-only: skip real runs.",
    )
    parser.add_argument(
        "--num_rounds",
        type=_validate_rounds,
        default=1,
        help="Rounds for real and dry-run smoke commands (must be between 1 and 3).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory for suite summaries/logs. Defaults to results/validation/<timestamp>.",
    )
    parser.add_argument(
        "--stop_on_failure",
        action="store_true",
        help="Stop immediately after the first failure.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (PACKAGE_ROOT / "results" / "validation" / f"validation_{_timestamp()}").resolve()
    )

    cases = build_validation_cases(
        mode=str(args.mode),
        num_rounds=int(args.num_rounds),
        output_dir=output_dir,
    )
    return run_validation_suite(
        cases=cases,
        output_dir=output_dir,
        stop_on_failure=bool(args.stop_on_failure),
    )


if __name__ == "__main__":
    raise SystemExit(main())

