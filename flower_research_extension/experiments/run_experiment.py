"""Minimal entrypoint for running experiments.

Usage:
  - `python -m flower_research_extension.experiments.run_experiment --help`
  - `python -m flower_research_extension.experiments.run_experiment --list_capabilities`
  - `python -m flower_research_extension.experiments.run_experiment --dry_run`
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    """Allow running as `-m experiments.run_experiment` from package directory."""
    repo_root_str = str(_REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from flower_research_extension.experiments.run_experiment_cli import (
    DATASET_MODEL_POLICIES,
    DISTRIBUTIONS,
    MODEL_BUILDERS,
    MODEL_FIT_PROFILE,
    _capabilities,
    _load_distribution_matrix,
    _normalize_args,
    _parse_size_partition_weights,
    _resolve_model_name,
    _to_serializable_config,
    _validate_args,
    build_parser,
    get_model_builder,
)

__all__ = [
    "DATASET_MODEL_POLICIES",
    "DISTRIBUTIONS",
    "MODEL_BUILDERS",
    "MODEL_FIT_PROFILE",
    "_capabilities",
    "_load_distribution_matrix",
    "_normalize_args",
    "_parse_size_partition_weights",
    "_resolve_model_name",
    "_to_serializable_config",
    "_validate_args",
    "build_parser",
    "get_model_builder",
    "main",
]


def _prepare_simulation_runtime(args) -> None:
    """Make path args stable if we need to switch cwd for Ray worker imports."""
    launch_cwd = Path.cwd().resolve()
    for field in ("dataset_root", "csv_log_dir", "wandb_dir", "distribution_matrix_json"):
        value = getattr(args, field, "")
        if not isinstance(value, str) or value == "":
            continue
        path = Path(value)
        if not path.is_absolute():
            setattr(args, field, str((launch_cwd / path).resolve()))

    # Ray workers can fail to import the package if launched from inside the package
    # directory. Run simulation from repo root to keep worker imports stable.
    if launch_cwd != _REPO_ROOT:
        os.chdir(_REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_capabilities:
        print(json.dumps(_capabilities(), indent=2, sort_keys=True))
        return 0

    _validate_args(parser, args)
    args = _normalize_args(args)

    if args.dry_run:
        print(json.dumps(_to_serializable_config(args), indent=2, sort_keys=True))
        return 0

    _prepare_simulation_runtime(args)

    from flower_research_extension.experiments.experiment_setup import run_experiment

    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
