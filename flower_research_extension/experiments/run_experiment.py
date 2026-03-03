"""Minimal entrypoint for running experiments.

Usage:
  - `python -m flower_research_extension.experiments.run_experiment --help`
  - `python -m flower_research_extension.experiments.run_experiment --list_capabilities`
  - `python -m flower_research_extension.experiments.run_experiment --dry_run`
"""

from __future__ import annotations

import json

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

    from flower_research_extension.experiments.experiment_setup import run_experiment

    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
