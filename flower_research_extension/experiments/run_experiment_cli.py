from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import torch.nn as nn

from flower_research_extension.data_files import REGISTRY as _REGISTRY
from flower_research_extension.experiments.catalog import (
    DATASET_MODEL_POLICIES,
    DISTRIBUTIONS,
    MODEL_BUILDERS,
    MODEL_FIT_PROFILE,
    build_capabilities,
    resolve_model_name,
)


def _sanitize(name: str) -> str:
    name = name.strip()
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", name)


def _auto_run_name(
    dataset: str,
    model: str,
    distribution: str,
    num_rounds: int,
    num_partitions: int,
    batch_size: int,
    fraction_fit: float,
    seed: int,
) -> str:
    ff = f"{fraction_fit:.2f}".rstrip("0").rstrip(".")
    suffix = f"m{model}_d{distribution}_r{num_rounds}_C{num_partitions}_b{batch_size}_ff{ff}_s{seed}"
    return _sanitize(f"{dataset}_fedavg_{suffix}")


def get_model_builder(model_name: str) -> Callable[[int], nn.Module]:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_BUILDERS[model_name]


def _resolve_model_name(dataset: str, requested_model: str) -> str:
    return resolve_model_name(dataset, requested_model)


def _capabilities() -> dict:
    return build_capabilities(datasets=_REGISTRY.available())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Flower federated learning experiment with configurable parameters."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=_REGISTRY.available(),
        help="Dataset to use",
    )
    parser.add_argument("--dataset_root", type=str, default="data", help="Dataset root directory")
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=sorted(["auto", *MODEL_BUILDERS.keys()]),
        help="Model architecture or 'auto' to use dataset-specific default",
    )
    parser.add_argument(
        "--list_capabilities",
        action="store_true",
        help="Print datasets/distributions/models, descriptions, and dataset-model policies as JSON, then exit.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per client")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for partitioning")
    parser.add_argument(
        "--distribution",
        type=str,
        default="iid",
        choices=DISTRIBUTIONS,
        help="Client data distribution mode (use --list_capabilities for mode descriptions)",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration (used when --distribution dirichlet)",
    )
    parser.add_argument(
        "--label_skew_classes",
        type=int,
        default=2,
        help="Classes per client (used when --distribution label_skew/pathological)",
    )
    parser.add_argument(
        "--shard_num_shards_per_partition",
        type=int,
        default=2,
        help="Number of shards per partition (used when --distribution shard)",
    )
    parser.add_argument(
        "--inner_dirichlet_alpha",
        type=float,
        default=0.5,
        help="Inner Dirichlet concentration (used when --distribution inner_dirichlet)",
    )
    parser.add_argument(
        "--size_partition_weights",
        type=str,
        default="",
        help="Comma-separated partition weights (used when --distribution size/inner_dirichlet)",
    )
    parser.add_argument(
        "--distribution_matrix_json",
        type=str,
        default="",
        help="Path to JSON matrix [num_partitions][num_classes] (used when --distribution distribution)",
    )

    parser.add_argument("--num_rounds", type=int, default=10, help="Total number of federated rounds")
    parser.add_argument("--num_partitions", type=int, default=10, help="Number of simulated clients")
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=0.25,
        help="Fraction of clients used for training each round",
    )
    parser.add_argument(
        "--min_fit_clients",
        type=int,
        default=3,
        help="Minimum number of clients to sample for training",
    )
    parser.add_argument(
        "--min_evaluate_clients",
        type=int,
        default=3,
        help="Minimum number of clients to sample for evaluation",
    )
    parser.add_argument("--client_cpu", type=int, default=1, help="CPUs per client for simulation backend")
    parser.add_argument("--client_gpu", type=float, default=0.01, help="GPU fraction per client for simulation backend")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local training epochs on each selected client")
    parser.add_argument("--lr", type=float, default=0.01, help="Client optimizer learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Client optimizer momentum")
    parser.add_argument("--csv_log_dir", type=str, default="results/logs", help="Directory for CSV logs")
    parser.add_argument("--wandb_dir", type=str, default="results/wandb", help="Directory for Weights & Biases logs")
    parser.add_argument("--wandb_project", type=str, default="flower-federated", help="Weights & Biases project name")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="auto",
        help="Weights & Biases run name. Use 'auto' to name by dataset/rounds/clients/batch/fraction_fit.",
    )
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved config and exit without running")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_rounds < 1:
        parser.error("--num_rounds must be >= 1")
    if args.num_partitions < 1:
        parser.error("--num_partitions must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.local_epochs < 1:
        parser.error("--local_epochs must be >= 1")
    if not (0.0 < args.fraction_fit <= 1.0):
        parser.error("--fraction_fit must be in the range (0, 1]")
    if args.min_fit_clients < 1:
        parser.error("--min_fit_clients must be >= 1")
    if args.min_evaluate_clients < 1:
        parser.error("--min_evaluate_clients must be >= 1")
    if args.min_fit_clients > args.num_partitions:
        parser.error("--min_fit_clients cannot exceed --num_partitions")
    if args.min_evaluate_clients > args.num_partitions:
        parser.error("--min_evaluate_clients cannot exceed --num_partitions")
    if args.client_cpu < 1:
        parser.error("--client_cpu must be >= 1")
    if args.client_gpu < 0:
        parser.error("--client_gpu must be >= 0")
    if args.lr <= 0:
        parser.error("--lr must be > 0")
    if not (0.0 <= args.momentum <= 1.0):
        parser.error("--momentum must be in the range [0, 1]")

    if args.distribution == "dirichlet" and args.dirichlet_alpha <= 0:
        parser.error("--dirichlet_alpha must be > 0 when --distribution=dirichlet")
    if args.distribution in {"label_skew", "pathological"} and args.label_skew_classes < 1:
        parser.error("--label_skew_classes must be >= 1 when --distribution=label_skew/pathological")
    if args.distribution == "shard" and args.shard_num_shards_per_partition < 1:
        parser.error("--shard_num_shards_per_partition must be >= 1 when --distribution=shard")
    if args.distribution == "inner_dirichlet" and args.inner_dirichlet_alpha <= 0:
        parser.error("--inner_dirichlet_alpha must be > 0 when --distribution=inner_dirichlet")

    if args.size_partition_weights:
        weights = _parse_size_partition_weights(args.size_partition_weights)
        if len(weights) != args.num_partitions:
            parser.error("--size_partition_weights length must equal --num_partitions")
        if any(w < 0 for w in weights):
            parser.error("--size_partition_weights values must be non-negative")

    if args.distribution == "distribution":
        path = Path(args.distribution_matrix_json)
        if not args.distribution_matrix_json or not path.exists():
            parser.error("--distribution_matrix_json must point to an existing JSON file when --distribution=distribution")
        try:
            matrix = _load_distribution_matrix(path)
        except Exception as exc:
            parser.error(f"invalid --distribution_matrix_json: {exc}")
        if len(matrix) != args.num_partitions:
            parser.error("distribution matrix row count must equal --num_partitions")
        provider = _REGISTRY.get(args.dataset)
        required_num_classes = int(getattr(provider, "num_classes", 0))
        if matrix and len(matrix[0]) < required_num_classes:
            parser.error(
                f"distribution matrix column count ({len(matrix[0])}) must be >= dataset num_classes ({required_num_classes})"
            )

    try:
        _ = _resolve_model_name(args.dataset, args.model)
    except Exception as exc:
        parser.error(str(exc))


def _parse_size_partition_weights(text: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in text.split(",") if v.strip() != "")


def _load_distribution_matrix(path: Path) -> tuple[tuple[float, ...], ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("distribution matrix JSON must be a non-empty 2D list")

    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for row in data:
        if not isinstance(row, list) or not row:
            raise ValueError("distribution matrix JSON rows must be non-empty lists")
        parsed = tuple(float(v) for v in row)
        if width is None:
            width = len(parsed)
        if len(parsed) != width:
            raise ValueError("distribution matrix JSON rows must have equal length")
        rows.append(parsed)
    return tuple(rows)


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.requested_model = args.model
    args.model = _resolve_model_name(args.dataset, args.requested_model)

    if args.wandb_run_name in {"auto", "", None}:
        args.wandb_run_name = _auto_run_name(
            dataset=args.dataset,
            model=args.model,
            distribution=args.distribution,
            num_rounds=args.num_rounds,
            num_partitions=args.num_partitions,
            batch_size=args.batch_size,
            fraction_fit=args.fraction_fit,
            seed=args.seed,
        )

    args.model_builder = get_model_builder(args.model)
    args.model_fit_profile = MODEL_FIT_PROFILE.get(args.model, "medium")
    args.size_partition_weights = (
        _parse_size_partition_weights(args.size_partition_weights) if args.size_partition_weights else None
    )
    args.distribution_matrix = (
        _load_distribution_matrix(Path(args.distribution_matrix_json)) if args.distribution_matrix_json else None
    )
    return args


def _to_serializable_config(args: argparse.Namespace) -> dict:
    cfg = vars(args).copy()
    cfg["model_builder"] = args.model
    return cfg

