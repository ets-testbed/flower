import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from flower_research_extension.plugins.base import MetricsPlugin


class CSVLogger(MetricsPlugin):
    """
    Persist run configuration plus round/client metrics in local files.

    Each run directory contains:
    - round_metrics_<timestamp>.csv
    - client_metrics_<timestamp>.csv
    - round_metrics.jsonl
    - run_config.json
    - run_summary.json
    """

    def __init__(self, log_dir: str = "results/logs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_folder = Path(log_dir) / f"run_{timestamp}"
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.started_at_utc = datetime.now(timezone.utc).isoformat()

        self.global_path = self.log_folder / f"round_metrics_{timestamp}.csv"
        self.client_path = self.log_folder / f"client_metrics_{timestamp}.csv"
        self.round_metrics_path = self.log_folder / "round_metrics.jsonl"
        self.run_config_path = self.log_folder / "run_config.json"
        self.summary_path = self.log_folder / "run_summary.json"

        self.round_metrics_file = self.round_metrics_path.open("w", encoding="utf-8")

        self.round_rows: list[Dict[str, Any]] = []
        self.client_rows: list[Dict[str, Any]] = []
        self.round_index: dict[int, int] = {}
        self.round_fields = ["round"]
        self.client_fields = ["round", "client_id"]

        self.run_config: Dict[str, Any] = {}
        self.last_fit_metrics: Dict[str, Any] = {}
        self.last_eval_metrics: Dict[str, Any] = {}
        self.fit_rounds = 0
        self.eval_rounds = 0
        self.client_result_count = 0
        self.client_failure_count = 0

    def _sanitize_key(self, key: str) -> str:
        return key.replace("/", "_").replace(" ", "_")

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return json.dumps(value, sort_keys=True)

    def _prefixed_metrics(self, prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            f"{prefix}{self._sanitize_key(str(key))}": self._normalize_value(value)
            for key, value in metrics.items()
        }

    def _rewrite_csv(self, path: Path, fieldnames: list[str], rows: list[Dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})

    def _ensure_fields(self, known_fields: list[str], row: Dict[str, Any]) -> list[str]:
        for key in row:
            if key not in known_fields:
                known_fields.append(key)
        return known_fields

    def _upsert_round_row(self, round_num: int, values: Dict[str, Any]) -> None:
        row = {"round": round_num, **values}
        existing_index = self.round_index.get(round_num)
        if existing_index is None:
            self.round_rows.append(row)
            self.round_index[round_num] = len(self.round_rows) - 1
        else:
            self.round_rows[existing_index].update(values)
        self.round_fields = self._ensure_fields(self.round_fields, row)
        self._rewrite_csv(self.global_path, self.round_fields, self.round_rows)

    def _to_json_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_json_value(v) for v in value]
        if callable(value):
            return getattr(value, "__name__", str(value))
        return str(value)

    def _append_round_record(self, *, phase: str, round_num: int, metrics: Dict[str, Any]) -> None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "round": int(round_num),
            "metrics": self._to_json_value(metrics),
        }
        self.round_metrics_file.write(json.dumps(payload, sort_keys=True) + "\n")
        self.round_metrics_file.flush()

    def on_training_start(self, config: Dict = None):
        self.run_config = self._to_json_value(config or {})
        self.run_config_path.write_text(
            json.dumps(self.run_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def on_round_end(self, round_num: int, aggregated_metrics: Dict):
        aggregated_metrics = aggregated_metrics or {}
        self.last_fit_metrics = self._to_json_value(aggregated_metrics)
        self.fit_rounds += 1
        self._append_round_record(phase="fit", round_num=round_num, metrics=aggregated_metrics)
        if aggregated_metrics:
            self._upsert_round_row(round_num, self._prefixed_metrics("fit_", aggregated_metrics))

    def on_server_evaluate(self, round_num: int, metrics: Dict):
        metrics = metrics or {}
        self.last_eval_metrics = self._to_json_value(metrics)
        self.eval_rounds += 1
        self._append_round_record(phase="eval", round_num=round_num, metrics=metrics)
        if metrics:
            self._upsert_round_row(round_num, self._prefixed_metrics("eval_", metrics))

    def on_client_result(self, round_num: int, client_id: str, metrics: Dict):
        self.client_result_count += 1
        row = {"round": round_num, "client_id": client_id}
        row.update(self._prefixed_metrics("", metrics or {}))
        self.client_rows.append(row)
        self.client_fields = self._ensure_fields(self.client_fields, row)
        self._rewrite_csv(self.client_path, self.client_fields, self.client_rows)

    def on_client_failure(self, round_num: int, client_id: str, error: Exception):
        self.client_failure_count += 1
        self._append_round_record(
            phase="client_failure",
            round_num=round_num,
            metrics={"client_id": str(client_id), "error": str(error)},
        )

        row = {
            "round": round_num,
            "client_id": client_id,
            "failure": str(error),
        }
        self.client_rows.append(row)
        self.client_fields = self._ensure_fields(self.client_fields, row)
        self._rewrite_csv(self.client_path, self.client_fields, self.client_rows)

    def finalize(self):
        finished_at_utc = datetime.now(timezone.utc).isoformat()
        self._rewrite_csv(self.global_path, self.round_fields, self.round_rows)
        self._rewrite_csv(self.client_path, self.client_fields, self.client_rows)

        summary = {
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": finished_at_utc,
            "fit_rounds": self.fit_rounds,
            "eval_rounds": self.eval_rounds,
            "client_result_count": self.client_result_count,
            "client_failure_count": self.client_failure_count,
            "last_fit_metrics": self.last_fit_metrics,
            "last_eval_metrics": self.last_eval_metrics,
            "artifacts": {
                "round_metrics_csv": str(self.global_path),
                "client_metrics_csv": str(self.client_path),
                "round_metrics_jsonl": str(self.round_metrics_path),
                "run_config_json": str(self.run_config_path),
                "run_summary_json": str(self.summary_path),
            },
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        self.round_metrics_file.close()


CsvLogger = CSVLogger
