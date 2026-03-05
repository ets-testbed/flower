import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict

from flower_research_extension.plugins.base import MetricsPlugin


class CSVLogger(MetricsPlugin):
    """
    Logs global and client metrics to CSV files.
    Creates timestamped logs to avoid overwriting:
      - logs/run_<timestamp>/global_metrics_<timestamp>.csv
      - logs/run_<timestamp>/client_metrics_<timestamp>.csv
    """

    def __init__(self, log_dir: str = "results/logs"):
        # Create timestamped subfolder and filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_folder = Path(log_dir) / f"run_{timestamp}"
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.started_at_utc = datetime.now(timezone.utc).isoformat()

        self.global_path = self.log_folder / f"global_metrics_{timestamp}.csv"
        self.client_path = self.log_folder / f"client_metrics_{timestamp}.csv"
        self.round_metrics_path = self.log_folder / "round_metrics.jsonl"
        self.run_config_path = self.log_folder / "run_config.json"
        self.summary_path = self.log_folder / "run_summary.json"

        # Open both files for writing
        self.global_file = open(self.global_path, "w", newline="")
        self.client_file = open(self.client_path, "w", newline="")
        self.round_metrics_file = open(self.round_metrics_path, "w", encoding="utf-8")

        # Initialize CSV writers
        self.global_writer = csv.DictWriter(
            self.global_file, fieldnames=["round", "loss", "accuracy"]
        )
        self.global_writer.writeheader()

        self.client_writer = csv.DictWriter(
            self.client_file, fieldnames=["round", "client_id", "loss", "accuracy"]
        )
        self.client_writer.writeheader()

        self.run_config: Dict = {}
        self.last_fit_metrics: Dict = {}
        self.last_eval_metrics: Dict = {}
        self.fit_rounds = 0
        self.eval_rounds = 0
        self.client_result_count = 0
        self.client_failure_count = 0

    def _to_json_value(self, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_json_value(v) for v in value]
        if callable(value):
            return getattr(value, "__name__", str(value))
        return str(value)

    def _append_round_record(self, *, phase: str, round_num: int, metrics: Dict) -> None:
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

        loss = aggregated_metrics.get("loss")
        accuracy = aggregated_metrics.get("accuracy")
        if loss is not None or accuracy is not None:
            self.global_writer.writerow({
                "round": round_num,
                "loss": loss,
                "accuracy": accuracy
            })
            self.global_file.flush()

    def on_client_result(self, round_num: int, client_id: str, metrics: Dict):
        self.client_result_count += 1
        loss = metrics.get("loss")
        accuracy = metrics.get("accuracy")
        if loss is not None or accuracy is not None:
            self.client_writer.writerow({
                "round": round_num,
                "client_id": client_id,
                "loss": loss,
                "accuracy": accuracy
            })
            self.client_file.flush()

    def on_server_evaluate(self, round_num: int, metrics: Dict):
        metrics = metrics or {}
        self.last_eval_metrics = self._to_json_value(metrics)
        self.eval_rounds += 1
        self._append_round_record(phase="eval", round_num=round_num, metrics=metrics)

    def on_client_failure(self, round_num: int, client_id: str, error: Exception):
        self.client_failure_count += 1
        self._append_round_record(
            phase="client_failure",
            round_num=round_num,
            metrics={"client_id": str(client_id), "error": str(error)},
        )

    def finalize(self):
        finished_at_utc = datetime.now(timezone.utc).isoformat()
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
                "global_metrics_csv": str(self.global_path),
                "client_metrics_csv": str(self.client_path),
                "round_metrics_jsonl": str(self.round_metrics_path),
                "run_config_json": str(self.run_config_path),
                "run_summary_json": str(self.summary_path),
            },
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        self.round_metrics_file.close()
        self.global_file.close()
        self.client_file.close()
