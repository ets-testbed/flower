import csv
from datetime import datetime
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

        self.global_path = self.log_folder / f"global_metrics_{timestamp}.csv"
        self.client_path = self.log_folder / f"client_metrics_{timestamp}.csv"

        # Open both files for writing
        self.global_file = open(self.global_path, "w", newline="")
        self.client_file = open(self.client_path, "w", newline="")

        # Initialize CSV writers
        self.global_writer = csv.DictWriter(
            self.global_file, fieldnames=["round", "loss", "accuracy"]
        )
        self.global_writer.writeheader()

        self.client_writer = csv.DictWriter(
            self.client_file, fieldnames=["round", "client_id", "loss", "accuracy"]
        )
        self.client_writer.writeheader()

    def on_round_end(self, round_num: int, aggregated_metrics: Dict):
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

    def finalize(self):
        self.global_file.close()
        self.client_file.close()
