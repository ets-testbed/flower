import wandb
from pathlib import Path
from typing import Dict

from flower_research_extension.plugins.base import MetricsPlugin


class WandBLogger(MetricsPlugin):
    """
    Logs training progress to Weights & Biases (W&B).
    Includes:
      - Global metrics per round
      - Per-client metrics
      - Server-side evaluation results
    """

    def __init__(self, exp_dir: str = "results/wandb", project: str = "flower-research", run_name: str = None):
        self.latest_client_metrics = {}

        # Create output directory
        Path(exp_dir).mkdir(parents=True, exist_ok=True)

        # Init W&B run
        wandb.init(
            project=project,
            name=run_name,
            dir=exp_dir,
            job_type="server",
        )

    def on_client_result(self, round_num: int, client_id: str, metrics: Dict):
        self.latest_client_metrics[client_id] = {"round": round_num, "client_id": client_id, **metrics}

    def on_round_end(self, round_num: int, aggregated_metrics: Dict):
        # Log global metrics
        wandb.log({
            "round": round_num,
            **aggregated_metrics
        })

        # Log client metrics as a table
        rows = list(self.latest_client_metrics.values())
        if rows:
            columns = sorted(rows[0].keys())
            data = [[row[col] for col in columns] for row in rows]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"client_metrics_table": table})

        self.latest_client_metrics.clear()

    def on_server_evaluate(self, round_num: int, metrics: Dict):
        wandb.log({"round": round_num, **metrics})

    def on_client_failure(self, round_num: int, client_id: str, error: Exception):
        wandb.log({
            "round": round_num,
            f"{client_id}/failure": str(error)
        })

    def finalize(self):
        wandb.finish()
