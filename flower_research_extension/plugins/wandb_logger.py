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
        # Create output directory
        Path(exp_dir).mkdir(parents=True, exist_ok=True)

        # Init W&B run
        wandb.init(
            project=project,
            reinit=True,
            name=run_name,
            dir=exp_dir,
            job_type="server",
            resume=False
        )

    # def on_client_result(self, round_num: int, client_id: str, metrics: Dict):
    #     if metrics:
    #         metrics = {f"round/{round_num}/{k}": v for k, v in metrics.items()}
    #         wandb.log(metrics)

    def on_round_end(self, round_num: int, aggregated_metrics: Dict):
        # Only log fit/training metrics here if needed
        fit_metrics = {k: v for k, v in aggregated_metrics.items() if k.startswith("fit/")}
        if fit_metrics:
            wandb.log({"round": round_num, **fit_metrics})

    def on_server_evaluate(self, round_num: int, metrics: Dict):
        if metrics:
            wandb.log({
                "round": round_num,
                **{f"{k}": v for k, v in metrics.items()}
            })

    def on_client_failure(self, round_num: int, client_id: str, error: Exception):
        wandb.log({
            "round": round_num,
            f"{client_id}/failure": str(error)
        })

    def finalize(self):
        wandb.finish()
