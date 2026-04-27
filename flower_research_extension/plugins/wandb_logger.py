import wandb
from pathlib import Path
from typing import Dict, Optional

from flower_research_extension.plugins.base import MetricsPlugin


class WandBLogger(MetricsPlugin):
    """
    Original structure, but with explicit step binding:
    - bind all series to step 'round'
    - log with step=round_num
    - single commit per round (eval commits)
    """

    def __init__(
        self,
        exp_dir: str = "results/wandb",
        project: str = "flower-research",
        run_name: Optional[str] = None,
    ):
        Path(exp_dir).mkdir(parents=True, exist_ok=True)
        try:
            wandb.init(
                project=project,
                reinit=True,
                name=run_name,
                dir=exp_dir,
                job_type="server",
                resume=False,
            )
        except TypeError:
            wandb.init(
                project=project,
                name=run_name,
                dir=exp_dir,
                job_type="server",
                resume=False,
                return_previous=False,
                finish_previous=True,
            )

        # Bind step: prevents W&B’s auto-increment from doubling steps
        wandb.define_metric("round")
        wandb.define_metric("*", step_metric="round")

    def _to_wandb_value(self, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_wandb_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_wandb_value(v) for v in value]
        if callable(value):
            return getattr(value, "__name__", str(value))
        return str(value)

    def _flatten_dict(self, payload: Dict, *, prefix: str = "") -> Dict[str, object]:
        out: Dict[str, object] = {}
        for key, value in payload.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(self._flatten_dict(value, prefix=full_key))
            else:
                out[full_key] = value
        return out

    def on_training_start(self, config: Dict = None):
        if not config:
            return
        safe = self._to_wandb_value(config)
        if not isinstance(safe, dict):
            safe = {"run_config": safe}
        flat = self._flatten_dict(safe)
        if flat:
            wandb.config.update(flat, allow_val_change=True)

    # def on_client_result(self, round_num: int, client_id: str, metrics: Dict):
    #     if metrics:
    #         metrics = {f"round/{round_num}/{k}": v for k, v in metrics.items()}
    #         wandb.log(metrics, step=round_num, commit=False)

    def on_round_end(self, round_num: int, aggregated_metrics: Dict):
        aggregated_metrics = aggregated_metrics or {}
        fit_metrics = {f"fit/{k}": v for k, v in aggregated_metrics.items()}
        if fit_metrics:
            # Do not commit yet; eval will commit to produce 1 commit/round
            wandb.log({"round": round_num, **fit_metrics}, step=round_num, commit=False)

    def on_server_evaluate(self, round_num: int, metrics: Dict):
        if metrics:
            # Commit at eval so each round ends with a single commit
            wandb.log({"round": round_num, **{f"eval/{k}": v for k, v in metrics.items()}}, step=round_num, commit=True)

    def on_client_failure(self, round_num: int, client_id: str, error: Exception):
        wandb.log({"round": round_num, f"{client_id}/failure": str(error)}, step=round_num, commit=False)

    def finalize(self):
        wandb.finish()
