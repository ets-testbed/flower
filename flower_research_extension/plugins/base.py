from abc import ABC, abstractmethod
from typing import Dict


class MetricsPlugin(ABC):
    """
    Abstract base class for metrics/logging plugins.
    Defines hooks that strategies can call during training.
    """

    def on_training_start(self, config: Dict = None):
        """Called once before training starts (optional)."""
        pass

    def on_client_result(self, round_num: int, client_id: str, metrics: Dict):
        """Called for each client's result after `fit` or `evaluate`."""
        pass

    def on_round_end(self, round_num: int, aggregated_metrics: Dict):
        """Called after aggregation of client results."""
        pass

    def on_server_evaluate(self, round_num: int, metrics: Dict):
        """Called after server-side evaluation."""
        pass

    def finalize(self):
        """Called once at the end of training to flush, close, save..."""
        pass

    def on_client_failure(self, round_num: int, client_id: str, error: Exception):
        """Called if a client fails during fit/evaluation."""
        pass


