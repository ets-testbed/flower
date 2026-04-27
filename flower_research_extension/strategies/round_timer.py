from typing import List, Tuple, Optional, Dict

from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitRes

import time


class RoundTimerStrategy(Strategy):
    def __init__(self, base_strategy: Strategy):
        self.base_strategy = base_strategy
        self._fit_started_at: Dict[int, float] = {}
        self._fit_finished_at: Dict[int, float] = {}

    # 🟢 DELEGATE REQUIRED ABSTRACT METHODS
    def initialize_parameters(self, client_manager):
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        self._fit_started_at[server_round] = time.perf_counter()
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.base_strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        eval_started_at = time.perf_counter()
        result = self.base_strategy.evaluate(server_round, parameters)
        eval_finished_at = time.perf_counter()

        if result is None:
            self._fit_started_at.pop(server_round, None)
            self._fit_finished_at.pop(server_round, None)
            return None

        loss, metrics = result
        metrics = dict(metrics or {})
        fit_finished_at = self._fit_finished_at.get(server_round)
        fit_started_at = self._fit_started_at.get(server_round)

        metrics["server_eval_time"] = eval_finished_at - eval_started_at
        if fit_finished_at is not None:
            metrics["fit_to_eval_gap_time"] = max(0.0, eval_started_at - fit_finished_at)
        if fit_started_at is not None:
            metrics["round_total_time"] = eval_finished_at - fit_started_at

        self._fit_started_at.pop(server_round, None)
        self._fit_finished_at.pop(server_round, None)
        return loss, metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[Parameters], Dict]:
        aggregate_started_at = time.perf_counter()
        aggregated_params, aggregated_metrics = self.base_strategy.aggregate_fit(server_round, results, failures)
        aggregate_finished_at = time.perf_counter()
        fit_started_at = self._fit_started_at.get(server_round, aggregate_started_at)
        self._fit_finished_at[server_round] = aggregate_finished_at

        aggregated_metrics = dict(aggregated_metrics or {})
        aggregated_metrics["fit_elapsed_time"] = aggregate_finished_at - fit_started_at
        aggregated_metrics["fit_aggregation_time"] = aggregate_finished_at - aggregate_started_at
        aggregated_metrics["fit_client_count"] = len(results)
        aggregated_metrics["fit_failure_count"] = len(failures)

        return aggregated_params, aggregated_metrics
