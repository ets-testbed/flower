from typing import List, Tuple, Optional, Dict

from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitRes

from flower_research_extension.plugins.base import MetricsPlugin
import time


class RoundTimerStrategy(Strategy):
    def __init__(self, base_strategy: Strategy, plugins: Optional[List[MetricsPlugin]] = None):
        self.base_strategy = base_strategy
        self.plugins = plugins or []

    # 🟢 DELEGATE REQUIRED ABSTRACT METHODS
    def initialize_parameters(self, client_manager):
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.base_strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        return self.base_strategy.evaluate(server_round, parameters)

    # ✅ CUSTOM AGGREGATION WITH TIMING
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[Parameters], Dict]:

        start_time = time.time()
        aggregated_params, aggregated_metrics = self.base_strategy.aggregate_fit(server_round, results, failures)
        duration = time.time() - start_time

        aggregated_metrics["round_time"] = duration

        for plugin in self.plugins:
            plugin.on_round_end(server_round, aggregated_metrics)

        return aggregated_params, aggregated_metrics
