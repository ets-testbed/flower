from typing import List, Tuple, Optional, Dict
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitRes

from flower_research_extension.plugins.base import MetricsPlugin


class HookedStrategy(Strategy):
    def __init__(self, base_strategy: Strategy, plugins: Optional[List[MetricsPlugin]] = None):
        self.base_strategy = base_strategy
        self.plugins = plugins or []

    # Mandatory: delegate abstract methods
    def initialize_parameters(self, client_manager):
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.base_strategy.aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[Parameters], Dict]:
        aggregated_params, aggregated_metrics = self.base_strategy.aggregate_fit(server_round, results, failures)

        for proxy, fit_res in results:
            for plugin in self.plugins:
                plugin.on_client_result(server_round, proxy.cid, fit_res.metrics)

        for proxy, exc in failures:
            for plugin in self.plugins:
                plugin.on_client_failure(server_round, proxy.cid, exc)

        for plugin in self.plugins:
            plugin.on_round_end(server_round, aggregated_metrics)

        return aggregated_params, aggregated_metrics

    def evaluate(self, server_round: int, parameters: Parameters):
        result = self.base_strategy.evaluate(server_round, parameters)
        if result is None:
            return None

        _, metrics = result
        for plugin in self.plugins:
            plugin.on_server_evaluate(server_round, metrics)

        return result
