from typing import List, Tuple, Optional
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitRes
from flwr.server.strategy import FedAvg

from flower_research_extension.plugins.base import MetricsPlugin


class CustomFedAvg(FedAvg):
    def __init__(self, *args, plugins: List[MetricsPlugin] = None, **kwargs):
        self.plugins = plugins or []
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[Parameters], Dict]:

        # Call base FedAvg aggregation
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Log individual client results
        for proxy, fit_res in results:
            for plugin in self.plugins:
                plugin.on_client_result(server_round, proxy.cid, fit_res.metrics)

        # Log aggregated metrics
        for plugin in self.plugins:
            plugin.on_round_end(server_round, aggregated_metrics)

        # Handle any failures
        for proxy, exc in failures:
            for plugin in self.plugins:
                plugin.on_client_failure(server_round, proxy.cid, exc)

        return aggregated_params, aggregated_metrics

    def evaluate(self, server_round: int, parameters: Parameters):
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None
        loss, metrics = result

        for plugin in self.plugins:
            plugin.on_server_evaluate(server_round, metrics)

        return result
