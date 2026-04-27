import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Keep this test runnable in lightweight environments where Flower or its
# optional dependencies may be absent.
try:
    from flwr.server.strategy import Strategy
except Exception:
    flwr_mod = types.ModuleType("flwr")
    flwr_server_mod = types.ModuleType("flwr.server")
    flwr_server_strategy_mod = types.ModuleType("flwr.server.strategy")
    flwr_server_client_proxy_mod = types.ModuleType("flwr.server.client_proxy")
    flwr_common_mod = types.ModuleType("flwr.common")

    class Strategy:  # type: ignore[no-redef]
        pass

    class ClientProxy:
        pass

    class Parameters:
        pass

    class FitRes:
        pass

    flwr_server_strategy_mod.Strategy = Strategy
    flwr_server_client_proxy_mod.ClientProxy = ClientProxy
    flwr_common_mod.Parameters = Parameters
    flwr_common_mod.FitRes = FitRes

    flwr_server_mod.strategy = flwr_server_strategy_mod
    flwr_server_mod.client_proxy = flwr_server_client_proxy_mod
    flwr_mod.server = flwr_server_mod
    flwr_mod.common = flwr_common_mod

    sys.modules["flwr"] = flwr_mod
    sys.modules["flwr.server"] = flwr_server_mod
    sys.modules["flwr.server.strategy"] = flwr_server_strategy_mod
    sys.modules["flwr.server.client_proxy"] = flwr_server_client_proxy_mod
    sys.modules["flwr.common"] = flwr_common_mod

from flower_research_extension.plugins.csv_logger import CSVLogger
from flower_research_extension.strategies.round_timer import RoundTimerStrategy


class FakeStrategy(Strategy):
    def initialize_parameters(self, client_manager):
        return None

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None

    def aggregate_fit(self, server_round, results, failures):
        return None, {"loss": 1.0, "accuracy": 0.5}

    def evaluate(self, server_round, parameters):
        return 0.25, {"accuracy": 0.75}


class LoggingAndTimingTests(unittest.TestCase):
    def test_round_timer_adds_stage_metrics(self) -> None:
        strategy = RoundTimerStrategy(FakeStrategy())
        strategy.configure_fit(1, None, None)
        time.sleep(0.01)
        _, fit_metrics = strategy.aggregate_fit(1, [], [])
        time.sleep(0.01)
        _, eval_metrics = strategy.evaluate(1, None)

        self.assertGreater(fit_metrics["fit_elapsed_time"], 0.0)
        self.assertIn("fit_aggregation_time", fit_metrics)
        self.assertIn("fit_client_count", fit_metrics)
        self.assertIn("fit_failure_count", fit_metrics)
        self.assertGreater(eval_metrics["round_total_time"], 0.0)
        self.assertIn("server_eval_time", eval_metrics)

    def test_csv_logger_writes_combined_round_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = CSVLogger(log_dir=tmp)
            logger.on_training_start({"dataset": "mnist", "lr": 0.1})
            logger.on_round_end(1, {"loss": 1.0, "fit_elapsed_time": 0.2})
            logger.on_server_evaluate(1, {"accuracy": 0.8, "round_total_time": 0.3})
            logger.on_client_result(1, "c1", {"loss": 0.4, "accuracy": 0.9})
            logger.on_client_failure(1, "c2", RuntimeError("boom"))
            logger.finalize()

            run_dir = next(Path(tmp).iterdir())
            round_csv = next(run_dir.glob("round_metrics_*.csv")).read_text(encoding="utf-8")
            client_csv = next(run_dir.glob("client_metrics_*.csv")).read_text(encoding="utf-8")
            config_json = next(run_dir.glob("run_config*.json")).read_text(encoding="utf-8")
            summary_json = next(run_dir.glob("run_summary*.json")).read_text(encoding="utf-8")

        self.assertIn("fit_loss", round_csv)
        self.assertIn("eval_accuracy", round_csv)
        self.assertIn("eval_round_total_time", round_csv)
        self.assertIn("failure", client_csv)
        self.assertIn("boom", client_csv)
        self.assertIn('"lr": 0.1', config_json)
        self.assertIn('"client_failure_count": 1', summary_json)


if __name__ == "__main__":
    unittest.main()
