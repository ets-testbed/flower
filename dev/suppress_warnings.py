import logging
import os
import warnings

# ─── DeprecationWarnings from known modules ────────────────────────────────
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*co_lnotab is deprecated.*",
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"wandb\.analytics\.sentry"
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"google\.protobuf\.internal\.well_known_types"
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"datasets\.utils\._dill"
)


# ─── (Optional) TensorFlow CPU/GPU logs ────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# 1) Silence all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 2) Silence that “No fit_metrics_aggregation_fn provided” warning from FedAvg
#    which is logged at WARNING level in flwr.server.strategy
# logging.getLogger("flwr.server.strategy").setLevel(logging.ERROR)

# 3) Stop Ray’s “repeated across cluster” logs
os.environ["RAY_DEDUP_LOGS"] = "0"

# 4) (Optional) Reduce TF GPU chatter if you load TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


