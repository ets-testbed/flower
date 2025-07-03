# 🌼 Flower Research Extension

This repository is a modular research-oriented extension for
the [Flower Federated Learning Framework](https://flower.dev).  
It introduces a **plugin-based metrics logging system**, **custom strategy wrappers**, and a **flexible experiment
runner** for benchmarking federated setups like CIFAR-10 with `FedAvg`.

---

## 📂 Project Structure

```
flower_research_extension/
│
├── data/                        # Dataset loaders (e.g., CIFAR-10)
│   └── cifar10.py
│
├── models/                      # Neural network models
│   └── model.py
│
├── plugins/                     # Hookable metrics plugins
│   ├── base.py                  # Abstract plugin interface
│   ├── csv_logger.py            # Logs round/client metrics to CSV
│   └── wandb_logger.py          # Logs to Weights & Biases
│
├── strategies/                  # Custom strategy wrappers
│   ├── custom_fedavg.py         # Customizable FedAvg variant
│   ├── hooked_strategy.py       # Plugin-calling wrapper
│   └── round_timer.py           # Adds timing hooks
│
├── experiments/                 # Entrypoint and utilities for experiment
│   ├── run_experiment.py        # Entrypoint for simulation run
│   ├── experiment_setup.py      # Common logic for modular setup
│   └── hyperparam_runs.sh       # Sample script for multiple runs
│
├── client.py                    # Client logic using Flower's ClientApp
├── training.py                  # Fit and evaluate functions
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Features

- ✅ Plugin interface for metrics (`MetricsPlugin`)
- 📊 CSV and Weights & Biases logging
- ⏱️ Per-round timing hooks
- 🧪 Run config via CLI + optional shell script
- 🔌 Easily extendable to test other strategies

---

## 🚀 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/ets-testbed/flower.git
cd flower_research_extension
```

### 2. Install Dependencies

It is recommended to use a Python virtual environment.

```bash
pip install -e framework[simulation]
pip install -e ./datasets
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install wandb
pip install scikit-learn
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 3. Run the Experiment

```bash
python -m flower_research_extension.experiments.run_experiment
```

To batch multiple runs with different hyperparameters:

```bash
cd flower_research_extension/experiments
bash hyperparam_runs.sh
```

---

## 🧩 Adding Your Own Plugin

To create a custom plugin:

1. Subclass `MetricsPlugin` from `plugins/base.py`
2. Implement one or more of:
  - `on_round_end(...)`
  - `on_client_result(...)`
3. Add it to the `plugins` list in `run_experiment.py`

---

## 📈 Example Output

```
results/
├── logs/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── global_metrics.csv
│       └── client_metrics.csv
└── wandb/
    └── Weights & Biases online dashboard
```

Each file contains round-by-round accuracy/loss logs.

---

## 📬 Questions?

Open an issue or reach out
via [Flower Slack](https://friendly-flower.slack.com/join/shared_invite/zt-35epydsx3-_e~KjYPEcyevkJZ4Ja3XkA#/shared-invite/email).
