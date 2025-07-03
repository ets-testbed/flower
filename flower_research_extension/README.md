# 🌼 Flower Research Extension

This repository is a modular research-oriented extension for the [Flower Federated Learning Framework](https://flower.dev).  
It introduces a **plugin-based metrics logging system**, **custom training strategies**, and a reproducible **experiment runner** for benchmarking federated setups like CIFAR-10 with `FedAvg`.

---

## 📂 Project Structure

```
flower_research_extension/
│
├── data/                    # Dataset loaders (e.g. CIFAR-10)
│   └── cifar10.py
│
├── models/                  # Neural network models and parameter utilities
│   └── model.py
│
├── plugins/                 # Hookable metrics plugins
│   ├── base.py              # Abstract plugin interface
│   ├── csv_logger.py        # Logs round/client metrics to CSV
│   └── wandb_logger.py      # Logs to Weights & Biases
│
├── strategies/              # Custom strategy wrappers
│   ├── custom_fedavg.py     # Customizable FedAvg variant
│   ├── hooked_strategy.py   # Strategy that calls plugin hooks
│   └── round_timer.py       # Adds timing hooks for each round
│
├── utils/                   # Utility files (init, helpers)
│   └── __init__.py
│
├── client.py                # Client logic using Flower's ClientApp
├── training.py              # Fit and evaluate functions
├── run_experiment.py        # Entrypoint for simulation run
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Features

- ✅ Plugin interface for metrics (`MetricsPlugin`)
- 📊 CSV and Weights & Biases logging
- ⏱️ Per-round timing and hooks
- 🧪 Simple run configuration for CIFAR-10 using `FedAvg`
- 🔌 Easily extendable to test other strategies (e.g., FedBN, FedPer, etc.)

---

## 🚀 How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/YOUR_USERNAME/flower_research_extension.git
cd flower_research_extension
```

### 2. Install Dependencies
It is recommended to use a Python virtual environment.

```bash
pip install -r requirements.txt
```

Make sure Flower, PyTorch, and Weights & Biases (optional) are installed.

### 3. Run the Experiment
```bash
python -m flower_research_extension.experiments.run_experiment
```

This will:
- Start a Flower simulation with 20 clients
- Train a small CNN on CIFAR-10 using `FedAvg`
- Log global/client metrics to:
  - `results/logs/run_<timestamp>/`
  - [Weights & Biases](https://wandb.ai/) if enabled

---

## 🧩 Adding Your Own Plugin

To create a custom plugin:
1. Subclass `MetricsPlugin` from `plugins/base.py`
2. Implement any of these hooks:
   - `on_round_end(...)`
   - `on_client_result(...)`
3. Add it to the `plugins` list in `run_experiment.py`

---

## 📈 Example Output

```
results/
└── logs/
    └── run_20250702_153000/
        ├── global_metrics_20250702_153000.csv
        └── client_metrics_20250702_153000.csv
```

Each file contains round-by-round accuracy/loss logs.

---

## 🧪 Notes

- This project is designed for research and prototyping.
- It does not modify the core Flower source code.
- To integrate into other projects, copy only the needed parts (`plugins`, `strategies`, etc.).

---

## 📬 Questions?

Open an issue or reach out via [Flower Slack](https://friendly-flower.slack.com/join/shared_invite/zt-35epydsx3-_e~KjYPEcyevkJZ4Ja3XkA#/shared-invite/email).
