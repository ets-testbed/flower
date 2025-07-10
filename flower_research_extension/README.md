# 🌼 Flower Research Extension

This repository is a modular research-oriented extension for the [Flower Federated Learning Framework](https://flower.dev). It introduces a **plugin-based metrics logging system**, **custom strategy wrappers**, and a **flexible experiment runner** for benchmarking federated setups like CIFAR-10 with `FedAvg`.

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
├── experiments/                 # Entrypoint and utilities for experiments
│   ├── run_experiment.py        # Entrypoint for simulation run
│   ├── experiment_setup.py      # Common logic for modular setup
│   ├── hyperparam_runs.sh       # Sample script for multiple runs
│   └── setup.sh                 # All-in-one setup & install script
│
├── client.py                    # Client logic using Flower's ClientApp
├── training.py                  # Fit and evaluate functions
├── requirements.txt             # Pinned dependencies
└── README.md
```

---

## 🚀 Quickstart

These steps work on Linux with pip.

1. **Clone the Flower framework**  
   ```bash
   git clone https://github.com/ets-testbed/flower.git
   cd flower
   ```

2. **Run the all-in-one setup script**  
   ```bash
   bash flower_research_extension/setup.sh
   ```
   This will:
   - Create and activate a Python 3 virtual environment (`venv`)
   - Install core dependencies:
     - `framework[simulation]`
     - `./datasets`
     - PyTorch + CUDA 12.1
     - `wandb`, `scikit-learn`
   - Print instructions to activate the env and run your experiment

3. **Activate and launch**  
   ```bash
   source venv/bin/activate
   python -m flower_research_extension.experiments.run_experiment
   ```

4. **Batch runs (optional)**  
   ```bash
   cd flower_research_extension/experiments
   bash hyperparam_runs.sh
   ```

---

## 🛠️ Custom Setup

- To use **conda** instead of `venv`, run:
  ```bash
  bash flower_research_extension/experiments/setup.sh --conda
  ```
- On **Windows**, replace activation with:
  ```powershell
  venv\Scripts\activate
  ```

---

## 🧩 Adding Your Own Plugin

1. Subclass `MetricsPlugin` in `plugins/base.py`  
2. Implement one or more hooks:  
   - `on_round_end(...)`  
   - `on_client_result(...)`  
3. Add your plugin class to the `plugins` list in `run_experiment.py`

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

Each run folder contains round-by-round accuracy/loss logs.

---

## 📬 Questions?

Open an issue or join us on [Flower Slack](https://friendly-flower.slack.com/join/shared_invite/zt-35epydsx3-_e~KjYPEcyevkJZ4Ja3XkA#/shared-invite/email).
