# Local Flower Framework Setup for Development

This guide explains how to install and use the Flower framework locally in editable mode for development and debugging purposes.

## 📁 Repository Structure

After cloning the official Flower repo, the relevant structure is:

```
flower/
├── framework/
│   ├── pyproject.toml  ✅
│   ├── src/
│   │   └── flwr/        ✅ Core Flower framework
```

## ✅ Setup Steps

1. **Clone the Repository**
   ```bash
   git clone git@github.com:ets-testbed/flower.git
   cd flower/framework
   ```

2. **(Optional but Recommended) Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install Flower in Editable Mode**
   ```bash
   pip install -e .
   ```

4. **Verify Installation**
   Run Python and check the source path:
   ```python
   import flwr
   print(flwr.__file__)
   ```
   You should see a path like:
   ```
   .../flower/framework/src/flwr/__init__.py
   ```

## 📦 Using Imports in Your Code

You **do not need to change any imports**. Keep using:

```python
from flwr.client import Client, NumPyClient
from flwr.simulation import run_simulation
from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters
```

These imports will now reference the **local editable source** you installed.

## 🛠️ Making and Testing Changes

- Edit source files under:  
  `flower/framework/src/flwr/`
  
- Your changes will be applied immediately (no reinstall required)

- To run your custom simulations, use:
  ```bash
  python path/to/your_script.py
  ```

## 🔍 Troubleshooting

- If `flwr.__file__` points to `site-packages`, the local install didn't succeed. Make sure you ran:
  ```bash
  cd flower/framework
  pip install -e .
  ```

## ✅ Tip: Avoid `pip install flwr`

Once you're using a local editable install, do **not** run `pip install flwr` again — that would overwrite your setup.
