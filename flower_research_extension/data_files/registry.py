
# =========================================
# file: flower_research_extension/data_files/registry.py
# =========================================
from __future__ import annotations
from typing import Dict, Type
from pathlib import Path
from .base import DatasetProvider

class DatasetRegistry:
    def __init__(self) -> None:
        self._providers: Dict[str, DatasetProvider] = {}

    def register(self, provider: DatasetProvider) -> None:
        key = provider.name.lower()
        self._providers[key] = provider

    def get(self, name: str) -> DatasetProvider:
        key = (name or "").lower()
        if key not in self._providers:
            available = ", ".join(sorted(self._providers.keys()))
            raise ValueError(f"Unknown dataset '{name}'. Available: [{available}]")
        return self._providers[key]

    def available(self):
        return sorted(self._providers.keys())

REGISTRY = DatasetRegistry()
