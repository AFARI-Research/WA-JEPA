"""Small generic registry for optional model registration."""

from __future__ import annotations

from typing import Any, Type

from utils.utils import cfg_get


class _Registry:
    def __init__(self, name: str):
        self._name = name
        self._registry: dict[str, Type] = {}

    def register(self, name: str):
        def decorator(cls):
            if name in self._registry and self._registry[name] is not cls:
                raise ValueError(
                    f"[{self._name}] '{name}' already registered by "
                    f"{self._registry[name].__name__}"
                )
            self._registry[name] = cls
            return cls

        return decorator

    def build(self, cfg: Any, **kwargs):
        name = cfg_get(cfg, "name", None)
        if name is None:
            raise ValueError(f"[{self._name}] config must have a 'name' field")
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "<none>"
            raise KeyError(f"[{self._name}] '{name}' not found. Available: {available}")
        return self._registry[name](cfg, **kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def keys(self):
        return self._registry.keys()


MODEL_REGISTRY = _Registry("ModelRegistry")
