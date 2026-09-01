"""Model factory for the reset project.

Prefer ``model.target`` for new work:

    model:
      target: my_package.my_model:MyModel

The target class is instantiated as ``MyModel(model_cfg, full_cfg=config)``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch.nn as nn

from models.registry import MODEL_REGISTRY
from utils.utils import cfg_get


def _load_target(target: str):
    if ":" in target:
        module_name, class_name = target.split(":", 1)
    else:
        module_name, class_name = target.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def build_world_model(config: Any) -> nn.Module:
    model_cfg = cfg_get(config, "model", None)
    if model_cfg is None:
        raise ValueError("Config must define a top-level `model:` section.")

    target = cfg_get(model_cfg, "target", None)
    if target:
        cls = _load_target(str(target))
        model = cls(model_cfg, full_cfg=config)
        if not isinstance(model, nn.Module):
            raise TypeError(f"{target} did not return a torch.nn.Module")
        return model

    name = cfg_get(model_cfg, "name", None)
    if name:
        return MODEL_REGISTRY.build(model_cfg, full_cfg=config)

    raise ValueError(
        "No model implementation is configured. Set `model.target` to a dotted "
        "class path, for example `models.my_model:MyModel`, or register a model "
        "and set `model.name`."
    )
