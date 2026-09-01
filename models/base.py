"""Base model contract for new experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class ModelBase(nn.Module, ABC):
    """Minimal interface expected by ``WorldModelTrainer``.

    Subclasses should implement ``forward`` and return:

    ``{"loss": {"loss_name": scalar_tensor}, "other_log": {"metric": scalar_tensor}}``
    """

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def scalar_loss(value: torch.Tensor, name: str = "loss") -> dict[str, Any]:
        if value.dim() != 0:
            raise ValueError(f"{name} must be a scalar tensor, got shape={tuple(value.shape)}")
        return {"loss": {name: value}}
