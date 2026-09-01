"""Minimal placeholder model for smoke-testing the training scaffold.

This is intentionally not a real world model. It keeps a single trainable
parameter so optimizer, checkpoint, logging, and inline eval paths can run while
new model code is being designed.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from utils.utils import cfg_get


class PlaceholderWorldModel(nn.Module):
    """A tiny stand-in that satisfies the trainer and NAVSIM eval contracts."""

    def __init__(self, cfg: Any, full_cfg: Any | None = None):
        super().__init__()
        self.cfg = cfg
        self.full_cfg = full_cfg or {}
        self.dummy = nn.Parameter(torch.zeros(()))

        data_cfg = cfg_get(self.full_cfg, "data", {}) or {}
        self.horizon = int(cfg_get(data_cfg, "num_future_frames", 8))

    def _configured_loss_names(self) -> list[str]:
        trainer_cfg = cfg_get(self.full_cfg, "trainer", {}) or {}
        losses = cfg_get(trainer_cfg, "losses", {}) or {}
        if isinstance(losses, dict) and losses:
            return [str(name) for name in losses.keys()]
        return ["placeholder_loss"]

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        zero = self.dummy * 0.0
        loss_dict = {name: zero for name in self._configured_loss_names()}
        return {
            "loss": loss_dict,
            "other_log": {"placeholder_zero": zero.detach()},
        }

    @torch.no_grad()
    def predict_trajectory(self, features: dict[str, Any]) -> torch.Tensor:
        batch_size = 1
        device = self.dummy.device
        for value in features.values():
            if isinstance(value, torch.Tensor):
                batch_size = int(value.shape[0]) if value.dim() > 0 else 1
                device = value.device
                break
        return torch.zeros(batch_size, self.horizon, 3, device=device)
