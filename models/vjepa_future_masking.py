"""Future-frame mask sampling for causal driving JEPA."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class FutureMaskBatch:
    masks_x: list[torch.Tensor]
    masks_y: list[torch.Tensor]
    indices: list[torch.Tensor]
    mask_indices: list[int]
    full_mask: torch.Tensor
    mask_ratios: torch.Tensor
    num_tokens: int
    num_history_tokens: int
    num_future_tokens: int


class FutureMaskSampler:
    """Sample per-example context/target indices for future masked prediction.

    Token order follows V-JEPA tubelet order: temporal, height, width.
    History tubelets are always visible. Future tubelets are masked with
    V-JEPA-style multi-block target masks.
    """

    def __init__(
        self,
        *,
        num_history_frames: int,
        num_future_frames: int,
        image_size: tuple[int, int],
        patch_size: int = 16,
        tubelet_size: int = 2,
        official_mask_configs: list[dict[str, Any]] | None = None,
    ) -> None:
        num_history_frames = int(num_history_frames)
        num_future_frames = int(num_future_frames)
        patch_size = int(patch_size)
        tubelet_size = int(tubelet_size)
        if num_history_frames <= 0:
            raise ValueError("num_history_frames must be > 0")
        if num_future_frames <= 0:
            raise ValueError("num_future_frames must be > 0")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if tubelet_size <= 0:
            raise ValueError("tubelet_size must be > 0")
        if num_history_frames % tubelet_size != 0:
            raise ValueError("num_history_frames must be divisible by tubelet_size")
        if num_future_frames % tubelet_size != 0:
            raise ValueError("num_future_frames must be divisible by tubelet_size")

        h, w = int(image_size[0]), int(image_size[1])
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_history_frames = num_history_frames
        self.num_future_frames = num_future_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_h = h // patch_size
        self.grid_w = w // patch_size
        self.tokens_per_step = self.grid_h * self.grid_w
        self.history_steps = num_history_frames // tubelet_size
        self.future_steps = num_future_frames // tubelet_size
        self.num_history_tokens = self.history_steps * self.tokens_per_step
        self.num_future_tokens = self.future_steps * self.tokens_per_step
        self.num_tokens = self.num_history_tokens + self.num_future_tokens
        self.official_mask_configs = self._validate_official_mask_configs(official_mask_configs)

        self._history_indices = torch.arange(self.num_history_tokens, dtype=torch.long)
        self._future_indices = torch.arange(
            self.num_history_tokens,
            self.num_tokens,
            dtype=torch.long,
        )
        self._sample_counter = 0

    @staticmethod
    def _validate_ratio_range(name: str, value: tuple[float, float] | list[float]) -> tuple[float, float]:
        try:
            values = list(value)
        except TypeError as exc:
            raise ValueError(f"{name} must have exactly two values") from exc
        if len(values) != 2:
            raise ValueError(f"{name} must have exactly two values")
        lo, hi = (float(values[0]), float(values[1]))
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"{name} values must be finite")
        if lo < 0.0 or hi > 1.0 or lo > hi:
            raise ValueError(f"{name} must satisfy 0 <= low <= high <= 1, got ({lo}, {hi})")
        return lo, hi

    def _validate_official_mask_configs(
        self,
        configs: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if configs is None:
            configs = [
                {
                    "num_blocks": 8,
                    "spatial_scale": (0.15, 0.15),
                    "temporal_scale": (1.0, 1.0),
                    "aspect_ratio": (0.75, 1.5),
                },
                {
                    "num_blocks": 2,
                    "spatial_scale": (0.7, 0.7),
                    "temporal_scale": (1.0, 1.0),
                    "aspect_ratio": (0.75, 1.5),
                },
            ]
        out: list[dict[str, Any]] = []
        for i, cfg in enumerate(configs):
            if not isinstance(cfg, Mapping):
                raise ValueError(f"official_mask_configs[{i}] must be a mapping")
            num_blocks = int(cfg.get("num_blocks", 1))
            if num_blocks <= 0:
                raise ValueError(f"official_mask_configs[{i}].num_blocks must be > 0")
            out.append(
                {
                    "num_blocks": num_blocks,
                    "spatial_scale": self._validate_ratio_range(
                        f"official_mask_configs[{i}].spatial_scale",
                        cfg.get("spatial_scale", (0.15, 0.15)),
                    ),
                    "temporal_scale": self._validate_ratio_range(
                        f"official_mask_configs[{i}].temporal_scale",
                        cfg.get("temporal_scale", (1.0, 1.0)),
                    ),
                    "aspect_ratio": self._validate_positive_range(
                        f"official_mask_configs[{i}].aspect_ratio",
                        cfg.get("aspect_ratio", (0.75, 1.5)),
                    ),
                    "max_keep": cfg.get("max_keep", None),
                }
            )
        if not out:
            raise ValueError("official_mask_configs must contain at least one mask spec")
        return out

    @staticmethod
    def _validate_positive_range(name: str, value: tuple[float, float] | list[float]) -> tuple[float, float]:
        try:
            values = list(value)
        except TypeError as exc:
            raise ValueError(f"{name} must have exactly two values") from exc
        if len(values) != 2:
            raise ValueError(f"{name} must have exactly two values")
        lo, hi = (float(values[0]), float(values[1]))
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"{name} values must be finite")
        if lo <= 0.0 or hi <= 0.0 or lo > hi:
            raise ValueError(f"{name} must satisfy 0 < low <= high, got ({lo}, {hi})")
        return lo, hi

    def _make_generator(self) -> torch.Generator:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        seed = int(torch.initial_seed()) + 1000003 * int(rank) + int(self._sample_counter)
        self._sample_counter += 1
        generator = torch.Generator()
        generator.manual_seed(seed % (2**63 - 1))
        return generator

    def _block_tokens(
        self,
        *,
        t0: int,
        t_len: int,
        h0: int,
        h_len: int,
        w0: int,
        w_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        t = torch.arange(t0, t0 + t_len, device=device)
        h = torch.arange(h0, h0 + h_len, device=device)
        w = torch.arange(w0, w0 + w_len, device=device)
        tt, hh, ww = torch.meshgrid(t, h, w, indexing="ij")
        return (tt * self.tokens_per_step + hh * self.grid_w + ww).reshape(-1)

    def _sample_block_location(
        self,
        block_size: tuple[int, int, int],
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        t_len, h_len, w_len = block_size
        t0 = int(torch.randint(0, self.future_steps - t_len + 1, (), device=device, generator=generator).item())
        h0 = int(torch.randint(0, self.grid_h - h_len + 1, (), device=device, generator=generator).item())
        w0 = int(torch.randint(0, self.grid_w - w_len + 1, (), device=device, generator=generator).item())
        return self._block_tokens(
            t0=t0,
            t_len=t_len,
            h0=h0,
            h_len=h_len,
            w0=w0,
            w_len=w_len,
            device=device,
        )

    def _sample_official_block_size(
        self,
        spec: dict[str, Any],
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[int, int, int]:
        min_t, max_t = spec["temporal_scale"]
        temporal_scale = float(torch.empty((), device=device).uniform_(min_t, max_t, generator=generator).item())
        t_len = max(1, int(self.future_steps * temporal_scale))
        t_len = min(t_len, self.future_steps)

        min_s, max_s = spec["spatial_scale"]
        spatial_scale = float(torch.empty((), device=device).uniform_(min_s, max_s, generator=generator).item())
        spatial_tokens = max(1, int(self.tokens_per_step * spatial_scale))
        spatial_tokens = min(spatial_tokens, self.tokens_per_step)

        min_ar, max_ar = spec["aspect_ratio"]
        aspect_ratio = float(torch.empty((), device=device).uniform_(min_ar, max_ar, generator=generator).item())
        h_len = int(round(math.sqrt(spatial_tokens * aspect_ratio)))
        w_len = int(round(math.sqrt(spatial_tokens / aspect_ratio)))
        h_len = min(max(h_len, 1), self.grid_h)
        w_len = min(max(w_len, 1), self.grid_w)
        return t_len, h_len, w_len

    def _sample_official_target_local(
        self,
        *,
        spec: dict[str, Any],
        block_size: tuple[int, int, int],
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        selected = torch.zeros(self.num_future_tokens, dtype=torch.bool, device=device)
        for _ in range(int(spec["num_blocks"])):
            selected[self._sample_block_location(block_size, device=device, generator=generator)] = True
        target = torch.nonzero(selected, as_tuple=False).flatten()
        if target.numel() == 0:
            # Official masks resample until usable. A tiny fallback keeps the
            # predictor target non-empty without changing the surrounding logic.
            target = torch.randint(0, self.num_future_tokens, (1,), device=device, generator=generator)
        return torch.sort(target).values

    def sample(self, batch_size: int, *, device: torch.device) -> FutureMaskBatch:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        sample_device = torch.device("cpu")
        generator = self._make_generator()
        future = self._future_indices.to(sample_device)
        history = self._history_indices.to(sample_device)
        arange = torch.arange(batch_size, dtype=torch.long, device=device)

        masks_x: list[torch.Tensor] = []
        masks_y: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        mask_indices: list[int] = []
        ratio_rows: list[torch.Tensor] = []
        full_rows: list[torch.Tensor] = []

        for spec_idx, spec in enumerate(self.official_mask_configs, start=1):
            block_size = self._sample_official_block_size(
                spec,
                device=sample_device,
                generator=generator,
            )
            targets: list[torch.Tensor] = []
            visible: list[torch.Tensor] = []
            for _ in range(batch_size):
                target_local = self._sample_official_target_local(
                    spec=spec,
                    block_size=block_size,
                    device=sample_device,
                    generator=generator,
                )
                keep = torch.ones(self.num_future_tokens, dtype=torch.bool, device=sample_device)
                keep[target_local] = False
                visible_local = torch.nonzero(keep, as_tuple=False).flatten()
                targets.append(target_local)
                visible.append(visible_local)

            min_keep_visible = min(int(v.numel()) for v in visible)
            max_keep = spec.get("max_keep", None)
            if max_keep is not None:
                min_keep_visible = min(min_keep_visible, int(max_keep))
            min_keep_pred = max(1, min(int(t.numel()) for t in targets))

            group_x: list[torch.Tensor] = []
            group_y: list[torch.Tensor] = []
            group_ratios: list[float] = []
            group_full: list[bool] = []
            for target_local, visible_local in zip(targets, visible):
                target_local = target_local[:min_keep_pred]
                visible_local = visible_local[:min_keep_visible]
                target = future[target_local]
                visible_future = future[visible_local]
                group_x.append(torch.cat([history, visible_future], dim=0))
                group_y.append(target)
                group_ratios.append(float(target.numel()) / float(self.num_future_tokens))
                group_full.append(target.numel() == self.num_future_tokens)

            masks_x.append(torch.stack(group_x, dim=0).to(device=device, non_blocking=True))
            masks_y.append(torch.stack(group_y, dim=0).to(device=device, non_blocking=True))
            indices.append(arange)
            mask_indices.append(0)
            ratio_rows.append(torch.tensor(group_ratios, dtype=torch.float32, device=device))
            full_rows.append(torch.tensor(group_full, dtype=torch.bool, device=device))

        return FutureMaskBatch(
            masks_x=masks_x,
            masks_y=masks_y,
            indices=indices,
            mask_indices=mask_indices,
            full_mask=torch.stack(full_rows, dim=0),
            mask_ratios=torch.stack(ratio_rows, dim=0),
            num_tokens=self.num_tokens,
            num_history_tokens=self.num_history_tokens,
            num_future_tokens=self.num_future_tokens,
        )

    def full_future_mask(
        self,
        batch_size: int,
        *,
        device: torch.device,
        mask_index: int = 0,
    ) -> FutureMaskBatch:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        history = self._history_indices.to(device)
        future = self._future_indices.to(device)
        masks_x = history.unsqueeze(0).repeat(batch_size, 1)
        masks_y = future.unsqueeze(0).repeat(batch_size, 1)
        return FutureMaskBatch(
            masks_x=[masks_x],
            masks_y=[masks_y],
            indices=[torch.arange(batch_size, dtype=torch.long, device=device)],
            mask_indices=[int(mask_index)],
            full_mask=torch.ones(batch_size, dtype=torch.bool, device=device),
            mask_ratios=torch.ones(batch_size, dtype=torch.float32, device=device),
            num_tokens=self.num_tokens,
            num_history_tokens=self.num_history_tokens,
            num_future_tokens=self.num_future_tokens,
        )
