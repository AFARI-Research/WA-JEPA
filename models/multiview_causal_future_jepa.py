"""Multi-view causal future-masked JEPA for driving trajectory prediction."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import ModelBase
from models.jepa_common import (
    _as_float_tuple,
    _as_pair,
    _chunk_layer_norm,
    _checkpoint_path_list,
    _load_state_from_path,
    _load_submodule_state,
    _module_param_device,
    _module_param_dtype,
    _move_tensor_for_module,
    _select_checkpoint_state,
)
from models.vjepa_future_masking import FutureMaskBatch, FutureMaskSampler
from third_party.vjepa2.models import vision_transformer as vjepa_vit
from utils.utils import cfg_get

logger = logging.getLogger(__name__)


@dataclass
class DrivingConditionBatch:
    target_trajectory: torch.Tensor | None = None
    predictor_inputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def select(self, idx: torch.Tensor) -> "DrivingConditionBatch":
        target = None if self.target_trajectory is None else self.target_trajectory.index_select(0, idx)
        return DrivingConditionBatch(
            target_trajectory=target,
            predictor_inputs={
                key: value.index_select(0, idx)
                for key, value in self.predictor_inputs.items()
            },
        )


@dataclass
class TrajectoryFlowBatch:
    clean_target: torch.Tensor
    raw_target: torch.Tensor
    predictor_inputs: dict[str, torch.Tensor]


class DrivingConditionAdapter:
    tensor_keys: tuple[str, ...] = ()

    def prepare_batch(
        self,
        batch: dict[str, Any],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Any]:
        out = dict(batch)
        for key in self.tensor_keys:
            value = out.get(key)
            if isinstance(value, torch.Tensor):
                out[key] = _move_tensor_for_module(value, dtype=dtype, device=device)
        return out

    def training_conditions(
        self,
        batch: dict[str, Any],
        *,
        require_targets: bool,
    ) -> DrivingConditionBatch:
        raise NotImplementedError

    def inference_conditions(self, batch: dict[str, Any]) -> DrivingConditionBatch:
        raise NotImplementedError

    def has_training_conditions(self, conditions: DrivingConditionBatch) -> bool:
        return conditions.target_trajectory is not None and bool(conditions.predictor_inputs)

    def zero_conditions(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trajectory_horizon: int,
        history_steps: int,
        ego_status_dim: int,
    ) -> DrivingConditionBatch:
        raise NotImplementedError

    def normalize_trajectory(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def denormalize_trajectory(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def prepare_training_flow(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        conditions: DrivingConditionBatch,
        *,
        t_cont: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> TrajectoryFlowBatch:
        raise NotImplementedError

    def prepare_inference_inputs(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        conditions: DrivingConditionBatch,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def initial_inference_trajectory(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        sample_device: torch.device | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def trajectory_logs(self, preds: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def inference_output(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class NavsimDrivingConditionAdapter(DrivingConditionAdapter):
    tensor_keys = ("ego_status", "history_trajectory", "future_trajectory")

    @staticmethod
    def _optional_tensor(batch: dict[str, Any], key: str) -> torch.Tensor | None:
        value = batch.get(key)
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Batch field {key!r} must be a torch.Tensor, got {type(value).__name__}")
        return value

    @staticmethod
    def _require_tensor(batch: dict[str, Any], key: str) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"Batch field {key!r} is required")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Batch field {key!r} must be a torch.Tensor, got {type(value).__name__}")
        return value

    def training_conditions(
        self,
        batch: dict[str, Any],
        *,
        require_targets: bool,
    ) -> DrivingConditionBatch:
        if require_targets:
            return DrivingConditionBatch(
                target_trajectory=self._require_tensor(batch, "future_trajectory"),
                predictor_inputs={
                    "ego_status": self._require_tensor(batch, "ego_status"),
                    "history_trajectory": self._require_tensor(batch, "history_trajectory"),
                },
            )
        inputs: dict[str, torch.Tensor] = {}
        ego_status = self._optional_tensor(batch, "ego_status")
        history_trajectory = self._optional_tensor(batch, "history_trajectory")
        if ego_status is not None:
            inputs["ego_status"] = ego_status
        if history_trajectory is not None:
            inputs["history_trajectory"] = history_trajectory
        return DrivingConditionBatch(
            target_trajectory=self._optional_tensor(batch, "future_trajectory"),
            predictor_inputs=inputs,
        )

    def inference_conditions(self, batch: dict[str, Any]) -> DrivingConditionBatch:
        return DrivingConditionBatch(
            predictor_inputs={
                "ego_status": self._require_tensor(batch, "ego_status"),
                "history_trajectory": self._require_tensor(batch, "history_trajectory"),
            },
        )

    def has_training_conditions(self, conditions: DrivingConditionBatch) -> bool:
        return (
            conditions.target_trajectory is not None
            and "ego_status" in conditions.predictor_inputs
            and "history_trajectory" in conditions.predictor_inputs
        )

    def zero_conditions(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trajectory_horizon: int,
        history_steps: int,
        ego_status_dim: int,
    ) -> DrivingConditionBatch:
        return DrivingConditionBatch(
            target_trajectory=torch.zeros(batch_size, trajectory_horizon, 3, device=device, dtype=dtype),
            predictor_inputs={
                "ego_status": torch.zeros(batch_size, ego_status_dim, device=device, dtype=dtype),
                "history_trajectory": torch.zeros(batch_size, history_steps, 3, device=device, dtype=dtype),
            },
        )

    def normalize_trajectory(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        mean = model.trajectory_norm_mean.to(device=trajectory.device, dtype=trajectory.dtype)
        std = model.trajectory_norm_std.to(device=trajectory.device, dtype=trajectory.dtype)
        return (trajectory - mean) / std

    def denormalize_trajectory(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        mean = model.trajectory_norm_mean.to(device=trajectory.device, dtype=trajectory.dtype)
        std = model.trajectory_norm_std.to(device=trajectory.device, dtype=trajectory.dtype)
        return trajectory * std + mean

    def prepare_training_flow(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        conditions: DrivingConditionBatch,
        *,
        t_cont: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> TrajectoryFlowBatch:
        if conditions.target_trajectory is None:
            raise RuntimeError("NAVSIM driving condition adapter did not provide a trajectory target")
        raw_target = conditions.target_trajectory.float().to(dtype=dtype, device=device)
        clean_target = self.normalize_trajectory(model, raw_target)
        noisy_trajectory, _noise = model._flow_interpolate(clean_target, t_cont)
        predictor_inputs = {
            key: value.to(dtype=dtype, device=device)
            for key, value in conditions.predictor_inputs.items()
        }
        predictor_inputs["noisy_trajectory"] = noisy_trajectory
        if "history_trajectory" in predictor_inputs:
            predictor_inputs["history_trajectory"] = self.normalize_trajectory(
                model,
                predictor_inputs["history_trajectory"],
            )
        return TrajectoryFlowBatch(
            clean_target=clean_target,
            raw_target=raw_target,
            predictor_inputs=predictor_inputs,
        )

    def prepare_inference_inputs(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        conditions: DrivingConditionBatch,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        predictor_inputs = {
            key: value.to(dtype=dtype, device=device)
            for key, value in conditions.predictor_inputs.items()
        }
        if "history_trajectory" in predictor_inputs:
            predictor_inputs["history_trajectory"] = self.normalize_trajectory(
                model,
                predictor_inputs["history_trajectory"],
            )
        return predictor_inputs

    def initial_inference_trajectory(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        sample_device: torch.device | None,
    ) -> torch.Tensor:
        return model._randn_for_inference(
            (batch_size, model.predictor.trajectory_horizon, 3),
            device=device,
            dtype=dtype,
            generator=generator,
            sample_device=sample_device,
        ) * model.flow_inference_noise_scale

    def trajectory_logs(self, preds: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        traj_preds = [p for p in preds if bool(p["compute_traj_loss"])]
        if not traj_preds:
            zero = preds[0]["pred_scene"].detach().float().new_zeros(()) if preds else torch.zeros(())
            return {
                "active_traj_ade": zero,
                "active_traj_fde": zero,
            }
        pred = torch.cat([p["pred_traj_raw"].detach().float() for p in traj_preds], dim=0)
        target = torch.cat([p["target_traj_raw"].detach().float() for p in traj_preds], dim=0)
        err = torch.linalg.vector_norm(pred[..., :2] - target[..., :2], dim=-1)
        return {
            "active_traj_ade": err.mean(),
            "active_traj_fde": err[:, -1].mean(),
        }

    def inference_output(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        return self.denormalize_trajectory(model, trajectory).float()


class Bench2DriveDrivingConditionAdapter(DrivingConditionAdapter):
    def _raise_unimplemented(self) -> None:
        raise NotImplementedError(
            "driving_condition_adapter='bench2drive' is reserved but not implemented yet"
        )

    def training_conditions(
        self,
        batch: dict[str, Any],
        *,
        require_targets: bool,
    ) -> DrivingConditionBatch:
        self._raise_unimplemented()

    def inference_conditions(self, batch: dict[str, Any]) -> DrivingConditionBatch:
        self._raise_unimplemented()

    def zero_conditions(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trajectory_horizon: int,
        history_steps: int,
        ego_status_dim: int,
    ) -> DrivingConditionBatch:
        self._raise_unimplemented()

    def normalize_trajectory(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        self._raise_unimplemented()

    def denormalize_trajectory(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        self._raise_unimplemented()

    def prepare_training_flow(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        conditions: DrivingConditionBatch,
        *,
        t_cont: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> TrajectoryFlowBatch:
        self._raise_unimplemented()

    def prepare_inference_inputs(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        conditions: DrivingConditionBatch,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        self._raise_unimplemented()

    def initial_inference_trajectory(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        sample_device: torch.device | None,
    ) -> torch.Tensor:
        self._raise_unimplemented()

    def trajectory_logs(self, preds: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        self._raise_unimplemented()

    def inference_output(
        self,
        model: "MultiViewCausalFutureMaskedJEPA",
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        self._raise_unimplemented()


def _build_driving_condition_adapter(name: str) -> DrivingConditionAdapter:
    normalized = str(name).lower()
    if normalized in ("navsim", "nuplan_navsim"):
        return NavsimDrivingConditionAdapter()
    if normalized in ("bench2drive", "b2d"):
        return Bench2DriveDrivingConditionAdapter()
    raise ValueError(
        "Unknown driving_condition_adapter: "
        f"{name!r}. Supported values are 'navsim' and 'bench2drive'."
    )


def _camera_names(value: Any) -> list[str]:
    if value is None:
        return ["cam_l0", "cam_f0", "cam_r0", "cam_b0"]
    if isinstance(value, str):
        return [value.lower()]
    names = [str(v).lower() for v in value]
    if not names:
        raise ValueError("camera_names must contain at least one camera")
    return names


class SceneTokenProjector(nn.Module):
    """Lightweight projector from per-camera V-JEPA tokens to scene-token feature space."""

    def __init__(
        self,
        *,
        target_dim: int,
        scene_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        self.scene_dim = int(scene_dim)
        if self.scene_dim <= 0:
            raise ValueError("mv_scene_dim must be > 0")
        self.input_norm = nn.LayerNorm(self.target_dim)
        self.proj = nn.Linear(self.target_dim, self.scene_dim)

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"tokens must be [B,V,N,D], got {tuple(tokens.shape)}")
        return self.proj(self.input_norm(tokens))


class ScalarTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        return self.net(t[:, None])


class SceneInputPositioner(nn.Module):
    """Add camera/time/spatial embeddings to scene features inside the predictor."""

    def __init__(
        self,
        *,
        scene_dim: int,
        num_cameras: int,
        grid_size: tuple[int, int],
        max_time_steps: int,
    ) -> None:
        super().__init__()
        self.scene_dim = int(scene_dim)
        self.num_cameras = int(num_cameras)
        self.grid_h, self.grid_w = int(grid_size[0]), int(grid_size[1])
        self.max_time_steps = int(max_time_steps)
        if self.scene_dim <= 0:
            raise ValueError("scene_dim must be > 0")
        if self.num_cameras <= 0:
            raise ValueError("num_cameras must be > 0")
        if self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError("grid_size must contain positive values")
        if self.max_time_steps <= 0:
            raise ValueError("max_time_steps must be > 0")
        self.tokens_per_step = self.grid_h * self.grid_w
        self.camera_embed = nn.Parameter(torch.zeros(self.num_cameras, self.scene_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, self.max_time_steps, self.scene_dim))
        self.spatial_embed = nn.Parameter(torch.zeros(1, self.grid_h, self.grid_w, self.scene_dim))
        nn.init.normal_(self.camera_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.normal_(self.spatial_embed, std=0.02)
        self.out_norm = nn.LayerNorm(self.scene_dim)

    def _position_from_token_indices(
        self,
        token_indices: torch.Tensor,
        *,
        time_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_indices = token_indices.long()
        local_t = torch.div(token_indices, self.tokens_per_step, rounding_mode="floor")
        spatial = token_indices - local_t * self.tokens_per_step
        time_index = local_t + int(time_offset)
        if int(time_index.min().item()) < 0 or int(time_index.max().item()) >= self.max_time_steps:
            raise ValueError(
                "token_indices/time_offset produce time positions outside the configured "
                f"absolute time embedding range: offset={time_offset}, max_time_steps={self.max_time_steps}"
            )
        if int(spatial.min().item()) < 0 or int(spatial.max().item()) >= self.tokens_per_step:
            raise ValueError("token_indices contain spatial positions outside the V-JEPA patch grid")
        return time_index, spatial

    def _position_embeddings(
        self,
        token_indices: torch.Tensor,
        *,
        time_offset: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        time_index, spatial = self._position_from_token_indices(token_indices, time_offset=time_offset)
        bsz, num_tokens = token_indices.shape
        cam = self.camera_embed.to(dtype=dtype, device=device).view(1, self.num_cameras, 1, self.scene_dim)
        time = self.time_embed.to(dtype=dtype, device=device)[0].index_select(
            0,
            time_index.reshape(-1),
        ).reshape(bsz, num_tokens, self.scene_dim)
        spatial_embed = self.spatial_embed.to(dtype=dtype, device=device).reshape(
            self.tokens_per_step,
            self.scene_dim,
        )
        spatial_pos = spatial_embed.index_select(0, spatial.reshape(-1)).reshape(
            bsz,
            num_tokens,
            self.scene_dim,
        )
        return cam + time.unsqueeze(1) + spatial_pos.unsqueeze(1)

    def forward(
        self,
        tokens: torch.Tensor,
        token_indices: torch.Tensor,
        *,
        time_offset: int,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,V*N,D], got {tuple(tokens.shape)}")
        bsz, flat_tokens, dim = tokens.shape
        if dim != self.scene_dim:
            raise ValueError(f"Expected scene_dim={self.scene_dim}, got {dim}")
        if token_indices.ndim != 2 or token_indices.size(0) != bsz:
            raise ValueError(f"token_indices must be [B,N], got {tuple(token_indices.shape)}")
        num_tokens = token_indices.size(1)
        if flat_tokens != self.num_cameras * num_tokens:
            raise ValueError(
                "Flat scene token count must equal num_cameras * token_indices length, "
                f"got {flat_tokens} vs {self.num_cameras} * {num_tokens}"
            )
        tokens_by_view = tokens.reshape(bsz, self.num_cameras, num_tokens, self.scene_dim)
        pos = self._position_embeddings(
            token_indices,
            time_offset=time_offset,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return self.out_norm((tokens_by_view + pos).reshape(bsz, flat_tokens, self.scene_dim))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight.to(dtype=x.dtype, device=x.device)


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 3),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cond.ndim != 2 or cond.size(0) != x.size(0):
            raise ValueError(f"cond must be [B,D] for x [B,N,D], got {tuple(cond.shape)}")
        shift, scale, gate = self.modulation(cond).chunk(3, dim=-1)
        x = self.norm(x)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x, gate


class ModalityAttentionProjection(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim <= 0 or self.dim % self.num_heads != 0:
            raise ValueError("dim must be positive and divisible by num_heads")
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def pre_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_tokens, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k, v

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(x)


class ModalitySpecificJointAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        enable_trajectory_head: bool = True,
        traj_loss_grad_to_scene_flow: bool = False,
        scene_loss_grad_to_traj_flow: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.enable_trajectory_head = bool(enable_trajectory_head)
        self.traj_loss_grad_to_scene_flow = bool(traj_loss_grad_to_scene_flow)
        self.scene_loss_grad_to_traj_flow = bool(scene_loss_grad_to_traj_flow)
        if self.dim <= 0 or self.dim % self.num_heads != 0:
            raise ValueError("dim must be positive and divisible by num_heads")
        self.context_proj = ModalityAttentionProjection(self.dim, self.num_heads)
        self.scene_proj = ModalityAttentionProjection(self.dim, self.num_heads)
        if self.enable_trajectory_head:
            self.traj_proj = ModalityAttentionProjection(self.dim, self.num_heads)

    def forward(
        self,
        context: torch.Tensor,
        scene: torch.Tensor,
        traj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        context_len = context.size(1)
        scene_len = scene.size(1)
        context_qkv = self.context_proj.pre_attention(context)
        scene_qkv = self.scene_proj.pre_attention(scene)
        context_q, context_k, context_v = context_qkv
        scene_q, scene_k, scene_v = scene_qkv

        def merge_heads(tensor: torch.Tensor) -> torch.Tensor:
            bsz, _, num_tokens, _ = tensor.shape
            return tensor.transpose(1, 2).reshape(bsz, num_tokens, self.dim)

        if self.enable_trajectory_head:
            if traj is None:
                raise ValueError("traj must be provided when enable_trajectory_head=true")
            traj_qkv = self.traj_proj.pre_attention(traj)
            traj_q, traj_k, traj_v = traj_qkv

            scene_k_for_traj = scene_k if self.traj_loss_grad_to_scene_flow else scene_k.detach()
            scene_v_for_traj = scene_v if self.traj_loss_grad_to_scene_flow else scene_v.detach()
            traj_k_for_scene = traj_k if self.scene_loss_grad_to_traj_flow else traj_k.detach()
            traj_v_for_scene = traj_v if self.scene_loss_grad_to_traj_flow else traj_v.detach()

            context_k_parts = [
                context_k,
                scene_k_for_traj,
                traj_k_for_scene,
            ]
            context_v_parts = [
                context_v,
                scene_v_for_traj,
                traj_v_for_scene,
            ]
            scene_k_parts = [context_k, scene_k, traj_k_for_scene]
            scene_v_parts = [context_v, scene_v, traj_v_for_scene]
            traj_k_parts = [context_k, scene_k_for_traj, traj_k]
            traj_v_parts = [context_v, scene_v_for_traj, traj_v]

            def attend(
                query: torch.Tensor,
                key_parts: list[torch.Tensor],
                value_parts: list[torch.Tensor],
            ) -> torch.Tensor:
                return F.scaled_dot_product_attention(
                    query,
                    torch.cat(key_parts, dim=2),
                    torch.cat(value_parts, dim=2),
                )

            context_out = attend(context_q, context_k_parts, context_v_parts)
            scene_out = attend(scene_q, scene_k_parts, scene_v_parts)
            traj_out = attend(traj_q, traj_k_parts, traj_v_parts)
        else:
            q = torch.cat([context_q, scene_q], dim=2)
            k = torch.cat([context_k, scene_k], dim=2)
            v = torch.cat([context_v, scene_v], dim=2)
            x = F.scaled_dot_product_attention(q, k, v)
            context_out, scene_out = x.split([context_len, scene_len], dim=2)
            traj_out = None

        context_out = self.context_proj.post_attention(merge_heads(context_out))
        scene_out = self.scene_proj.post_attention(merge_heads(scene_out))
        if self.enable_trajectory_head:
            if traj_out is None:
                raise RuntimeError("trajectory attention output missing")
            traj_out = self.traj_proj.post_attention(merge_heads(traj_out))
        return context_out, scene_out, traj_out


def _flow_ffn(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Linear(dim * 4, dim),
    )


class JointFlowBlock(nn.Module):
    """Joint-attend context, future scene, and trajectory tokens with separate FFNs."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        enable_trajectory_head: bool = True,
        traj_loss_grad_to_scene_flow: bool = False,
        scene_loss_grad_to_traj_flow: bool = False,
    ) -> None:
        super().__init__()
        self.enable_trajectory_head = bool(enable_trajectory_head)
        self.context_attn_norm = AdaLayerNormZero(dim)
        self.scene_attn_norm = AdaLayerNormZero(dim)
        if self.enable_trajectory_head:
            self.traj_attn_norm = AdaLayerNormZero(dim)
        self.joint_attn = ModalitySpecificJointAttention(
            dim,
            num_heads,
            enable_trajectory_head=self.enable_trajectory_head,
            traj_loss_grad_to_scene_flow=traj_loss_grad_to_scene_flow,
            scene_loss_grad_to_traj_flow=scene_loss_grad_to_traj_flow,
        )
        self.context_ffn_norm = AdaLayerNormZero(dim)
        self.scene_ffn_norm = AdaLayerNormZero(dim)
        if self.enable_trajectory_head:
            self.traj_ffn_norm = AdaLayerNormZero(dim)
        self.context_ffn = _flow_ffn(dim)
        self.scene_ffn = _flow_ffn(dim)
        if self.enable_trajectory_head:
            self.traj_ffn = _flow_ffn(dim)

    def forward(
        self,
        context: torch.Tensor,
        scene: torch.Tensor,
        traj: torch.Tensor | None,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        context_norm, context_attn_gate = self.context_attn_norm(context, time)
        scene_norm, scene_attn_gate = self.scene_attn_norm(scene, time)
        if self.enable_trajectory_head:
            if traj is None:
                raise ValueError("traj must be provided when enable_trajectory_head=true")
            traj_norm, traj_attn_gate = self.traj_attn_norm(traj, time)
        else:
            traj_norm = None
        context_delta, scene_delta, traj_delta = self.joint_attn(context_norm, scene_norm, traj_norm)
        context = context + context_attn_gate.unsqueeze(1) * context_delta
        scene = scene + scene_attn_gate.unsqueeze(1) * scene_delta
        if self.enable_trajectory_head:
            if traj_delta is None:
                raise RuntimeError("trajectory attention delta missing")
            traj = traj + traj_attn_gate.unsqueeze(1) * traj_delta

        context_norm, context_ffn_gate = self.context_ffn_norm(context, time)
        scene_norm, scene_ffn_gate = self.scene_ffn_norm(scene, time)
        context = context + context_ffn_gate.unsqueeze(1) * self.context_ffn(context_norm)
        scene = scene + scene_ffn_gate.unsqueeze(1) * self.scene_ffn(scene_norm)
        if self.enable_trajectory_head:
            if traj is None:
                raise RuntimeError("trajectory hidden state missing")
            traj_norm, traj_ffn_gate = self.traj_ffn_norm(traj, time)
            traj = traj + traj_ffn_gate.unsqueeze(1) * self.traj_ffn(traj_norm)
        return context, scene, traj


class TrajectoryPredictorAdapter:
    name = "base"

    def init_modules(self, predictor: "SceneTrajectoryFlowPredictor") -> None:
        raise NotImplementedError

    def init_parameters(self, predictor: "SceneTrajectoryFlowPredictor") -> None:
        raise NotImplementedError

    def validate_inputs(
        self,
        predictor: "SceneTrajectoryFlowPredictor",
        inputs: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        raise NotImplementedError

    def encode(
        self,
        predictor: "SceneTrajectoryFlowPredictor",
        inputs: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, predictor: "SceneTrajectoryFlowPredictor", traj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NavsimTrajectoryPredictorAdapter(TrajectoryPredictorAdapter):
    name = "navsim"

    def init_modules(self, predictor: "SceneTrajectoryFlowPredictor") -> None:
        predictor.noisy_traj_proj = nn.Linear(3, predictor.hidden_dim)
        predictor.history_traj_encoder = nn.Sequential(
            nn.Linear(predictor.history_steps * 3, predictor.hidden_dim),
            nn.GELU(),
            nn.Linear(predictor.hidden_dim, predictor.hidden_dim),
        )
        predictor.ego_status_encoder = nn.Sequential(
            nn.Linear(predictor.ego_status_dim, predictor.hidden_dim),
            nn.GELU(),
            nn.Linear(predictor.hidden_dim, predictor.hidden_dim),
        )
        predictor.traj_fusion = nn.Linear(predictor.hidden_dim * 3, predictor.hidden_dim)
        predictor.traj_out = nn.Sequential(
            nn.LayerNorm(predictor.hidden_dim),
            nn.Linear(predictor.hidden_dim, predictor.hidden_dim),
            nn.GELU(),
            nn.Linear(predictor.hidden_dim, 3),
        )
        predictor.traj_type_embed = nn.Parameter(torch.zeros(1, 1, predictor.hidden_dim))
        predictor.traj_pos = nn.Parameter(
            torch.zeros(1, predictor.trajectory_horizon, predictor.hidden_dim)
        )

    def init_parameters(self, predictor: "SceneTrajectoryFlowPredictor") -> None:
        nn.init.normal_(predictor.traj_type_embed, std=0.02)
        nn.init.normal_(predictor.traj_pos, std=0.02)

    def validate_inputs(
        self,
        predictor: "SceneTrajectoryFlowPredictor",
        inputs: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        required = ("noisy_trajectory", "ego_status", "history_trajectory")
        missing = [key for key in required if key not in inputs]
        if missing:
            raise ValueError(
                "NAVSIM trajectory predictor inputs missing: " + ", ".join(missing)
            )
        noisy_trajectory = inputs["noisy_trajectory"]
        ego_status = inputs["ego_status"]
        history_trajectory = inputs["history_trajectory"]
        if tuple(noisy_trajectory.shape) != (batch_size, predictor.trajectory_horizon, 3):
            raise ValueError(
                "noisy_trajectory shape mismatch: "
                f"expected {(batch_size, predictor.trajectory_horizon, 3)}, "
                f"got {tuple(noisy_trajectory.shape)}"
            )
        if tuple(ego_status.shape) != (batch_size, predictor.ego_status_dim):
            raise ValueError(
                f"ego_status shape mismatch: expected {(batch_size, predictor.ego_status_dim)}, "
                f"got {tuple(ego_status.shape)}"
            )
        if tuple(history_trajectory.shape) != (batch_size, predictor.history_steps, 3):
            raise ValueError(
                "history_trajectory shape mismatch: "
                f"expected {(batch_size, predictor.history_steps, 3)}, "
                f"got {tuple(history_trajectory.shape)}"
            )

    def encode(
        self,
        predictor: "SceneTrajectoryFlowPredictor",
        inputs: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        noisy_trajectory = inputs["noisy_trajectory"]
        ego_status = inputs["ego_status"]
        history_trajectory = inputs["history_trajectory"]
        noisy_traj = predictor.noisy_traj_proj(noisy_trajectory.to(dtype=dtype))
        hist = predictor.history_traj_encoder(history_trajectory.to(dtype=dtype).flatten(1)).unsqueeze(1)
        ego = predictor.ego_status_encoder(ego_status.to(dtype=dtype)).unsqueeze(1)
        traj = predictor.traj_fusion(
            torch.cat(
                [
                    noisy_traj + predictor.traj_pos.to(dtype=dtype, device=noisy_traj.device),
                    hist.expand(-1, predictor.trajectory_horizon, -1),
                    ego.expand(-1, predictor.trajectory_horizon, -1),
                ],
                dim=-1,
            )
        )
        return traj + predictor.traj_type_embed.to(dtype=dtype, device=traj.device)

    def decode(self, predictor: "SceneTrajectoryFlowPredictor", traj: torch.Tensor) -> torch.Tensor:
        return predictor.traj_out(traj)


class Bench2DriveTrajectoryPredictorAdapter(TrajectoryPredictorAdapter):
    name = "bench2drive"

    def _raise_unimplemented(self) -> None:
        raise NotImplementedError(
            "trajectory predictor adapter 'bench2drive' is reserved but not implemented yet"
        )

    def init_modules(self, predictor: "SceneTrajectoryFlowPredictor") -> None:
        return None

    def init_parameters(self, predictor: "SceneTrajectoryFlowPredictor") -> None:
        return None

    def validate_inputs(
        self,
        predictor: "SceneTrajectoryFlowPredictor",
        inputs: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        self._raise_unimplemented()

    def encode(
        self,
        predictor: "SceneTrajectoryFlowPredictor",
        inputs: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self._raise_unimplemented()

    def decode(self, predictor: "SceneTrajectoryFlowPredictor", traj: torch.Tensor) -> torch.Tensor:
        self._raise_unimplemented()


def _build_trajectory_predictor_adapter(name: str) -> TrajectoryPredictorAdapter:
    normalized = str(name).lower()
    if normalized in ("navsim", "nuplan_navsim"):
        return NavsimTrajectoryPredictorAdapter()
    if normalized in ("bench2drive", "b2d"):
        return Bench2DriveTrajectoryPredictorAdapter()
    raise ValueError(
        "Unknown trajectory predictor adapter: "
        f"{name!r}. Supported values are 'navsim' and 'bench2drive'."
    )


class SceneTrajectoryFlowPredictor(nn.Module):
    """Joint x-pred flow predictor for future scene tokens and trajectory."""

    def __init__(
        self,
        *,
        scene_dim: int,
        num_scene_tokens: int,
        num_cameras: int,
        scene_grid_size: tuple[int, int],
        scene_history_steps: int,
        scene_future_steps: int,
        ego_status_dim: int,
        history_steps: int,
        trajectory_horizon: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        enable_trajectory_head: bool = True,
        trajectory_adapter: str = "navsim",
        traj_loss_grad_to_scene_flow: bool = False,
        scene_loss_grad_to_traj_flow: bool = False,
    ) -> None:
        super().__init__()
        self.enable_trajectory_head = bool(enable_trajectory_head)
        self.trajectory_adapter_name = str(trajectory_adapter).lower()
        self.trajectory_adapter = _build_trajectory_predictor_adapter(self.trajectory_adapter_name)
        self.traj_loss_grad_to_scene_flow = bool(traj_loss_grad_to_scene_flow)
        self.scene_loss_grad_to_traj_flow = bool(scene_loss_grad_to_traj_flow)
        self.scene_dim = int(scene_dim)
        self.num_scene_tokens = int(num_scene_tokens)
        self.num_cameras = int(num_cameras)
        self.scene_grid_h, self.scene_grid_w = int(scene_grid_size[0]), int(scene_grid_size[1])
        self.scene_history_steps = int(scene_history_steps)
        self.scene_future_steps = int(scene_future_steps)
        self.ego_status_dim = int(ego_status_dim)
        self.history_steps = int(history_steps)
        self.trajectory_horizon = int(trajectory_horizon)
        self.hidden_dim = int(hidden_dim)
        if self.enable_trajectory_head and self.trajectory_horizon <= 0:
            raise ValueError("trajectory_horizon must be > 0")
        if self.num_scene_tokens <= 0:
            raise ValueError("num_scene_tokens must be > 0")
        if self.num_cameras <= 0:
            raise ValueError("num_cameras must be > 0")
        if self.scene_grid_h <= 0 or self.scene_grid_w <= 0:
            raise ValueError("scene_grid_size must contain positive values")
        if self.scene_history_steps <= 0 or self.scene_future_steps <= 0:
            raise ValueError("scene_history_steps and scene_future_steps must be > 0")
        self.scene_tokens_per_step = self.scene_grid_h * self.scene_grid_w
        self.scene_future_tokens_per_view = self.scene_future_steps * self.scene_tokens_per_step
        if self.num_scene_tokens != self.num_cameras * self.scene_future_tokens_per_view:
            raise ValueError(
                "num_scene_tokens must equal num_cameras * scene_future_steps * tokens_per_step, "
                f"got {self.num_scene_tokens}"
            )
        if self.enable_trajectory_head and self.history_steps <= 0:
            raise ValueError("num_history_trajectory_steps must be > 0")
        if self.hidden_dim <= 0 or self.hidden_dim % int(num_heads) != 0:
            raise ValueError("flow_hidden_dim must be positive and divisible by flow_num_heads")

        self.scene_positioner = SceneInputPositioner(
            scene_dim=self.scene_dim,
            num_cameras=self.num_cameras,
            grid_size=(self.scene_grid_h, self.scene_grid_w),
            max_time_steps=self.scene_history_steps + self.scene_future_steps,
        )
        future_indices = torch.arange(self.scene_future_tokens_per_view, dtype=torch.long).view(1, -1)
        self.register_buffer("future_token_indices", future_indices, persistent=False)

        self.context_proj = nn.Linear(self.scene_dim, self.hidden_dim)
        self.noisy_scene_proj = nn.Linear(self.scene_dim, self.hidden_dim)
        self.future_condition_proj = nn.Linear(self.scene_dim, self.hidden_dim)
        self.scene_noise_condition_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.scene_out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.scene_dim),
        )
        if self.enable_trajectory_head:
            self.trajectory_adapter.init_modules(self)
        self.time_embed = ScalarTimeEmbedding(self.hidden_dim)
        self.context_type_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.scene_type_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.context_type_embed, std=0.02)
        nn.init.normal_(self.scene_type_embed, std=0.02)
        if self.enable_trajectory_head:
            self.trajectory_adapter.init_parameters(self)
        self.blocks = nn.ModuleList(
            [
                JointFlowBlock(
                    self.hidden_dim,
                    int(num_heads),
                    enable_trajectory_head=self.enable_trajectory_head,
                    traj_loss_grad_to_scene_flow=self.traj_loss_grad_to_scene_flow,
                    scene_loss_grad_to_traj_flow=self.scene_loss_grad_to_traj_flow,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        *,
        context_scene: torch.Tensor,
        context_token_indices: torch.Tensor,
        noisy_future_scene: torch.Tensor,
        future_condition_scene: torch.Tensor,
        t_cont: torch.Tensor,
        trajectory_inputs: dict[str, torch.Tensor] | None = None,
        noisy_trajectory: torch.Tensor | None = None,
        ego_status: torch.Tensor | None = None,
        history_trajectory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if context_scene.ndim != 3 or noisy_future_scene.ndim != 3 or future_condition_scene.ndim != 3:
            raise ValueError("context_scene, noisy_future_scene, and future_condition_scene must be [B,N,D]")
        bsz = context_scene.size(0)
        if (
            noisy_future_scene.shape[:2] != (bsz, self.num_scene_tokens)
            or future_condition_scene.shape[:2] != (bsz, self.num_scene_tokens)
        ):
            raise ValueError(
                "Flow predictor shape mismatch: "
                f"context={tuple(context_scene.shape)} scene={tuple(noisy_future_scene.shape)} "
                f"future_condition={tuple(future_condition_scene.shape)}"
            )
        if noisy_future_scene.size(-1) != self.scene_dim or future_condition_scene.size(-1) != self.scene_dim:
            raise ValueError(
                "Future scene tensors must have scene_dim features, got "
                f"scene={tuple(noisy_future_scene.shape)} condition={tuple(future_condition_scene.shape)}"
            )
        if self.enable_trajectory_head:
            if trajectory_inputs is None:
                trajectory_inputs = {}
                if noisy_trajectory is not None:
                    trajectory_inputs["noisy_trajectory"] = noisy_trajectory
                if ego_status is not None:
                    trajectory_inputs["ego_status"] = ego_status
                if history_trajectory is not None:
                    trajectory_inputs["history_trajectory"] = history_trajectory
            self.trajectory_adapter.validate_inputs(self, trajectory_inputs, batch_size=bsz)

        dtype = self.context_proj.weight.dtype
        context_scene = self.scene_positioner(
            context_scene.to(dtype=dtype),
            context_token_indices,
            time_offset=0,
        )
        future_indices = self.future_token_indices.to(device=noisy_future_scene.device).expand(bsz, -1)
        noisy_future_scene = self.scene_positioner(
            noisy_future_scene.to(dtype=dtype),
            future_indices,
            time_offset=self.scene_history_steps,
        )
        future_condition_scene = self.scene_positioner(
            future_condition_scene.to(dtype=dtype),
            future_indices,
            time_offset=self.scene_history_steps,
        )

        context = self.context_proj(context_scene)
        context = context + self.context_type_embed.to(dtype=dtype, device=context.device)
        time = self.time_embed(t_cont.to(dtype=dtype))
        noisy_scene = self.noisy_scene_proj(noisy_future_scene)
        future_condition = self.future_condition_proj(future_condition_scene)
        scene = self.scene_noise_condition_fusion(torch.cat([noisy_scene, future_condition], dim=-1))
        scene = scene + self.scene_type_embed.to(dtype=dtype, device=scene.device)

        traj = None
        if self.enable_trajectory_head:
            if trajectory_inputs is None:
                raise RuntimeError("trajectory inputs unexpectedly missing")
            traj = self.trajectory_adapter.encode(self, trajectory_inputs, dtype=dtype)

        for block in self.blocks:
            context, scene, traj = block(context, scene, traj, time)
        pred_scene = self.scene_out(scene)
        if not self.enable_trajectory_head:
            return pred_scene, None
        if traj is None:
            raise RuntimeError("trajectory hidden state missing")
        return pred_scene, self.trajectory_adapter.decode(self, traj)


class MultiViewCausalFutureMaskedJEPA(ModelBase):
    _VARIANT_ALIASES = {
        "vjepa2_1_vit_base_384": "vit_base",
        "vit_b": "vit_base",
        "vitb": "vit_base",
        "vjepa2_1_vit_large_384": "vit_large",
        "vit_l": "vit_large",
        "vitl": "vit_large",
        "vjepa2_1_vit_giant_384": "vit_giant_xformers",
        "vit_g": "vit_giant_xformers",
        "vitg": "vit_giant_xformers",
        "vjepa2_1_vit_gigantic_384": "vit_gigantic_xformers",
        "vit_G": "vit_gigantic_xformers",
        "vitG": "vit_gigantic_xformers",
    }

    def __init__(self, cfg: Any, full_cfg: Any | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.full_cfg = full_cfg or {}
        self.camera_names = _camera_names(cfg_get(cfg, "camera_names", cfg_get(self.full_cfg.get("data", {}) or {}, "camera_names", None)))
        self.num_cameras = len(self.camera_names)
        data_camera_names = cfg_get(cfg_get(self.full_cfg, "data", {}) or {}, "camera_names", None)
        if data_camera_names is not None:
            normalized_data_camera_names = _camera_names(data_camera_names)
            if normalized_data_camera_names != self.camera_names:
                raise ValueError(
                    "model.camera_names and data.camera_names must have the same order. "
                    f"got model={self.camera_names} data={normalized_data_camera_names}"
                )

        input_hw = _as_pair(cfg_get(cfg, "input_hw", (256, 512)), (256, 512))
        self.num_history_frames = int(cfg_get(cfg, "num_history_frames", 4))
        self.num_future_frames = int(cfg_get(cfg, "num_future_frames", 8))
        self.num_frames = self.num_history_frames + self.num_future_frames
        self.patch_size = int(cfg_get(cfg, "patch_size", 16))
        self.tubelet_size = int(cfg_get(cfg, "tubelet_size", 2))
        self.image_normalization = str(cfg_get(cfg, "image_normalization", "minus_one_to_imagenet"))
        mean = _as_float_tuple(cfg_get(cfg, "image_mean", (0.485, 0.456, 0.406)), (0.485, 0.456, 0.406))
        std = _as_float_tuple(cfg_get(cfg, "image_std", (0.229, 0.224, 0.225)), (0.229, 0.224, 0.225))
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1, 1), persistent=False)

        variant_raw = str(cfg_get(cfg, "variant", "vjepa2_1_vit_large_384"))
        variant = self._VARIANT_ALIASES.get(variant_raw, variant_raw)
        if not hasattr(vjepa_vit, variant):
            raise ValueError(f"Unknown V-JEPA ViT variant: {variant}")
        pretrained_img_size = _as_pair(cfg_get(cfg, "pretrained_input_hw", (256, 256)), (256, 256))
        common = dict(
            img_size=input_hw,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            use_rope=bool(cfg_get(cfg, "use_rope", True)),
            use_sdpa=bool(cfg_get(cfg, "use_sdpa", True)),
            use_silu=bool(cfg_get(cfg, "use_silu", False)),
            wide_silu=bool(cfg_get(cfg, "wide_silu", True)),
            use_activation_checkpointing=bool(cfg_get(cfg, "use_activation_checkpointing", False)),
            interpolate_rope=bool(cfg_get(cfg, "interpolate_rope", True)),
            modality_embedding=bool(cfg_get(cfg, "modality_embedding", True)),
            img_temporal_dim_size=cfg_get(cfg, "img_temporal_dim_size", 1),
            pretrained_img_size=pretrained_img_size,
        )
        self.encoder = getattr(vjepa_vit, variant)(**common)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)
        embed_dim = int(self.encoder.embed_dim)
        self.target_dim = embed_dim * len(self.encoder.hierarchical_layers)

        self.mask_sampler = FutureMaskSampler(
            num_history_frames=self.num_history_frames,
            num_future_frames=self.num_future_frames,
            image_size=input_hw,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
            official_mask_configs=cfg_get(cfg, "jepa_mask_configs", None),
        )
        scene_dim = int(cfg_get(cfg, "mv_scene_dim", 1024))
        self.scene_projector = SceneTokenProjector(
            target_dim=self.target_dim,
            scene_dim=scene_dim,
        )
        self.target_scene_projector = copy.deepcopy(self.scene_projector)
        self.target_scene_projector.requires_grad_(False)
        self.future_mask_token = nn.Parameter(torch.zeros(1, 1, 1, scene_dim))
        nn.init.normal_(self.future_mask_token, std=0.02)

        self.enable_trajectory_head = bool(cfg_get(cfg, "enable_trajectory_head", True))
        self.driving_condition_adapter_name = str(cfg_get(cfg, "driving_condition_adapter", "navsim")).lower()
        self.driving_condition_adapter = _build_driving_condition_adapter(self.driving_condition_adapter_name)
        data_cfg = cfg_get(self.full_cfg, "data", {}) or {}
        if self.enable_trajectory_head:
            horizon = int(cfg_get(cfg, "trajectory_horizon", cfg_get(data_cfg, "num_future_frames", 8)))
            history_steps = int(cfg_get(data_cfg, "num_history_trajectory_steps", 4))
            ego_status_dim = int(cfg_get(cfg, "ego_status_dim", 8))
        else:
            horizon = int(cfg_get(cfg, "trajectory_horizon", 0) or 0)
            history_steps = int(cfg_get(data_cfg, "num_history_trajectory_steps", 0) or 0)
            ego_status_dim = int(cfg_get(cfg, "ego_status_dim", 0) or 0)
        self.predictor = SceneTrajectoryFlowPredictor(
            scene_dim=scene_dim,
            num_scene_tokens=self.num_cameras * self.mask_sampler.num_future_tokens,
            num_cameras=self.num_cameras,
            scene_grid_size=(self.mask_sampler.grid_h, self.mask_sampler.grid_w),
            scene_history_steps=self.mask_sampler.history_steps,
            scene_future_steps=self.mask_sampler.future_steps,
            ego_status_dim=ego_status_dim,
            history_steps=history_steps,
            trajectory_horizon=horizon,
            hidden_dim=int(cfg_get(cfg, "flow_hidden_dim", 512)),
            num_layers=int(cfg_get(cfg, "flow_num_layers", 4)),
            num_heads=int(cfg_get(cfg, "flow_num_heads", 8)),
            enable_trajectory_head=self.enable_trajectory_head,
            trajectory_adapter=self.driving_condition_adapter_name,
            traj_loss_grad_to_scene_flow=bool(cfg_get(cfg, "traj_loss_grad_to_scene_flow", False)),
            scene_loss_grad_to_traj_flow=bool(cfg_get(cfg, "scene_loss_grad_to_traj_flow", False)),
        )
        self.flow_time_sampling = str(cfg_get(cfg, "flow_time_sampling", "logit_normal")).lower()
        if self.flow_time_sampling != "logit_normal":
            raise ValueError("Multi-view flow predictor currently supports flow_time_sampling=logit_normal only")
        self.flow_logit_mean = float(cfg_get(cfg, "flow_logit_mean", 0.0))
        self.flow_logit_std = float(cfg_get(cfg, "flow_logit_std", 1.0))
        if self.flow_logit_std <= 0.0:
            raise ValueError("flow_logit_std must be > 0")
        self.full_mask_prob = float(cfg_get(cfg, "full_mask_prob", 0.3))
        if self.full_mask_prob < 0.0 or self.full_mask_prob > 1.0:
            raise ValueError("full_mask_prob must satisfy 0 <= full_mask_prob <= 1")
        self.flow_num_inference_steps = int(cfg_get(cfg, "num_inference_steps", 5))
        self.flow_inference_noise_scale = float(cfg_get(cfg, "flow_inference_noise_scale", 1.0))
        inference_seed = cfg_get(cfg, "flow_inference_seed", 0)
        self.flow_inference_seed = None if inference_seed is None else int(inference_seed)
        if self.flow_num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")
        self.dynamic_collapse_topk = int(cfg_get(cfg, "dynamic_collapse_topk", 64))
        if self.dynamic_collapse_topk <= 0:
            raise ValueError("dynamic_collapse_topk must be > 0")
        if self.enable_trajectory_head:
            traj_mean = _as_float_tuple(cfg_get(cfg, "trajectory_norm_mean", None), (31.8, 1.32, 0.095))
            traj_std = _as_float_tuple(
                cfg_get(cfg, "trajectory_norm_std", None),
                (33.37, 21.0, 1.765),
            )
            if len(traj_mean) != 3 or len(traj_std) != 3:
                raise ValueError("trajectory_norm_mean and trajectory_norm_std must each contain 3 values")
            if any(v <= 0.0 for v in traj_std):
                raise ValueError("trajectory_norm_std values must be > 0")
            self.register_buffer(
                "trajectory_norm_mean",
                torch.tensor(traj_mean, dtype=torch.float32).view(1, 1, 3),
                persistent=False,
            )
            self.register_buffer(
                "trajectory_norm_std",
                torch.tensor(traj_std, dtype=torch.float32).view(1, 1, 3),
                persistent=False,
            )
        self.ema_start = float(cfg_get(cfg, "ema_start", 0.99925))
        self.ema_end = float(cfg_get(cfg, "ema_end", self.ema_start))
        self.ema_total_steps = int(cfg_get(cfg, "ema_total_steps", 0))
        self.target_scene_ema_warmup_start = float(
            cfg_get(cfg, "target_scene_ema_warmup_start", self.ema_start)
        )
        self.target_scene_ema_warmup_steps = int(cfg_get(cfg, "target_scene_ema_warmup_steps", 0))
        if self.target_scene_ema_warmup_steps < 0:
            raise ValueError("target_scene_ema_warmup_steps must be >= 0")
        self.freeze_encoder_steps = int(cfg_get(cfg, "freeze_encoder_steps", 0))
        if self.freeze_encoder_steps < 0:
            raise ValueError("freeze_encoder_steps must be >= 0")
        self._train_step: int | None = 0
        self._active_loss_weights: dict[str, float] = {}

        ckpt = cfg_get(cfg, "vjepa2_ckpt", None)
        if ckpt:
            self._load_vjepa2_checkpoint(str(ckpt))
        elif bool(cfg_get(cfg, "require_pretrained", False)):
            raise ValueError("model.vjepa2_ckpt must be set when require_pretrained=true")

        init_ckpts = _checkpoint_path_list(cfg_get(cfg, "init_from_checkpoint", None))
        if init_ckpts:
            self._load_model_init_checkpoints(init_ckpts)
        elif bool(cfg_get(cfg, "require_init_from_checkpoint", False)):
            raise ValueError(
                "model.init_from_checkpoint must be set when "
                "require_init_from_checkpoint=true"
            )

    @torch.no_grad()
    def _load_vjepa2_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        encoder_key = cfg_get(self.cfg, "encoder_checkpoint_key", None)
        target_key = cfg_get(self.cfg, "target_encoder_checkpoint_key", None)
        encoder_names = (str(encoder_key),) if encoder_key else ("target_encoder", "ema_encoder", "encoder")
        enc_state = _select_checkpoint_state(payload, encoder_names)
        target_names = (str(target_key),) if target_key else encoder_names
        target_state = _select_checkpoint_state(payload, target_names)
        copy_target_from_student = bool(
            cfg_get(self.cfg, "copy_target_encoder_from_student", target_key is None)
        )
        min_ratio = float(cfg_get(self.cfg, "min_checkpoint_load_ratio", 0.95))
        allow_partial = bool(cfg_get(self.cfg, "allow_partial_load", False))
        report_dir = cfg_get(self.full_cfg, "save_root", None)
        if not enc_state and not allow_partial:
            raise RuntimeError(f"No encoder state found in V-JEPA checkpoint: {path}")
        if enc_state:
            _load_submodule_state(
                self.encoder,
                enc_state,
                log_prefix="vjepa2.encoder",
                min_load_ratio=min_ratio,
                allow_partial=allow_partial,
                checkpoint_path=path,
                report_dir=report_dir,
            )
        if enc_state and copy_target_from_student:
            self.target_encoder.load_state_dict(self.encoder.state_dict(), strict=True)
            logger.info("Initialized EMA target encoder by copying the loaded student encoder.")
        elif target_state:
            _load_submodule_state(
                self.target_encoder,
                target_state,
                log_prefix="vjepa2.target_encoder",
                min_load_ratio=min_ratio,
                allow_partial=allow_partial,
                checkpoint_path=path,
                report_dir=report_dir,
            )
        elif enc_state:
            logger.warning("No target encoder state found; reusing encoder state for EMA target.")
            _load_submodule_state(
                self.target_encoder,
                enc_state,
                log_prefix="vjepa2.target_encoder",
                min_load_ratio=min_ratio,
                allow_partial=allow_partial,
                checkpoint_path=path,
                report_dir=report_dir,
            )

    def _load_model_init_checkpoints(self, paths: list[str]) -> None:
        if len(paths) == 1:
            self._load_model_init_checkpoint(paths[0])
            return
        for idx, path in enumerate(paths):
            self._load_model_init_checkpoint(
                path,
                log_prefix=f"model.init_from_checkpoint[{idx}]",
                allow_partial=True,
                min_load_ratio=0.0,
                require_non_empty=True,
            )

    def _load_model_init_checkpoint(
        self,
        path: str,
        *,
        log_prefix: str = "model.init_from_checkpoint",
        allow_partial: bool | None = None,
        min_load_ratio: float | None = None,
        require_non_empty: bool = False,
    ) -> None:
        state = _load_state_from_path(path)
        if allow_partial is None:
            allow_partial = bool(cfg_get(self.cfg, "init_checkpoint_allow_partial", True))
        if min_load_ratio is None:
            min_load_ratio = 0.0 if allow_partial else 1.0
        _load_submodule_state(
            self,
            state,
            log_prefix=log_prefix,
            min_load_ratio=min_load_ratio,
            allow_partial=allow_partial,
            require_non_empty=require_non_empty,
            checkpoint_path=path,
            report_dir=cfg_get(self.full_cfg, "save_root", None),
        )

    @staticmethod
    def _require_tensor(mapping: dict[str, Any], key: str) -> torch.Tensor:
        if key not in mapping:
            raise KeyError(f"Batch field {key!r} is required")
        value = mapping[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Batch field {key!r} must be a torch.Tensor, got {type(value).__name__}")
        return value

    @staticmethod
    def _to_bcthw(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,T,C,H,W], got {tuple(images.shape)}")
        return images.permute(0, 2, 1, 3, 4).contiguous()

    def _ensure_multiview(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            return images.unsqueeze(2)
        if images.ndim == 6:
            return images
        raise ValueError(f"Expected images [B,T,C,H,W] or [B,T,V,C,H,W], got {tuple(images.shape)}")

    def _make_full_clip(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        history = self._ensure_multiview(history)
        future = self._ensure_multiview(future)
        if history.shape[0] != future.shape[0] or history.shape[2:] != future.shape[2:]:
            raise ValueError(
                "history_images and future_images must share B,V,C,H,W, "
                f"got history={tuple(history.shape)} future={tuple(future.shape)}"
            )
        if history.size(1) != self.num_history_frames or future.size(1) != self.num_future_frames:
            raise ValueError(
                f"Expected history/future frames {(self.num_history_frames, self.num_future_frames)}, "
                f"got {(history.size(1), future.size(1))}"
            )
        if history.size(2) != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {history.size(2)}")
        clip = torch.cat([history, future], dim=1)
        bsz, steps, views, channels, height, width = clip.shape
        clip = clip.permute(0, 2, 1, 3, 4, 5).reshape(bsz * views, steps, channels, height, width)
        return self._normalize_images(self._to_bcthw(clip))

    def _normalize_images(self, clip: torch.Tensor) -> torch.Tensor:
        mode = self.image_normalization
        if mode in ("none", "identity"):
            return clip
        if mode == "minus_one_to_imagenet":
            clip = clip.add(1.0).mul(0.5).clamp(0.0, 1.0)
            return (clip - self.image_mean.to(dtype=clip.dtype)) / self.image_std.to(dtype=clip.dtype)
        if mode == "zero_one_to_imagenet":
            clip = clip.clamp(0.0, 1.0)
            return (clip - self.image_mean.to(dtype=clip.dtype)) / self.image_std.to(dtype=clip.dtype)
        raise ValueError(f"Unknown image_normalization: {mode}")

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(batch, dict):
            return batch
        out = dict(batch)
        visual_dtype = _module_param_dtype(self.encoder)
        visual_device = _module_param_device(self.encoder)
        predictor_dtype = _module_param_dtype(self.predictor)
        predictor_device = _module_param_device(self.predictor)
        for key in ("history_images", "future_images"):
            value = out.get(key)
            if isinstance(value, torch.Tensor):
                out[key] = _move_tensor_for_module(value, dtype=visual_dtype, device=visual_device)
        return self.driving_condition_adapter.prepare_batch(
            out,
            dtype=predictor_dtype,
            device=predictor_device,
        )

    def set_loss_weights(self, weights: dict[str, float] | None) -> None:
        self._active_loss_weights = {str(k): float(v) for k, v in (weights or {}).items()}

    def _loss_enabled(self, name: str) -> bool:
        if not self._active_loss_weights:
            return True
        return abs(float(self._active_loss_weights.get(name, 0.0))) > 0.0

    def _reshape_views(self, tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
        return tokens.reshape(batch_size, self.num_cameras, tokens.size(1), tokens.size(2))

    def _sample_training_masks(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> tuple[FutureMaskBatch, torch.Tensor]:
        if self.full_mask_prob <= 0.0:
            return (
                self.mask_sampler.sample(batch_size, device=device),
                torch.zeros(batch_size, device=device, dtype=torch.bool),
            )
        if self.full_mask_prob >= 1.0:
            return (
                self.mask_sampler.full_future_mask(batch_size, device=device),
                torch.ones(batch_size, device=device, dtype=torch.bool),
            )

        use_full_batch = bool((torch.rand((), device=device) < self.full_mask_prob).item())
        if use_full_batch:
            return (
                self.mask_sampler.full_future_mask(batch_size, device=device),
                torch.ones(batch_size, device=device, dtype=torch.bool),
            )
        return (
            self.mask_sampler.sample(batch_size, device=device),
            torch.zeros(batch_size, device=device, dtype=torch.bool),
        )

    @staticmethod
    def _mask_group_rows(value: torch.Tensor, group_i: int, idx: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2:
            rows = value[group_i]
        elif value.ndim == 1:
            rows = value
        else:
            raise ValueError(f"Unsupported mask metadata shape: {tuple(value.shape)}")
        if rows.size(0) == idx.numel():
            return rows
        return rows.index_select(0, idx)

    @staticmethod
    def _flatten_mask_metadata(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return value.to(device=device, dtype=dtype).reshape(-1)

    def _mask_context_stats(
        self,
        mask_batch: FutureMaskBatch,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_rows = 0
        weighted_context_tokens = 0.0
        for idx, masks_x in zip(mask_batch.indices, mask_batch.masks_x):
            rows = int(idx.numel())
            total_rows += rows
            weighted_context_tokens += float(masks_x.size(1)) * float(rows)
        if total_rows <= 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            return zero, zero
        visible_tokens = torch.as_tensor(
            weighted_context_tokens / float(total_rows),
            device=device,
            dtype=dtype,
        )
        context_tokens = torch.as_tensor(
            float(self.mask_sampler.num_history_tokens),
            device=device,
            dtype=dtype,
        )
        visible_future_frac = (
            (visible_tokens - float(self.mask_sampler.num_history_tokens))
            / float(self.mask_sampler.num_future_tokens)
        ).clamp(0.0, 1.0)
        return context_tokens, visible_future_frac

    def _teacher_targets(self, clip: torch.Tensor, batch_size: int) -> torch.Tensor:
        with torch.no_grad():
            h = self.target_encoder(clip, training=True)
            h = _chunk_layer_norm(h, self.encoder.embed_dim)
            return self._reshape_views(h, batch_size)

    def _encoder_frozen_for_step(self) -> bool:
        return bool(
            self.training
            and self.freeze_encoder_steps > 0
            and self._train_step < self.freeze_encoder_steps
        )

    def _encode_context(self, clip: torch.Tensor, masks_x: torch.Tensor, batch_size: int) -> torch.Tensor:
        masks = masks_x.repeat_interleave(self.num_cameras, dim=0)
        if self._encoder_frozen_for_step():
            with torch.no_grad():
                h = self.encoder(clip, masks=masks, training=True)
        else:
            h = self.encoder(clip, masks=masks, training=True)
        return self._reshape_views(h, batch_size)

    def _history_indices_from_masks(self, masks_x: torch.Tensor, batch_size: int) -> torch.Tensor:
        num_history_tokens = self.mask_sampler.num_history_tokens
        if masks_x.ndim != 2 or masks_x.size(0) != batch_size or masks_x.size(1) < num_history_tokens:
            raise ValueError(
                "masks_x must be [B,N] and contain all history tokens first, "
                f"got {tuple(masks_x.shape)} for B={batch_size}"
            )
        history_indices = masks_x[:, :num_history_tokens]
        expected_history = torch.arange(num_history_tokens, device=masks_x.device, dtype=masks_x.dtype)
        if not torch.equal(history_indices, expected_history.view(1, -1).expand_as(history_indices)):
            raise ValueError("masks_x must keep all history tokens first for multi-view JEPA flow")
        return history_indices

    def _future_mask_condition(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = self.future_mask_token.to(dtype=dtype, device=device).expand(
            batch_size,
            self.num_cameras,
            self.mask_sampler.num_future_tokens,
            -1,
        )
        return mask.reshape(batch_size, self.predictor.num_scene_tokens, -1)

    def _future_condition_from_visible(
        self,
        visible_future: torch.Tensor,
        visible_future_indices: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        future_condition = self._future_mask_condition(batch_size, device=device, dtype=dtype)
        if visible_future.size(2) == 0:
            return future_condition
        future_local_indices = visible_future_indices.to(device=device, dtype=torch.long) - int(
            self.mask_sampler.num_history_tokens
        )
        if (
            int(future_local_indices.min().item()) < 0
            or int(future_local_indices.max().item()) >= self.mask_sampler.num_future_tokens
        ):
            raise ValueError("Visible future token indices must lie inside the future token range")
        visible_scene = self.scene_projector(visible_future)
        future_condition = future_condition.reshape(
            batch_size,
            self.num_cameras,
            self.mask_sampler.num_future_tokens,
            -1,
        )
        scatter_index = future_local_indices[:, None, :, None].expand(
            -1,
            self.num_cameras,
            -1,
            future_condition.size(-1),
        )
        future_condition = future_condition.scatter(2, scatter_index, visible_scene)
        return future_condition.reshape(batch_size, self.predictor.num_scene_tokens, -1)

    def _encode_scene_inputs(
        self,
        clip: torch.Tensor,
        masks_x: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_history_tokens = self.mask_sampler.num_history_tokens
        history_indices = self._history_indices_from_masks(masks_x, batch_size)
        visible = self._encode_context(clip, masks_x, batch_size)
        history_tokens = visible[:, :, :num_history_tokens]
        context_scene = self.scene_projector(history_tokens).reshape(batch_size, -1, self.scene_projector.scene_dim)
        future_condition = self._future_condition_from_visible(
            visible[:, :, num_history_tokens:],
            masks_x[:, num_history_tokens:],
            batch_size=batch_size,
            dtype=context_scene.dtype,
            device=context_scene.device,
        )
        return context_scene, future_condition, history_indices

    def _encode_scene_context(self, clip: torch.Tensor, masks_x: torch.Tensor, batch_size: int) -> torch.Tensor:
        history_indices = self._history_indices_from_masks(masks_x, batch_size)
        visible = self._encode_context(clip, masks_x, batch_size)
        history_tokens = visible[:, :, : self.mask_sampler.num_history_tokens]
        return self.scene_projector(history_tokens).reshape(batch_size, -1, self.scene_projector.scene_dim)

    def _future_teacher_tokens(self, targets: torch.Tensor) -> torch.Tensor:
        start = self.mask_sampler.num_history_tokens
        end = start + self.mask_sampler.num_future_tokens
        future = targets[:, :, start:end]
        if future.size(2) != self.mask_sampler.num_future_tokens:
            raise RuntimeError(
                "Teacher target token layout is incompatible with future token slice: "
                f"got {future.size(2)} / {self.mask_sampler.num_future_tokens}"
            )
        return future

    def _target_future_scene(self, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            future = self.target_scene_projector(self._future_teacher_tokens(targets))
            return future.reshape(targets.size(0), -1, self.scene_projector.scene_dim)

    def _sample_flow_time(self, bsz: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        logits = torch.randn(bsz, device=device, dtype=dtype)
        logits = logits * self.flow_logit_std + self.flow_logit_mean
        return torch.sigmoid(logits).clamp_(1e-4, 1.0 - 1e-4)

    @staticmethod
    def _flow_interpolate(clean: torch.Tensor, t_cont: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean)
        view_shape = (clean.size(0),) + (1,) * (clean.ndim - 1)
        t = t_cont.view(view_shape).to(dtype=clean.dtype, device=clean.device)
        return (1.0 - t) * noise + t * clean, noise

    def _normalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        if not self.enable_trajectory_head:
            raise RuntimeError("trajectory normalization requires enable_trajectory_head=true")
        return self.driving_condition_adapter.normalize_trajectory(self, trajectory)

    def _denormalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        if not self.enable_trajectory_head:
            raise RuntimeError("trajectory denormalization requires enable_trajectory_head=true")
        return self.driving_condition_adapter.denormalize_trajectory(self, trajectory)

    def _zero_flow_conditions(
        self,
        bsz: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DrivingConditionBatch:
        if not self.enable_trajectory_head:
            raise RuntimeError("trajectory flow conditions require enable_trajectory_head=true")
        return self.driving_condition_adapter.zero_conditions(
            bsz,
            device=device,
            dtype=dtype,
            trajectory_horizon=self.predictor.trajectory_horizon,
            history_steps=self.predictor.history_steps,
            ego_status_dim=self.predictor.ego_status_dim,
        )

    def _flow_predict_groups(
        self,
        clip: torch.Tensor,
        mask_batch: FutureMaskBatch,
        target_scene: torch.Tensor,
        *,
        driving_conditions: DrivingConditionBatch | None,
        compute_traj_loss: bool,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for group_i, (idx, masks_x, _masks_y) in enumerate(
            zip(mask_batch.indices, mask_batch.masks_x, mask_batch.masks_y)
        ):
            group_masks_x = masks_x if masks_x.size(0) == idx.numel() else masks_x.index_select(0, idx)
            group_full = self._mask_group_rows(mask_batch.full_mask, group_i, idx).to(
                device=clip.device,
                dtype=torch.bool,
            )
            group_ratio = self._mask_group_rows(mask_batch.mask_ratios, group_i, idx).to(
                device=clip.device,
                dtype=torch.float32,
            )
            view_idx = (
                idx.unsqueeze(1) * self.num_cameras
                + torch.arange(self.num_cameras, device=clip.device).unsqueeze(0)
            ).reshape(-1)
            sub_clip = clip.index_select(0, view_idx)
            context_scene, future_condition_scene, context_token_indices = self._encode_scene_inputs(
                sub_clip,
                group_masks_x,
                idx.numel(),
            )
            clean_scene = target_scene.index_select(0, idx).to(dtype=context_scene.dtype)
            t_cont = self._sample_flow_time(idx.numel(), device=clip.device, dtype=context_scene.dtype)
            noisy_scene, _ = self._flow_interpolate(clean_scene, t_cont)

            if self.enable_trajectory_head:
                conditions = driving_conditions or DrivingConditionBatch()
                group_conditions = conditions.select(idx)
                if not self.driving_condition_adapter.has_training_conditions(group_conditions):
                    group_conditions = self._zero_flow_conditions(
                        idx.numel(),
                        device=clip.device,
                        dtype=context_scene.dtype,
                    )
                    compute_traj = False
                else:
                    compute_traj = compute_traj_loss
                traj_flow = self.driving_condition_adapter.prepare_training_flow(
                    self,
                    group_conditions,
                    t_cont=t_cont,
                    dtype=context_scene.dtype,
                    device=clip.device,
                )
                pred_scene, pred_traj = self.predictor(
                    context_scene=context_scene,
                    context_token_indices=context_token_indices,
                    noisy_future_scene=noisy_scene,
                    future_condition_scene=future_condition_scene,
                    t_cont=t_cont,
                    trajectory_inputs=traj_flow.predictor_inputs,
                )
                if pred_traj is None:
                    raise RuntimeError("trajectory predictor returned no trajectory output")
                output = {
                    "idx": idx,
                    "pred_scene": pred_scene,
                    "target_scene": clean_scene,
                    "pred_traj": pred_traj,
                    "target_traj": traj_flow.clean_target,
                    "pred_traj_raw": self.driving_condition_adapter.denormalize_trajectory(
                        self,
                        pred_traj,
                    ),
                    "target_traj_raw": traj_flow.raw_target,
                    "t_cont": t_cont,
                    "compute_traj_loss": compute_traj,
                    "full_mask": group_full,
                    "mask_ratio": group_ratio,
                }
            else:
                pred_scene, _pred_traj = self.predictor(
                    context_scene=context_scene,
                    context_token_indices=context_token_indices,
                    noisy_future_scene=noisy_scene,
                    future_condition_scene=future_condition_scene,
                    t_cont=t_cont,
                )
                output = {
                    "idx": idx,
                    "pred_scene": pred_scene,
                    "target_scene": clean_scene,
                    "t_cont": t_cont,
                    "full_mask": group_full,
                    "mask_ratio": group_ratio,
                }
            outputs.append(output)
        return outputs

    def _scene_flow_loss(
        self,
        preds: list[dict[str, Any]],
        zero_ref: torch.Tensor,
        *,
        full_mask: bool | None = None,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for pred in preds:
            sample_loss = F.mse_loss(
                pred["pred_scene"].float(),
                pred["target_scene"].float(),
                reduction="none",
            ).flatten(1).mean(dim=1)
            if full_mask is not None:
                selector = pred["full_mask"].to(device=sample_loss.device, dtype=torch.bool)
                selector = selector if full_mask else ~selector
                if not bool(selector.any().item()):
                    continue
                sample_loss = sample_loss[selector]
            losses.append(sample_loss)
        if not losses:
            return zero_ref.sum() * 0.0
        return torch.cat(losses, dim=0).mean()

    def _trajectory_flow_loss(self, preds: list[dict[str, Any]], zero_ref: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for pred in preds:
            if bool(pred["compute_traj_loss"]):
                losses.append(
                    F.mse_loss(
                        pred["pred_traj"].float(),
                        pred["target_traj"].float(),
                        reduction="none",
                    ).flatten(1).mean(dim=1)
                )
        if not losses:
            return zero_ref.sum() * 0.0
        return torch.cat(losses, dim=0).mean()

    def _trajectory_velocity_loss(self, preds: list[dict[str, Any]], zero_ref: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for pred in preds:
            if not bool(pred["compute_traj_loss"]):
                continue
            pred_raw = pred["pred_traj_raw"].float()
            target_raw = pred["target_traj_raw"].float()
            if pred_raw.size(1) < 2:
                continue
            pred_delta = pred_raw[:, 1:, :2] - pred_raw[:, :-1, :2]
            target_delta = target_raw[:, 1:, :2] - target_raw[:, :-1, :2]
            losses.append(
                F.smooth_l1_loss(
                    pred_delta,
                    target_delta,
                    reduction="none",
                )
                .flatten(1)
                .mean(dim=1)
            )
        if not losses:
            return zero_ref.sum() * 0.0
        return torch.cat(losses, dim=0).mean()

    def _trajectory_smooth_loss(self, preds: list[dict[str, Any]], zero_ref: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for pred in preds:
            if not bool(pred["compute_traj_loss"]):
                continue
            pred_raw = pred["pred_traj_raw"].float()
            target_raw = pred["target_traj_raw"].float()
            if pred_raw.size(1) < 3:
                continue
            pred_acc = (
                pred_raw[:, 2:, :2]
                - 2.0 * pred_raw[:, 1:-1, :2]
                + pred_raw[:, :-2, :2]
            )
            target_acc = (
                target_raw[:, 2:, :2]
                - 2.0 * target_raw[:, 1:-1, :2]
                + target_raw[:, :-2, :2]
            )
            losses.append(
                F.smooth_l1_loss(
                    pred_acc,
                    target_acc,
                    reduction="none",
                )
                .flatten(1)
                .mean(dim=1)
            )
        if not losses:
            return zero_ref.sum() * 0.0
        return torch.cat(losses, dim=0).mean()

    def _trajectory_progress_loss(self, preds: list[dict[str, Any]], zero_ref: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for pred in preds:
            if not bool(pred["compute_traj_loss"]):
                continue
            pred_raw = pred["pred_traj_raw"].float()
            target_raw = pred["target_traj_raw"].float()
            progress_gap = (target_raw[..., 0] - pred_raw[..., 0]).clamp_min(0.0)
            losses.append(progress_gap.flatten(1).mean(dim=1))
        if not losses:
            return zero_ref.sum() * 0.0
        return torch.cat(losses, dim=0).mean()

    def _trajectory_logs(
        self,
        preds: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        return self.driving_condition_adapter.trajectory_logs(preds)

    @staticmethod
    def _flow_time_stats(
        preds: list[dict[str, Any]],
        zero_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values: list[torch.Tensor] = []
        for pred in preds:
            if "t_cont" not in pred:
                continue
            t_cont = pred["t_cont"].detach().float()
            values.append(t_cont.reshape(-1))
        if not values:
            zero = zero_ref.detach().float().new_zeros(())
            return zero, zero
        all_t = torch.cat(values, dim=0)
        return all_t.mean(), all_t.std(unbiased=False)

    def _scene_token_collapse_metrics(
        self,
        preds: list[dict[str, Any]],
        zero_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = zero_ref.detach().float().new_zeros(())
        pred_cos_sum = zero.clone()
        target_cos_sum = zero.clone()
        pair_count_sum = zero.clone()
        pred_delta_sum = zero.clone()
        target_delta_sum = zero.clone()
        delta_count_sum = zero.clone()
        future_steps = self.mask_sampler.future_steps
        tokens_per_step = self.mask_sampler.tokens_per_step
        view_spatial_tokens = self.num_cameras * tokens_per_step
        expected_tokens = self.num_cameras * future_steps * tokens_per_step
        dynamic_k = min(self.dynamic_collapse_topk, view_spatial_tokens)
        if future_steps < 2 or dynamic_k <= 0:
            return zero, zero, zero, zero, zero

        for pred in preds:
            pred_scene = pred["pred_scene"].detach().float()
            target_scene = pred["target_scene"].detach().float()
            if pred_scene.ndim != 3 or target_scene.ndim != 3 or pred_scene.size(0) == 0 or pred_scene.size(1) < 2:
                continue
            if pred_scene.shape != target_scene.shape:
                continue
            if pred_scene.size(1) != expected_tokens:
                continue

            bsz, _, dim = pred_scene.shape
            pred_by_time = pred_scene.reshape(
                bsz,
                self.num_cameras,
                future_steps,
                tokens_per_step,
                dim,
            ).permute(0, 1, 3, 2, 4).reshape(bsz, view_spatial_tokens, future_steps, dim)
            target_by_time = target_scene.reshape(
                bsz,
                self.num_cameras,
                future_steps,
                tokens_per_step,
                dim,
            ).permute(0, 1, 3, 2, 4).reshape(bsz, view_spatial_tokens, future_steps, dim)

            teacher_delta = torch.linalg.vector_norm(
                target_by_time[:, :, 1:] - target_by_time[:, :, :-1],
                dim=-1,
            )
            dynamic_score = teacher_delta.mean(dim=-1)
            topk = torch.topk(dynamic_score, k=dynamic_k, dim=1).indices
            gather_index = topk[:, :, None, None].expand(-1, -1, future_steps, dim)
            pred_token = pred_by_time.gather(1, gather_index)
            target_token = target_by_time.gather(1, gather_index)

            pred_delta = torch.linalg.vector_norm(pred_token[:, :, 1:] - pred_token[:, :, :-1], dim=-1)
            target_delta = torch.linalg.vector_norm(target_token[:, :, 1:] - target_token[:, :, :-1], dim=-1)
            delta_count = pred_delta.new_tensor(pred_delta.numel())
            pred_delta_sum = pred_delta_sum + pred_delta.sum()
            target_delta_sum = target_delta_sum + target_delta.sum()
            delta_count_sum = delta_count_sum + delta_count

            pred_norm = F.normalize(pred_token, dim=-1, eps=1e-6)
            target_norm = F.normalize(target_token, dim=-1, eps=1e-6)
            pred_sim = torch.matmul(pred_norm, pred_norm.transpose(-1, -2))
            target_sim = torch.matmul(target_norm, target_norm.transpose(-1, -2))
            offdiag = ~torch.eye(future_steps, dtype=torch.bool, device=pred_scene.device)
            weight = offdiag.view(1, 1, future_steps, future_steps).to(dtype=pred_scene.dtype)
            pair_count = pred_scene.new_tensor(bsz * dynamic_k * future_steps * (future_steps - 1))
            pred_cos_sum = pred_cos_sum + (pred_sim * weight).sum()
            target_cos_sum = target_cos_sum + (target_sim * weight).sum()
            pair_count_sum = pair_count_sum + pair_count

        if float(pair_count_sum.item()) <= 0.0:
            return zero, zero, zero, zero, zero

        pred_cos = pred_cos_sum / pair_count_sum
        target_cos = target_cos_sum / pair_count_sum
        delta_ratio = zero.clone()
        teacher_delta_norm = zero.clone()
        if float(delta_count_sum.item()) > 0.0:
            pred_delta_norm = pred_delta_sum / delta_count_sum
            teacher_delta_norm = target_delta_sum / delta_count_sum
            delta_ratio = pred_delta_norm / teacher_delta_norm.clamp_min(1e-6)
        return pred_cos, pred_cos - target_cos, target_cos, delta_ratio, teacher_delta_norm

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        history = self._require_tensor(batch, "history_images")
        future = self._require_tensor(batch, "future_images")
        clip = self._make_full_clip(history, future)
        bsz = history.size(0)

        targets = self._teacher_targets(clip, bsz)
        target_scene = self._target_future_scene(targets)

        compute_scene_loss = self._loss_enabled("loss_jepa_pred")
        if not self.enable_trajectory_head and self._active_loss_weights.get("loss_traj", 0.0) != 0.0:
            raise ValueError("loss_traj requires model.enable_trajectory_head=true")
        compute_traj_loss = self.enable_trajectory_head and self._loss_enabled("loss_traj")
        masks, sample_full_rows = self._sample_training_masks(bsz, device=clip.device)
        driving_conditions = (
            self.driving_condition_adapter.training_conditions(
                batch,
                require_targets=compute_traj_loss,
            )
            if self.enable_trajectory_head
            else None
        )

        preds = self._flow_predict_groups(
            clip,
            masks,
            target_scene,
            driving_conditions=driving_conditions,
            compute_traj_loss=compute_traj_loss,
        )
        loss_jepa_pred = self._scene_flow_loss(preds, target_scene)
        (
            active_scene_token_collapse_cos,
            active_scene_token_collapse_gap,
            active_scene_token_teacher_cos,
            active_scene_token_delta_ratio,
            active_scene_token_teacher_delta_norm,
        ) = self._scene_token_collapse_metrics(preds, target_scene)
        mask_ratios = self._flatten_mask_metadata(
            masks.mask_ratios,
            device=clip.device,
            dtype=torch.float32,
        )
        zero_loss = target_scene.float().sum() * 0.0
        losses = {
            "loss_jepa_pred": loss_jepa_pred if compute_scene_loss else zero_loss,
        }
        flow_t_mean, flow_t_std = self._flow_time_stats(preds, target_scene)
        active_context_tokens, active_visible_future_frac = self._mask_context_stats(
            masks,
            device=clip.device,
            dtype=torch.float32,
        )
        other_log = {
            "active_mask_ratio_mean": (
                mask_ratios.mean().detach() if mask_ratios.numel() > 0 else zero_loss.detach()
            ),
            "active_mask_full_frac": (
                sample_full_rows.float().mean().detach() if sample_full_rows.numel() > 0 else zero_loss.detach()
            ),
            "active_context_tokens": active_context_tokens.detach(),
            "active_visible_future_frac": active_visible_future_frac.detach(),
            "encoder_frozen": torch.as_tensor(
                float(self._encoder_frozen_for_step()),
                device=clip.device,
                dtype=torch.float32,
            ),
            "target_scene_ema_momentum": torch.as_tensor(
                self._target_scene_ema_momentum(self._train_step),
                device=clip.device,
                dtype=torch.float32,
            ),
            "active_scene_xpred_mse": loss_jepa_pred.detach(),
            "active_flow_t_mean": flow_t_mean,
            "active_flow_t_std": flow_t_std,
            "active_scene_token_collapse_cos": active_scene_token_collapse_cos,
            "active_scene_token_collapse_gap": active_scene_token_collapse_gap,
            "active_scene_token_teacher_cos": active_scene_token_teacher_cos,
            "active_scene_token_delta_ratio": active_scene_token_delta_ratio,
            "active_scene_token_teacher_delta_norm": active_scene_token_teacher_delta_norm,
        }

        if compute_traj_loss:
            loss_traj = self._trajectory_flow_loss(preds, target_scene)
            losses["loss_traj"] = loss_traj
            other_log["active_traj_xpred_mse"] = loss_traj.detach()
            aux_loss_fns = {
                "loss_traj_velocity": self._trajectory_velocity_loss,
                "loss_traj_smooth": self._trajectory_smooth_loss,
                "loss_traj_progress": self._trajectory_progress_loss,
            }
            for loss_name, loss_fn in aux_loss_fns.items():
                if self._loss_enabled(loss_name):
                    aux_loss = loss_fn(preds, target_scene)
                    losses[loss_name] = aux_loss
                    other_log[f"active_{loss_name}"] = aux_loss.detach()
                elif loss_name in self._active_loss_weights:
                    losses[loss_name] = zero_loss
            other_log.update(self._trajectory_logs(preds))
        elif self.enable_trajectory_head:
            losses["loss_traj"] = zero_loss
            for loss_name in ("loss_traj_velocity", "loss_traj_smooth", "loss_traj_progress"):
                if loss_name in self._active_loss_weights:
                    losses[loss_name] = zero_loss

        return {"loss": losses, "other_log": other_log}

    def _make_inference_generator(
        self,
        device: torch.device,
    ) -> tuple[torch.Generator | None, torch.device | None]:
        if self.flow_inference_seed is None:
            return None, None
        try:
            generator = torch.Generator(device=device)
            sample_device: torch.device | None = device
        except (TypeError, RuntimeError):
            generator = torch.Generator()
            sample_device = torch.device("cpu")
        generator.manual_seed(self.flow_inference_seed)
        return generator, sample_device

    @staticmethod
    def _randn_for_inference(
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        sample_device: torch.device | None,
    ) -> torch.Tensor:
        if generator is None:
            return torch.randn(shape, device=device, dtype=dtype)
        sample_on = sample_device or device
        sample_dtype = dtype if sample_on == device else torch.float32
        noise = torch.randn(shape, device=sample_on, dtype=sample_dtype, generator=generator)
        return noise.to(device=device, dtype=dtype)

    @torch.no_grad()
    def predict_trajectory(self, features: dict[str, Any]) -> torch.Tensor:
        if not self.enable_trajectory_head:
            raise RuntimeError("predict_trajectory requires enable_trajectory_head=true")
        was_training = self.training
        self.eval()
        try:
            features = self.prepare_batch(features)
            history = self._require_tensor(features, "history_images")
            driving_conditions = self.driving_condition_adapter.inference_conditions(features)
            if not driving_conditions.predictor_inputs:
                raise RuntimeError("driving condition adapter did not provide inference inputs")
            history = self._ensure_multiview(history)
            bsz, _, views, channels, height, width = history.shape
            future = torch.zeros(
                bsz,
                self.num_future_frames,
                views,
                channels,
                height,
                width,
                device=history.device,
                dtype=history.dtype,
            )
            clip = self._make_full_clip(history, future)
            masks = self.mask_sampler.full_future_mask(bsz, device=clip.device)
            if len(masks.indices) != 1:
                raise RuntimeError("full_future_mask must produce one mask group for inference")
            idx = masks.indices[0]
            masks_x = masks.masks_x[0]
            view_idx = (
                idx.unsqueeze(1) * self.num_cameras
                + torch.arange(self.num_cameras, device=clip.device).unsqueeze(0)
            ).reshape(-1)
            context_scene = self._encode_scene_context(
                clip.index_select(0, view_idx),
                masks_x.index_select(0, idx),
                idx.numel(),
            )
            context_token_indices = self._history_indices_from_masks(masks_x.index_select(0, idx), idx.numel())
            future_condition_scene = self._future_mask_condition(
                bsz,
                device=clip.device,
                dtype=context_scene.dtype,
            )
            generator, sample_device = self._make_inference_generator(clip.device)
            scene = self._randn_for_inference(
                (
                    bsz,
                    self.predictor.num_scene_tokens,
                    self.scene_projector.scene_dim,
                ),
                device=clip.device,
                dtype=context_scene.dtype,
                generator=generator,
                sample_device=sample_device,
            ) * self.flow_inference_noise_scale
            group_conditions = driving_conditions.select(idx)
            traj = self.driving_condition_adapter.initial_inference_trajectory(
                self,
                bsz,
                device=clip.device,
                dtype=context_scene.dtype,
                generator=generator,
                sample_device=sample_device,
            )
            trajectory_inputs = self.driving_condition_adapter.prepare_inference_inputs(
                self,
                group_conditions,
                dtype=context_scene.dtype,
                device=clip.device,
            )
            steps = max(self.flow_num_inference_steps, 1)
            dt = 1.0 / float(steps)
            for step in range(steps):
                t_value = min(float(step) / float(steps), 1.0 - 1e-4)
                t_cont = torch.full((bsz,), t_value, device=clip.device, dtype=context_scene.dtype)
                trajectory_inputs["noisy_trajectory"] = traj
                pred_scene, pred_traj = self.predictor(
                    context_scene=context_scene,
                    context_token_indices=context_token_indices,
                    noisy_future_scene=scene,
                    future_condition_scene=future_condition_scene,
                    t_cont=t_cont,
                    trajectory_inputs=trajectory_inputs,
                )
                denom = (1.0 - t_cont).clamp_min(1e-3)
                scene = scene + dt * (pred_scene - scene) / denom.view(-1, 1, 1)
                traj = traj + dt * (pred_traj - traj) / denom.view(-1, 1, 1)
            return self.driving_condition_adapter.inference_output(self, traj)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict_future_scene_for_visualization(
        self,
        features: dict[str, Any],
        *,
        include_teacher: bool = True,
    ) -> dict[str, Any]:
        """Return inference-time future scene tokens and optional teacher targets.

        The prediction path intentionally uses only history images plus zero dummy
        future frames, matching deployment-time ``predict_trajectory``.  If
        ``include_teacher`` is true, teacher targets are computed only after the
        inference pass and are returned for visualization/reference.
        """
        was_training = self.training
        self.eval()
        try:
            features = self.prepare_batch(features)
            history = self._require_tensor(features, "history_images")
            history = self._ensure_multiview(history)
            bsz, _, views, channels, height, width = history.shape
            future_zeros = torch.zeros(
                bsz,
                self.num_future_frames,
                views,
                channels,
                height,
                width,
                device=history.device,
                dtype=history.dtype,
            )
            clip = self._make_full_clip(history, future_zeros)
            masks = self.mask_sampler.full_future_mask(bsz, device=clip.device)
            if len(masks.indices) != 1:
                raise RuntimeError("full_future_mask must produce one mask group for inference")
            idx = masks.indices[0]
            masks_x = masks.masks_x[0].index_select(0, idx)
            view_idx = (
                idx.unsqueeze(1) * self.num_cameras
                + torch.arange(self.num_cameras, device=clip.device).unsqueeze(0)
            ).reshape(-1)
            context_scene = self._encode_scene_context(
                clip.index_select(0, view_idx),
                masks_x,
                idx.numel(),
            )
            context_token_indices = self._history_indices_from_masks(masks_x, idx.numel())
            future_condition_scene = self._future_mask_condition(
                bsz,
                device=clip.device,
                dtype=context_scene.dtype,
            )
            generator, sample_device = self._make_inference_generator(clip.device)
            scene = self._randn_for_inference(
                (
                    bsz,
                    self.predictor.num_scene_tokens,
                    self.scene_projector.scene_dim,
                ),
                device=clip.device,
                dtype=context_scene.dtype,
                generator=generator,
                sample_device=sample_device,
            ) * self.flow_inference_noise_scale

            trajectory_inputs: dict[str, torch.Tensor] | None = None
            traj: torch.Tensor | None = None
            if self.enable_trajectory_head:
                driving_conditions = self.driving_condition_adapter.inference_conditions(features)
                if not driving_conditions.predictor_inputs:
                    raise RuntimeError("driving condition adapter did not provide inference inputs")
                group_conditions = driving_conditions.select(idx)
                traj = self.driving_condition_adapter.initial_inference_trajectory(
                    self,
                    bsz,
                    device=clip.device,
                    dtype=context_scene.dtype,
                    generator=generator,
                    sample_device=sample_device,
                )
                trajectory_inputs = self.driving_condition_adapter.prepare_inference_inputs(
                    self,
                    group_conditions,
                    dtype=context_scene.dtype,
                    device=clip.device,
                )

            steps = max(self.flow_num_inference_steps, 1)
            dt = 1.0 / float(steps)
            last_xpred_scene: torch.Tensor | None = None
            last_xpred_traj: torch.Tensor | None = None
            for step in range(steps):
                t_value = min(float(step) / float(steps), 1.0 - 1e-4)
                t_cont = torch.full((bsz,), t_value, device=clip.device, dtype=context_scene.dtype)
                if self.enable_trajectory_head:
                    if trajectory_inputs is None or traj is None:
                        raise RuntimeError("trajectory inference state is missing")
                    trajectory_inputs["noisy_trajectory"] = traj
                    pred_scene, pred_traj = self.predictor(
                        context_scene=context_scene,
                        context_token_indices=context_token_indices,
                        noisy_future_scene=scene,
                        future_condition_scene=future_condition_scene,
                        t_cont=t_cont,
                        trajectory_inputs=trajectory_inputs,
                    )
                    if pred_traj is None:
                        raise RuntimeError("trajectory predictor returned no trajectory output")
                    denom = (1.0 - t_cont).clamp_min(1e-3)
                    scene = scene + dt * (pred_scene - scene) / denom.view(-1, 1, 1)
                    traj = traj + dt * (pred_traj - traj) / denom.view(-1, 1, 1)
                    last_xpred_traj = pred_traj
                else:
                    pred_scene, _pred_traj = self.predictor(
                        context_scene=context_scene,
                        context_token_indices=context_token_indices,
                        noisy_future_scene=scene,
                        future_condition_scene=future_condition_scene,
                        t_cont=t_cont,
                    )
                    denom = (1.0 - t_cont).clamp_min(1e-3)
                    scene = scene + dt * (pred_scene - scene) / denom.view(-1, 1, 1)
                last_xpred_scene = pred_scene

            out: dict[str, Any] = {
                "pred_scene": scene.detach(),
                "last_xpred_scene": None if last_xpred_scene is None else last_xpred_scene.detach(),
                "future_condition_scene": future_condition_scene.detach(),
                "context_scene": context_scene.detach(),
                "num_cameras": self.num_cameras,
                "camera_names": list(self.camera_names),
                "future_steps": self.mask_sampler.future_steps,
                "tokens_per_step": self.mask_sampler.tokens_per_step,
                "grid_h": self.mask_sampler.grid_h,
                "grid_w": self.mask_sampler.grid_w,
                "tubelet_size": self.tubelet_size,
                "num_future_frames": self.num_future_frames,
            }
            if traj is not None:
                out["pred_traj"] = self.driving_condition_adapter.inference_output(self, traj.detach())
            if last_xpred_traj is not None:
                out["last_xpred_traj"] = self.driving_condition_adapter.inference_output(
                    self,
                    last_xpred_traj.detach(),
                )

            if include_teacher:
                future = self._require_tensor(features, "future_images")
                target_clip = self._make_full_clip(history, future)
                targets = self._teacher_targets(target_clip, bsz)
                out["target_scene"] = self._target_future_scene(targets).detach()
            return out
        finally:
            self.train(was_training)

    def _ema_momentum(self, global_step: int | None) -> float:
        if self.ema_total_steps <= 0 or global_step is None:
            return self.ema_end
        progress = min(max(float(global_step) / float(self.ema_total_steps), 0.0), 1.0)
        return self.ema_start + progress * (self.ema_end - self.ema_start)

    def _target_scene_ema_momentum(self, global_step: int | None) -> float:
        if (
            self.target_scene_ema_warmup_steps <= 0
            or global_step is None
            or global_step >= self.target_scene_ema_warmup_steps
        ):
            return self._ema_momentum(global_step)
        progress = min(max(float(global_step) / float(self.target_scene_ema_warmup_steps), 0.0), 1.0)
        warmup_end = self._ema_momentum(self.target_scene_ema_warmup_steps)
        return self.target_scene_ema_warmup_start + progress * (
            warmup_end - self.target_scene_ema_warmup_start
        )

    @torch.no_grad()
    def set_train_step(self, global_step: int | None) -> None:
        self._train_step = 0 if global_step is None else int(global_step)

    @torch.no_grad()
    def after_optimizer_step(self, global_step: int | None = None) -> None:
        self.set_train_step(global_step)
        momentum = self._ema_momentum(global_step)
        student_params = list(self.encoder.parameters())
        target_params = list(self.target_encoder.parameters())
        torch._foreach_mul_(target_params, momentum)
        torch._foreach_add_(target_params, student_params, alpha=1.0 - momentum)
        scene_momentum = self._target_scene_ema_momentum(global_step)
        scene_student_params = list(self.scene_projector.parameters())
        scene_target_params = list(self.target_scene_projector.parameters())
        torch._foreach_mul_(scene_target_params, scene_momentum)
        torch._foreach_add_(scene_target_params, scene_student_params, alpha=1.0 - scene_momentum)
