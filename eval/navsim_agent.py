"""NAVSIM feature builder and inference-only agent wrapper.

The same adapter is used by both the NAVSIM v1.1 PDM and NAVSIM v2 EPDMS
entrypoints.  It intentionally keeps the model interface small:

``YAML config -> build_world_model -> checkpoint -> predict_trajectory``.
"""

from __future__ import annotations

import os
import inspect
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from nuplan.common.actor_state.state_representation import StateSE2

_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _prepare_worldmodel_imports() -> None:
    if _REPO_ROOT in sys.path:
        sys.path.remove(_REPO_ROOT)
    sys.path.insert(0, _REPO_ROOT)
    existing_utils = sys.modules.get("utils")
    if existing_utils is not None and not hasattr(existing_utils, "__path__"):
        sys.modules.pop("utils", None)


_prepare_worldmodel_imports()

from datasets.ego_trajectory_utils import convert_absolute_to_relative_se2_array
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from models import build_world_model
from training.checkpoint import load_checkpoint


def _ego_pose_to_array(pose) -> np.ndarray:
    if hasattr(pose, "x") and hasattr(pose, "y") and hasattr(pose, "heading"):
        return np.array([pose.x, pose.y, pose.heading], dtype=np.float64)
    arr = np.asarray(pose, dtype=np.float64).ravel()
    if arr.size != 3:
        raise ValueError(f"ego_pose must have 3 values, got shape {np.asarray(pose).shape}")
    return arr.astype(np.float64)


class WorldModelFeatureBuilder(AbstractFeatureBuilder):
    """Build features from NAVSIM ``AgentInput`` in the training dataset format."""

    def __init__(
        self,
        resize_to: tuple[int, int] = (256, 512),
        num_history_image_frames: int = 1,
        num_history_trajectory_steps: int = 4,
        camera_names: list[str] | None = None,
        skip_images: bool = False,
    ):
        super().__init__()
        self.resize_to = resize_to
        self.num_history_image_frames = int(num_history_image_frames)
        self.num_history_trajectory_steps = int(num_history_trajectory_steps)
        self.camera_names = [str(c).lower() for c in (camera_names or ["cam_f0"])]
        self.skip_images = bool(skip_images)
        if self.num_history_image_frames < 1:
            raise ValueError("num_history_image_frames must be >= 1")
        if self.num_history_trajectory_steps < 1:
            raise ValueError("num_history_trajectory_steps must be >= 1")

    def get_unique_name(self) -> str:
        return "worldmodel_feature"

    def _image_tensor_from_camera(self, camera, name: str = "camera") -> torch.Tensor:
        import cv2

        img = camera.image
        if img is None:
            raise RuntimeError(f"{name}.image is None; check SensorConfig / SceneFilter history.")
        if isinstance(img, (str, Path)):
            img_bgr = cv2.imread(str(img))
            if img_bgr is None:
                raise RuntimeError(f"Failed to load image: {img}")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img,
            (self.resize_to[1], self.resize_to[0]),
            interpolation=cv2.INTER_AREA,
        )
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        if self.skip_images:
            history_images = None
        else:
            cameras_list = agent_input.cameras
            if len(cameras_list) < self.num_history_image_frames:
                raise RuntimeError(
                    f"agent_input.cameras has {len(cameras_list)} frames, "
                    f"expected >= {self.num_history_image_frames}"
                )
            hist_cams = cameras_list[-self.num_history_image_frames:]
            per_time = []
            for cams in hist_cams:
                per_time.append(
                    torch.stack(
                        [
                            self._image_tensor_from_camera(getattr(cams, cam_name), cam_name)
                            for cam_name in self.camera_names
                        ],
                        dim=0,
                    )
                )
            history_images = torch.stack(per_time, dim=0)
            if len(self.camera_names) == 1:
                history_images = history_images[:, 0]

        ego_statuses = agent_input.ego_statuses
        if len(ego_statuses) < self.num_history_trajectory_steps:
            raise RuntimeError(
                f"agent_input.ego_statuses has {len(ego_statuses)} frames, "
                f"expected >= {self.num_history_trajectory_steps}"
            )
        history_poses = np.array(
            [_ego_pose_to_array(e.ego_pose) for e in ego_statuses[-self.num_history_trajectory_steps:]],
            dtype=np.float64,
        )
        if history_poses.ndim != 2 or history_poses.shape[1] != 3:
            raise ValueError(
                "ego_status.ego_pose history must have shape "
                f"[T, 3], got {history_poses.shape}"
            )
        current_pose = history_poses[-1]
        origin = StateSE2(float(current_pose[0]), float(current_pose[1]), float(current_pose[2]))
        history_traj = torch.tensor(
            convert_absolute_to_relative_se2_array(origin, history_poses).astype(np.float32),
            dtype=torch.float32,
        )

        current_status = ego_statuses[-1]
        driving_cmd = current_status.driving_command
        if driving_cmd is not None:
            arr = np.asarray(driving_cmd, dtype=np.float32).ravel()
            if arr.size != 4:
                raise ValueError(f"driving_command must have 4 values, got {arr.size}")
            nav_cmd = int(np.argmax(arr))
        else:
            nav_cmd = 1

        ego_velocity = np.asarray(current_status.ego_velocity, dtype=np.float32).ravel()
        ego_acceleration = np.asarray(current_status.ego_acceleration, dtype=np.float32).ravel()
        if ego_velocity.size != 2 or ego_acceleration.size != 2:
            raise ValueError("ego_velocity and ego_acceleration must both have length 2")
        ego_speed = float(np.linalg.norm(ego_velocity))

        driving_cmd_oh = np.zeros(4, dtype=np.float32)
        driving_cmd_oh[min(nav_cmd, 3)] = 1.0
        ego_status = np.concatenate([driving_cmd_oh, ego_velocity, ego_acceleration])

        out: Dict[str, torch.Tensor] = {
            "history_trajectory": history_traj,
            "navigation_command": torch.tensor(nav_cmd),
            "ego_speed": torch.tensor(ego_speed),
            "ego_status": torch.from_numpy(ego_status),
            "trajectory_interval_s": torch.tensor(0.5),
        }
        if history_images is not None:
            out["history_images"] = history_images
        return out


class WorldModelTargetBuilder(AbstractTargetBuilder):
    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        future_traj = scene.get_future_trajectory(num_trajectory_frames=8)
        return {"trajectory": torch.tensor(future_traj.poses)}


class WorldModelNavsimAgent(AbstractAgent):
    """Optional offline NAVSIM agent for models that implement ``predict_trajectory``."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        config_overrides: list[str] | None = None,
        trajectory_sampling=None,
        **_unused_kwargs,
    ):
        if "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters:
            super().__init__(trajectory_sampling, requires_scene=False)
        else:
            super().__init__()
            self.requires_scene = False
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self._device = device
        self.config_overrides = list(config_overrides or [])
        self.model = None
        self._num_history_image_frames = 1
        self._num_history_trajectory_steps = 4
        self._resize_to = (256, 512)
        self._camera_names = ["cam_f0"]
        self._trajectory_sampling = trajectory_sampling
        try:
            cfg = self._load_cfg()
            data_cfg = cfg.get("data", {}) or {}
            camera_names = data_cfg.get("camera_names", None)
            if camera_names is None:
                camera_names = [data_cfg.get("camera_name", "cam_f0")]
            self._camera_names = [
                str(c).lower()
                for c in camera_names
            ]
        except Exception:
            pass

    def _load_cfg(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(self.config_path)
        if self.config_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(self.config_overrides))
        return cfg

    def name(self) -> str:
        return "worldmodel"

    def get_sensor_config(self) -> SensorConfig:
        names = set(self._camera_names)
        return SensorConfig(
            cam_f0="cam_f0" in names,
            cam_l0="cam_l0" in names,
            cam_l1="cam_l1" in names,
            cam_l2="cam_l2" in names,
            cam_r0="cam_r0" in names,
            cam_r1="cam_r1" in names,
            cam_r2="cam_r2" in names,
            cam_b0="cam_b0" in names,
            lidar_pc=False,
        )

    def initialize(self) -> None:
        cfg = self._load_cfg()
        from omegaconf import OmegaConf

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        data_cfg = cfg_dict.get("data", {}) or {}
        self._num_history_image_frames = int(data_cfg.get("num_history_image_frames", 1))
        self._num_history_trajectory_steps = int(data_cfg.get("num_history_trajectory_steps", 4))
        self._resize_to = (
            int(data_cfg.get("frame_height", 256)),
            int(data_cfg.get("frame_width", 512)),
        )
        self._camera_names = [str(c).lower() for c in data_cfg.get("camera_names", [data_cfg.get("camera_name", "cam_f0")])]
        self.model = build_world_model(cfg_dict).to(self._device)
        if self.checkpoint_path:
            load_checkpoint(self.checkpoint_path, self.model, map_location=self._device)
        self.model.eval()

    def get_feature_builders(self):
        return [
            WorldModelFeatureBuilder(
                resize_to=self._resize_to,
                num_history_image_frames=self._num_history_image_frames,
                num_history_trajectory_steps=self._num_history_trajectory_steps,
                camera_names=self._camera_names,
            )
        ]

    def get_target_builders(self):
        return [WorldModelTargetBuilder()]

    @torch.no_grad()
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        if self.model is None:
            raise RuntimeError("Agent is not initialized.")
        builder = self.get_feature_builders()[0]
        features = builder.compute_features(agent_input)
        features = {
            k: (v.unsqueeze(0).to(self._device) if isinstance(v, torch.Tensor) else v)
            for k, v in features.items()
        }
        pred = self.model.predict_trajectory(features)
        poses = pred.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        sampling = getattr(self, "_trajectory_sampling", None)
        try:
            return Trajectory(poses, sampling) if sampling is not None else Trajectory(poses)
        except TypeError:
            # Older NAVSIM dataclasses use a default 4 s / 0.5 Hz sampling.
            return Trajectory(poses)
