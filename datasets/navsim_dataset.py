"""NavSim dataset — loads from .pkl frame lists + sensor_blobs.

Each .pkl file contains a list of frame dicts for one log.  Frames are at 2 Hz
(0.5 s interval).  We cut sliding windows and output the same dict format as
``NuplanDevkitDataset`` for seamless multi-dataset training.

Supports official navsim scene_filter YAML files for train/val splitting
(navtrain.yaml, navtest.yaml) to ensure no data leakage.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset

from datasets.ego_trajectory_utils import convert_absolute_to_relative_se2_array
from datasets.nav_command_infer import (
    NAV_COMMAND_COMBINE,
    NAV_COMMAND_FORWARD_M,
    NAV_COMMAND_HEADING_RAD,
    NAV_COMMAND_INFER_MODE,
    NAV_COMMAND_LATERAL_M,
    infer_navigation_command,
)
from nuplan.common.actor_state.state_representation import StateSE2

try:
    from pyquaternion import Quaternion
except ImportError:
    Quaternion = None

NAVSIM_INTERVAL_S = 0.5
NAVSIM_FRAME_HZ = 2.0


def _normalize_scene_token(raw: str) -> str:
    """Align UUID-style vs compact hex tokens in scene_filter YAML."""
    if not raw:
        return ""
    s = str(raw).strip().lower()
    return "".join(c for c in s if c not in "-_ ")


def _expand_scene_token_allowlist(tokens: Set[str]) -> Set[str]:
    """Membership keys: raw + dash-stripped (navtest YAML uses 16-char hex)."""
    out: Set[str] = set(tokens)
    for t in tokens:
        out.add(_normalize_scene_token(t))
    return out


def _frame_token_allowed(raw: str, allow: Set[str]) -> bool:
    if raw in allow:
        return True
    return _normalize_scene_token(raw) in allow


def load_scene_filter_yaml(yaml_path: str) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
    """Parse a navsim scene_filter YAML to extract log_names and tokens sets.

    Uses a lightweight parser (no Hydra) since these files can be 100k+ lines.
    Returns (log_names_set, tokens_set). Either can be None if not specified.
    """
    log_names: List[str] = []
    tokens: List[str] = []
    current_section = None

    with open(yaml_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("log_names:"):
                current_section = "log_names"
                continue
            elif stripped.startswith("tokens:"):
                current_section = "tokens"
                continue
            elif stripped and not stripped.startswith("-") and not stripped.startswith("#"):
                if ":" in stripped:
                    current_section = None
                    continue

            if current_section and stripped.startswith("- "):
                val = stripped[2:].strip().strip("'\"")
                if val and val != "null":
                    if current_section == "log_names":
                        log_names.append(val)
                    elif current_section == "tokens":
                        tokens.append(val)

    return (
        set(log_names) if log_names else None,
        set(tokens) if tokens else None,
    )


def _global_pose_from_frame(frame: Dict[str, Any]) -> np.ndarray:
    """Extract (x, y, yaw) from a navsim frame dict."""
    trans = frame["ego2global_translation"]
    if Quaternion is not None:
        q = Quaternion(*frame["ego2global_rotation"])
        yaw = float(q.yaw_pitch_roll[0])
    else:
        rot = frame["ego2global_rotation"]
        q_w, q_x, q_y, q_z = rot[0], rot[1], rot[2], rot[3]
        yaw = float(np.arctan2(2.0 * (q_w * q_z + q_x * q_y),
                                1.0 - 2.0 * (q_y ** 2 + q_z ** 2)))
    return np.array([trans[0], trans[1], yaw], dtype=np.float64)


class NavsimDataset(Dataset):
    """Load navsim .pkl logs and produce training samples aligned with
    ``NuplanDevkitDataset``'s output format.

    Args:
        navsim_log_path:   directory with ``*.pkl`` log files.
        sensor_blobs_path: root directory for camera image relative paths.
        num_history_frames: number of history frames (including current).
        num_future_frames:  number of future frames.
        resize_to:          (H, W) for output images.
        camera_name:        which camera to use (default ``cam_f0``).
        log_names:          if set, only load these log names (no ``.pkl``).
        scene_tokens:       if set, only keep scenes with token in this set.
        window_stride:      advance in frames between window starts on the 2Hz
                             timeline. ``None`` means stride equals ``window_size``
                             (non-overlapping). Use ``1`` for dense / overlapping
                             sliding windows.
    """

    def __init__(
        self,
        navsim_log_path: str,
        sensor_blobs_path: str,
        num_history_image_frames: int = 1,
        num_history_trajectory_steps: int = 4,
        num_future_frames: int = 8,
        resize_to: Tuple[int, int] = (256, 512),
        camera_name: str = "cam_f0",
        log_names: Optional[Set[str]] = None,
        scene_tokens: Optional[Set[str]] = None,
        scene_filter_yaml: Optional[str] = None,
        window_stride: Optional[int] = None,
        verbose: bool = True,
        nav_command_infer_mode: Optional[str] = None,
        nav_command_forward_m: Optional[float] = None,
        nav_command_lateral_m: Optional[float] = None,
        nav_command_heading_rad: Optional[float] = None,
        nav_command_combine: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._nav_infer_mode = (
            nav_command_infer_mode
            if nav_command_infer_mode is not None
            else NAV_COMMAND_INFER_MODE
        )
        self._nav_forward_m = float(
            nav_command_forward_m
            if nav_command_forward_m is not None
            else NAV_COMMAND_FORWARD_M
        )
        self._nav_lateral_m = float(
            nav_command_lateral_m
            if nav_command_lateral_m is not None
            else NAV_COMMAND_LATERAL_M
        )
        self._nav_heading_rad = float(
            nav_command_heading_rad
            if nav_command_heading_rad is not None
            else NAV_COMMAND_HEADING_RAD
        )
        self._nav_combine = (
            nav_command_combine
            if nav_command_combine is not None
            else NAV_COMMAND_COMBINE
        )
        self.sensor_blobs_path = Path(sensor_blobs_path)
        self.resize_to = resize_to
        self.camera_name = camera_name.lower()
        self.num_history_images = num_history_image_frames
        self.num_history_traj = num_history_trajectory_steps
        self.num_history = max(num_history_image_frames, num_history_trajectory_steps)
        self.num_future = num_future_frames
        self.window_size = self.num_history + num_future_frames
        if window_stride is not None:
            if int(window_stride) < 1:
                raise ValueError(
                    f"window_stride must be >= 1, got {window_stride}"
                )
            self._window_stride = int(window_stride)
        else:
            self._window_stride = 1

        if scene_filter_yaml and os.path.isfile(scene_filter_yaml):
            yaml_logs, yaml_tokens = load_scene_filter_yaml(scene_filter_yaml)
            if yaml_logs and log_names is None:
                log_names = yaml_logs
            if yaml_tokens and scene_tokens is None:
                scene_tokens = yaml_tokens
            if verbose:
                print(
                    f"[NavsimDataset] Loaded scene_filter: "
                    f"{len(log_names) if log_names else 0} logs, "
                    f"{len(scene_tokens) if scene_tokens else 0} tokens "
                    f"from {scene_filter_yaml}"
                )

        token_allow: Optional[Set[str]] = (
            _expand_scene_token_allowlist(scene_tokens) if scene_tokens is not None else None
        )

        self.samples: List[Tuple[str, int, str]] = []

        log_dir = Path(navsim_log_path)
        pkl_files = sorted(log_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {log_dir}")

        for pkl_path in pkl_files:
            log_name = pkl_path.stem
            if log_names is not None and log_name not in log_names:
                continue

            with open(pkl_path, "rb") as f:
                frames: List[Dict] = pickle.load(f)

            if len(frames) < self.window_size:
                continue

            step = self._window_stride
            for start in range(0, len(frames) - self.window_size + 1, step):
                current_idx = start + self.num_history - 1
                token = frames[current_idx].get("token", "")
                if token_allow is not None and not _frame_token_allowed(token, token_allow):
                    continue
                self.samples.append((str(pkl_path), start, token))

        if verbose:
            print(
                f"[NavsimDataset] Loaded {len(self.samples)} samples "
                f"from {len(pkl_files)} pkl files "
                f"(window_stride={self._window_stride}, window_size={self.window_size})"
            )
        if len(self.samples) == 0 and scene_filter_yaml:
            print(
                "[NavsimDataset] WARNING: 0 samples after filters. "
                "navtest/navtrain tokens must match ``frame['token']`` in your .pkl (often a format mismatch). "
                "Temporarily set ``data.val_scene_filter_yaml: null`` to reuse the train split for val, or fix tokens."
            )

        self._frame_cache: Dict[str, List[Dict]] = {}
        self._frame_cache_max = 8

    def _load_frames(self, pkl_path: str) -> List[Dict]:
        if pkl_path not in self._frame_cache:
            if len(self._frame_cache) >= self._frame_cache_max:
                oldest = next(iter(self._frame_cache))
                del self._frame_cache[oldest]
            with open(pkl_path, "rb") as f:
                self._frame_cache[pkl_path] = pickle.load(f)
        return self._frame_cache[pkl_path]

    def _image_path(self, frame: Dict, cam_name: str) -> str:
        cams = frame.get("cams", {})
        cam_key = None
        for k in cams:
            if k.lower() == cam_name:
                cam_key = k
                break
        if cam_key is None:
            raise KeyError(f"Camera {cam_name} not found in frame. Available: {list(cams.keys())}")

        rel_path = cams[cam_key]["data_path"]
        return str(self.sensor_blobs_path / rel_path)

    def _load_image_tensor_from_path(self, img_path: str) -> torch.Tensor:
        """Load and preprocess a single camera image to [-1, 1] tensor."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resize_to[1], self.resize_to[0]),
                         interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    def _load_image_tensor(self, frame: Dict, cam_name: str) -> torch.Tensor:
        return self._load_image_tensor_from_path(self._image_path(frame, cam_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pkl_path, start, scene_token = self.samples[index]
        frames = self._load_frames(pkl_path)
        window = frames[start: start + self.window_size]

        current_idx = self.num_history - 1

        global_poses = np.array(
            [_global_pose_from_frame(f) for f in window], dtype=np.float64
        )
        origin = StateSE2(*global_poses[current_idx])
        local_poses = convert_absolute_to_relative_se2_array(origin, global_poses)

        traj_start = self.num_history - self.num_history_traj
        hist_traj_poses = local_poses[traj_start:self.num_history]
        fut_poses = local_poses[self.num_history:]

        img_start = self.num_history - self.num_history_images
        history_image_paths = [
            self._image_path(window[i], self.camera_name)
            for i in range(img_start, self.num_history)
        ]
        hist_tensors = [
            self._load_image_tensor_from_path(p)
            for p in history_image_paths
        ]
        fut_tensors = [
            self._load_image_tensor(window[self.num_history + i], self.camera_name)
            for i in range(self.num_future)
        ]

        history_images = torch.stack(hist_tensors, dim=0)
        future_images = torch.stack(fut_tensors, dim=0)

        current_frame = window[current_idx]
        driving_cmd = current_frame.get("driving_command", None)
        if driving_cmd is not None:
            arr = np.asarray(driving_cmd, dtype=np.float32).ravel()
            if arr.size != 4:
                raise ValueError(
                    f"driving_command must have 4 values (NAVSIM one-hot), got size {arr.size}"
                )
            nav_cmd = int(np.argmax(arr))
        else:
            warnings.warn(
                "[NavsimDataset] `driving_command` missing; falling back to "
                "`infer_navigation_command(fut_poses, ...)` (same rules as NuPlan). "
                f"scene_token={scene_token!r} index={index} pkl={pkl_path}",
                UserWarning,
            )
            nav_cmd = infer_navigation_command(
                fut_poses,
                self._nav_infer_mode,
                forward_m=self._nav_forward_m,
                lateral_threshold_m=self._nav_lateral_m,
                heading_threshold_rad=self._nav_heading_rad,
                combine=self._nav_combine,
            )

        # 与 recogdrive `EgoStatus` / `status_feature` 一致：ego_dynamic_state = 2 vel + 2 acc（总长 4）
        ego_dyn = current_frame["ego_dynamic_state"]
        ego_dyn = np.asarray(ego_dyn, dtype=np.float32).ravel()
        if ego_dyn.size != 4:
            raise ValueError(
                f"ego_dynamic_state must have length 4 (2 vel + 2 acc), got {ego_dyn.size}"
            )
        ego_velocity = ego_dyn[:2].astype(np.float32)
        ego_acceleration = ego_dyn[2:4].astype(np.float32)

        ego_speed = float(np.linalg.norm(ego_velocity))

        driving_cmd_oh = np.zeros(4, dtype=np.float32)
        driving_cmd_oh[min(int(nav_cmd), 3)] = 1.0
        ego_status = np.concatenate([driving_cmd_oh, ego_velocity, ego_acceleration])

        return {
            "history_images": history_images,
            "history_image_paths": history_image_paths,
            "future_images": future_images,
            "history_trajectory": torch.from_numpy(hist_traj_poses.astype(np.float32)),
            "future_trajectory": torch.from_numpy(fut_poses.astype(np.float32)),
            "trajectory_interval_s": NAVSIM_INTERVAL_S,
            "navigation_command": nav_cmd,
            "ego_speed": ego_speed,
            "ego_status": torch.from_numpy(ego_status),
            "dataset_id": 1,
            "scene_token": scene_token,
            "index": index,
        }
