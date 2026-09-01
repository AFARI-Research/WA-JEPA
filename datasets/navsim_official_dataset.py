"""NAVSIM 训练样本：通过官方 ``SceneLoader`` + ``SceneFilter`` 建索引（与 recogdrive / PDM 同源）。

与 ``NavsimDataset``（pkl 滑窗）输出字段一致，便于同一套模型与 trainer 复用；
区别在于样本列表 = 官方 filter 下的 scene token，锚点与 metric cache / 评测一致。
"""

from __future__ import annotations

import hashlib
import inspect
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import cv2
import numpy as np
import torch
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
from datasets.navsim_dataset import NAVSIM_INTERVAL_S, _global_pose_from_frame
from nuplan.common.actor_state.state_representation import StateSE2


def load_navsim_scene_filter_from_yaml(yaml_path: str):
    """从 NAVSIM devkit 的 Hydra YAML 实例化 ``SceneFilter``（与 eval_navsim_pdm 一致）。"""
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SceneFilter

    p = Path(os.path.expanduser(yaml_path))
    if not p.is_file():
        raise FileNotFoundError(f"Scene filter YAML not found: {p}")
    cfg = OmegaConf.load(p)
    d = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(d, dict):
        raise TypeError(f"Scene filter YAML root must be a mapping, got {type(d).__name__}")
    d.pop("_target_", None)
    d.pop("_convert_", None)
    return SceneFilter(**cast(Dict[str, Any], d))


def log_holdout_fraction(log_name: str) -> float:
    """Stable per-log value in ``[0, 1)`` used to carve a validation holdout.

    Hashed rather than sliced by position so the split does not move when the filter
    YAML is regenerated in a different order, and stable across processes and runs
    (``hash()`` is salted per interpreter, ``blake2b`` is not).
    """
    digest = hashlib.blake2b(log_name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def split_log_names(
    log_names: Iterable[str],
    holdout_ratio: float,
    *,
    want_holdout: bool,
) -> List[str]:
    """Partition log names into a train part and a validation holdout.

    ``want_holdout=True`` returns the holdout logs, ``False`` the remainder. The two
    calls are exact complements, so no log can appear in both loaders.
    """
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError(
            f"navsim_val_holdout_ratio must be in (0, 1), got {holdout_ratio}"
        )
    selected = [
        name
        for name in log_names
        if (log_holdout_fraction(name) < holdout_ratio) == want_holdout
    ]
    if not selected:
        raise ValueError(
            "Holdout split produced an empty side "
            f"(holdout_ratio={holdout_ratio}, want_holdout={want_holdout}). "
            "Choose a ratio that leaves logs on both sides."
        )
    return sorted(selected)


class NavsimOfficialDataset(Dataset):
    """使用官方 ``filter_scenes`` / ``SceneLoader`` 的 token 列表，按与 ``NavsimDataset`` 相同规则张量化。

    要求：

    - ``max(num_history_image_frames, num_history_trajectory_steps)`` 必须等于
      ``SceneFilter.num_history_frames``（与官方 scene 锚点一致）。
    - ``num_future_frames``（训练监督步数）不得大于 ``SceneFilter.num_future_frames``
      （场景里可用的未来原始帧数）；若 yaml 中未来帧更多，仅取前 ``num_future_frames`` 步。
    """

    def __init__(
        self,
        navsim_log_path: str,
        sensor_blobs_path: str,
        scene_filter_yaml: str,
        original_sensor_path: Optional[str] = None,
        synthetic_sensor_path: Optional[str] = None,
        synthetic_scenes_path: Optional[str] = None,
        num_history_image_frames: int = 1,
        num_history_trajectory_steps: int = 4,
        num_future_frames: int = 8,
        resize_to: Tuple[int, int] = (256, 512),
        camera_name: str = "cam_f0",
        camera_names: Optional[List[str]] = None,
        verbose: bool = True,
        nav_command_infer_mode: Optional[str] = None,
        nav_command_forward_m: Optional[float] = None,
        nav_command_lateral_m: Optional[float] = None,
        nav_command_heading_rad: Optional[float] = None,
        nav_command_combine: Optional[str] = None,
        num_history_image_paths: Optional[int] = None,
        log_holdout_ratio: Optional[float] = None,
        log_holdout_side: str = "train",
    ) -> None:
        super().__init__()
        from navsim.common.dataclasses import SceneFilter, SensorConfig
        from navsim.common.dataloader import SceneLoader

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

        original_image_root = original_sensor_path or sensor_blobs_path
        if not original_image_root and not synthetic_sensor_path:
            raise ValueError(
                "NavsimOfficialDataset requires sensor_blobs_path or "
                "original_sensor_path/synthetic_sensor_path for image loading."
            )
        self.original_sensor_path = (
            Path(original_image_root) if original_image_root else None
        )
        self.synthetic_sensor_path = (
            Path(synthetic_sensor_path) if synthetic_sensor_path else None
        )
        # Backward-compatible attribute used by callers that inspect the dataset.
        self.sensor_blobs_path = (
            self.original_sensor_path or self.synthetic_sensor_path
        )
        self.resize_to = resize_to
        self.camera_names = [str(c).lower() for c in (camera_names or [camera_name])]
        if not self.camera_names:
            raise ValueError("NavsimOfficialDataset requires at least one camera")
        self.camera_name = self.camera_names[0]
        self.num_history_images = num_history_image_frames
        self.num_history_traj = num_history_trajectory_steps
        self.num_history = max(num_history_image_frames, num_history_trajectory_steps)
        self.num_future = num_future_frames
        self.num_history_image_paths = (
            int(num_history_image_paths)
            if num_history_image_paths is not None
            else int(num_history_image_frames)
        )
        if self.num_history_image_paths > self.num_history:
            raise ValueError(
                "NavsimOfficialDataset: num_history_image_paths "
                f"({self.num_history_image_paths}) cannot exceed "
                f"max history frames ({self.num_history})."
            )

        self._scene_filter: SceneFilter = load_navsim_scene_filter_from_yaml(
            scene_filter_yaml
        )
        nh_sf = int(self._scene_filter.num_history_frames)
        nf_sf = int(self._scene_filter.num_future_frames)

        if self.num_history != nh_sf:
            raise ValueError(
                "NavsimOfficialDataset: max(num_history_image_frames, "
                "num_history_trajectory_steps) must equal SceneFilter.num_history_frames "
                f"({nh_sf}), got {self.num_history}"
            )
        if num_future_frames > nf_sf:
            raise ValueError(
                "NavsimOfficialDataset: num_future_frames cannot exceed "
                f"SceneFilter.num_future_frames ({nf_sf}), got {num_future_frames}"
            )

        # Carve validation out of the same filter by log name. NAVSIM's navtrain and
        # navtest filters share no logs, so validating on navtest means validating on the
        # benchmark's test split; a holdout keeps in-training validation honest and leaves
        # navtest for the final external scoring run.
        if log_holdout_ratio is not None:
            if log_holdout_side not in ("train", "val"):
                raise ValueError(
                    f"log_holdout_side must be 'train' or 'val', got {log_holdout_side!r}"
                )
            if self._scene_filter.log_names is None:
                raise ValueError(
                    "A log holdout needs log_names in the scene filter YAML, and "
                    f"{scene_filter_yaml} has none."
                )
            kept = split_log_names(
                self._scene_filter.log_names,
                float(log_holdout_ratio),
                want_holdout=log_holdout_side == "val",
            )
            if verbose:
                print(
                    f"[NavsimOfficialDataset] log holdout side={log_holdout_side} "
                    f"ratio={log_holdout_ratio}: {len(kept)} of "
                    f"{len(self._scene_filter.log_names)} logs"
                )
            self._scene_filter.log_names = kept
            # ``tokens`` stays as written. NAVSIM's ``filter_scenes`` drops whole log files
            # by name first and only then checks tokens, so tokens belonging to dropped logs
            # never match; clearing it would instead *add* scenes the filter excluded.

        scene_loader_params = inspect.signature(SceneLoader).parameters
        if "sensor_blobs_path" in scene_loader_params:
            if not sensor_blobs_path:
                raise ValueError("NAVSIM v1 SceneLoader requires sensor_blobs_path")
            self._scene_loader = SceneLoader(
                data_path=Path(navsim_log_path),
                sensor_blobs_path=Path(sensor_blobs_path),
                scene_filter=self._scene_filter,
                sensor_config=SensorConfig.build_no_sensors(),
            )
        else:
            original_sensor = original_sensor_path or sensor_blobs_path
            self._scene_loader = SceneLoader(
                synthetic_sensor_path=(
                    Path(synthetic_sensor_path) if synthetic_sensor_path else None
                ),
                original_sensor_path=(
                    Path(original_sensor) if original_sensor else None
                ),
                data_path=Path(navsim_log_path),
                synthetic_scenes_path=(
                    Path(synthetic_scenes_path) if synthetic_scenes_path else None
                ),
                scene_filter=self._scene_filter,
                sensor_config=SensorConfig.build_no_sensors(),
            )
        self._tokens: List[str] = sorted(self._scene_loader.scene_frames_dicts.keys())

        if verbose:
            print(
                f"[NavsimOfficialDataset] SceneLoader tokens={len(self._tokens)} "
                f"filter={scene_filter_yaml} "
                f"(num_history_frames={nh_sf}, num_future_frames={nf_sf}, "
                f"supervision_future={num_future_frames})"
            )

    @property
    def scene_filter(self):
        return self._scene_filter

    @property
    def scene_loader(self):
        return self._scene_loader

    @property
    def tokens(self) -> list:
        """返回有序 scene token 列表，与 ``__getitem__`` 的下标一一对应。无 I/O。"""
        return self._tokens

    def __len__(self) -> int:
        return len(self._tokens)

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
        path = Path(rel_path)
        if path.is_absolute():
            return str(path)

        roots = [
            root
            for root in (self.original_sensor_path, self.synthetic_sensor_path)
            if root is not None
        ]
        if len(roots) == 1:
            return str(roots[0] / path)

        candidates = [
            root / path
            for root in roots
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return str(candidates[0])

    def _load_image_tensor_from_path(self, img_path: str) -> torch.Tensor:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img,
            (self.resize_to[1], self.resize_to[0]),
            interpolation=cv2.INTER_AREA,
        )
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    def _load_image_tensor(self, frame: Dict, cam_name: str) -> torch.Tensor:
        return self._load_image_tensor_from_path(self._image_path(frame, cam_name))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        scene_token = self._tokens[index]
        frame_list = self._scene_loader.scene_frames_dicts[scene_token]

        nh_sf = int(self._scene_filter.num_history_frames)
        window_len = nh_sf + self.num_future
        if len(frame_list) < window_len:
            raise RuntimeError(
                f"Scene {scene_token!r} has {len(frame_list)} frames, need >= {window_len}"
            )
        window = frame_list[:window_len]

        current_idx = nh_sf - 1

        global_poses = np.array(
            [_global_pose_from_frame(f) for f in window], dtype=np.float64
        )
        origin = StateSE2(*global_poses[current_idx])
        local_poses = convert_absolute_to_relative_se2_array(origin, global_poses)

        traj_start = self.num_history - self.num_history_traj
        hist_traj_poses = local_poses[traj_start:nh_sf]
        fut_poses = local_poses[nh_sf:]

        img_start = nh_sf - self.num_history_images
        img_path_start = nh_sf - self.num_history_image_paths
        history_image_paths = [
            [self._image_path(window[i], cam_name) for cam_name in self.camera_names]
            for i in range(img_path_start, nh_sf)
        ]
        load_offset = self.num_history_image_paths - self.num_history_images
        hist_tensors = []
        for paths_t in history_image_paths[load_offset:]:
            hist_tensors.append(torch.stack([self._load_image_tensor_from_path(p) for p in paths_t], dim=0))
        fut_tensors = []
        for i in range(self.num_future):
            fut_tensors.append(
                torch.stack(
                    [self._load_image_tensor(window[nh_sf + i], cam_name) for cam_name in self.camera_names],
                    dim=0,
                )
            )

        history_images = torch.stack(hist_tensors, dim=0)
        future_images = torch.stack(fut_tensors, dim=0)
        if len(self.camera_names) == 1:
            history_images = history_images[:, 0]
            future_images = future_images[:, 0]
            history_image_paths_out: Any = [paths[0] for paths in history_image_paths]
        else:
            history_image_paths_out = history_image_paths

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
                "[NavsimOfficialDataset] `driving_command` missing; falling back to "
                "`infer_navigation_command(fut_poses, ...)`. "
                f"scene_token={scene_token!r} index={index}",
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
            "history_image_paths": history_image_paths_out,
            "future_images": future_images,
            "camera_names": list(self.camera_names),
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
