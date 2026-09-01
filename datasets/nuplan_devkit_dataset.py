# 使用 nuplan-devkit 底层 API 直接从 log 加载 NuPlan 相机数据（不依赖 scenario）
# 每个 log 是一段连续驾驶，直接在 log 时间线上切片采样，避免 scenario 重叠带来的冗余

from __future__ import annotations

import bisect
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import cv2

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.database.nuplan_db.image import Image
from nuplan.database.nuplan_db.nuplan_db_utils import get_camera_channel_sensor_data
from nuplan.database.nuplan_db import nuplan_scenario_queries as _nuplan_scenario_queries
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_sensor_data_from_sensor_data_tokens_from_db,
)
import nuplan.database.nuplan_db.query_session as nuplan_query_session
from nuplan.database.common.blob_store.local_store import LocalStore
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import discover_log_dbs
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    download_file_if_necessary,
    load_image,
)

from datasets.ego_trajectory_utils import (
    convert_absolute_to_relative_se2_array,
    ego_dynamic_from_token,
)
from datasets.nav_command_infer import (
    NAV_COMMAND_COMBINE,
    NAV_COMMAND_FORWARD_M,
    NAV_COMMAND_HEADING_RAD,
    NAV_COMMAND_INFER_MODE,
    NAV_COMMAND_LATERAL_M,
    infer_navigation_command,
)

# NuPlan 相机 image 序列名义帧率（用于默认相机时间对齐容差）
NUPLAN_IMAGE_FPS = 10.0

DEFAULT_CAMERA_TIME_TOLERANCE_US = int(0.5 / NUPLAN_IMAGE_FPS * 1_000_000)


def _install_nuplan_sqlite_busy_timeout() -> None:
    """
    多进程 DataLoader + 分布式下，大量并发打开同一 nuPlan log 的 sqlite 时，
    默认 connect(timeout=5s) 易触发 OperationalError（如 database is locked），
    导致某 rank 的 worker 崩溃，其余 rank 在 ALLREDUCE 上等到 NCCL 超时。
    """
    timeout_sec = float(os.environ.get("NUPLAN_SQLITE_BUSY_TIMEOUT_SEC", "120"))

    def execute_many(
        query_text: str, query_parameters: Any, db_file: str
    ) -> Generator["sqlite3.Row", None, None]:
        import sqlite3

        connection = sqlite3.connect(db_file, timeout=timeout_sec)
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        try:
            cursor.execute(query_text, query_parameters)
            for row in cursor:
                yield row
        finally:
            cursor.close()
            connection.close()

    def execute_one(
        query_text: str, query_parameters: Any, db_file: str
    ) -> Optional["sqlite3.Row"]:
        import sqlite3

        connection = sqlite3.connect(db_file, timeout=timeout_sec)
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        try:
            cursor.execute(query_text, query_parameters)
            result: Optional[sqlite3.Row] = cursor.fetchone()
            if result is not None and cursor.fetchone() is not None:
                raise RuntimeError("execute_one query returned multiple rows.")
            return result
        finally:
            cursor.close()
            connection.close()

    nuplan_query_session.execute_many = execute_many  # type: ignore[method-assign]
    nuplan_query_session.execute_one = execute_one  # type: ignore[method-assign]
    _nuplan_scenario_queries.execute_many = execute_many  # type: ignore[method-assign]
    _nuplan_scenario_queries.execute_one = execute_one  # type: ignore[method-assign]


_install_nuplan_sqlite_busy_timeout()


def _get_min_max_timestamp(log_file: str, table: str = "lidar_pc") -> Optional[Tuple[int, int]]:
    """获取 log 中指定表的 min/max timestamp (us)。"""
    result = nuplan_query_session.execute_one(
        f"SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM {table}",
        (),
        log_file,
    )
    if result is None or result["min_ts"] is None:
        return None
    return int(result["min_ts"]), int(result["max_ts"])


def _get_image_records_in_time_window(
    log_file: str,
    camera_channel: str,
    min_ts: int,
    max_ts: int,
    subsample_interval: int = 1,
) -> List[Tuple[str, int]]:
    """
    直接从 image 表按时间顺序获取 (image_token, timestamp_us)，不依赖 lidar。
    每个 log 的 image 表已按 camera 组织，可直接得到连续帧序列。
    """
    sensor_source = get_camera_channel_sensor_data(camera_channel)
    sensor_token = _nuplan_scenario_queries.get_sensor_token(
        log_file,
        sensor_source.sensor_table,
        sensor_source.channel,
    )
    query = f"""
    WITH numbered AS
    (
        SELECT token, timestamp, ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS row_num
        FROM {sensor_source.table}
        WHERE timestamp >= ?
        AND timestamp <= ?
        AND {sensor_source.sensor_token_column} == ?
    )
    SELECT token, timestamp
    FROM numbered
    WHERE ((row_num - 1) % ?) = 0
    ORDER BY timestamp ASC;
    """
    records: List[Tuple[str, int]] = []
    for row in nuplan_query_session.execute_many(
        query,
        (min_ts, max_ts, bytearray.fromhex(sensor_token), subsample_interval),
        log_file,
    ):
        token = row["token"]
        if isinstance(token, (bytes, bytearray)):
            token = token.hex()
        records.append((str(token), int(row["timestamp"])))
    return records


def _nearest_timestamp_index(
    timestamps: List[int],
    target_timestamp: int,
    tolerance_us: int,
) -> Optional[int]:
    if not timestamps:
        return None
    pos = bisect.bisect_left(timestamps, target_timestamp)
    candidates = []
    if pos < len(timestamps):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(timestamps[i] - target_timestamp))
    if abs(timestamps[best] - target_timestamp) > tolerance_us:
        return None
    return best


def _all_cameras_have_targets(
    timestamp_by_camera: dict[str, List[int]],
    target_timestamps: List[int],
    tolerance_us: int,
) -> bool:
    for timestamps in timestamp_by_camera.values():
        for target_timestamp in target_timestamps:
            if _nearest_timestamp_index(
                timestamps,
                target_timestamp,
                tolerance_us,
            ) is None:
                return False
    return True


def _sample_target_timestamps(
    anchor_timestamp: int,
    num_history: int,
    num_future: int,
    interval_us: int,
) -> Tuple[List[int], List[int]]:
    history = [
        anchor_timestamp - (num_history - 1 - j) * interval_us
        for j in range(num_history)
    ]
    future = [
        anchor_timestamp + (j + 1) * interval_us
        for j in range(num_future)
    ]
    return history, future


def _process_one_log(
    log_path: str,
    data_root: str,
    sensor_root: str,
    cameras: List[str],
    primary_camera: Optional[str],
    anchor_stride: int,
    verbose: bool,
    num_history: int,
    num_future: int,
    interval_us: int,
    camera_time_tolerance_us: int,
) -> Tuple[dict, List[Tuple[str, int, str]], List[str]]:
    """
    处理单个 log，返回 (log_image_tokens, samples, missing_folders)。
    samples 每项为 (log_path, anchor_image_idx, primary_cam)；
    anchor_image_idx 为 primary camera 按时间排序的 image 列表中「历史末帧 / 当前 ego 时刻」的下标。
    """
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import download_file_if_necessary

    log_image_tokens: dict = {}
    samples: List[Tuple[str, int, str]] = []
    missing_folders: List[str] = []

    local_path = download_file_if_necessary(data_root, log_path, verbose=verbose)
    ts_range = _get_min_max_timestamp(local_path, table="image")
    if ts_range is None:
        return log_image_tokens, samples, missing_folders

    min_ts, max_ts = ts_range
    valid_cameras: List[str] = []
    records_by_camera: dict[str, List[Tuple[str, int]]] = {}
    timestamps_by_camera: dict[str, List[int]] = {}
    for cam_name in cameras:
        records = _get_image_records_in_time_window(
            local_path,
            cam_name,
            min_ts,
            max_ts,
            subsample_interval=1,
        )
        if not records:
            continue

        sensor_source = get_camera_channel_sensor_data(cam_name)
        img_rows = list(
            get_sensor_data_from_sensor_data_tokens_from_db(
                local_path, sensor_source, Image, [records[0][0]]
            )
        )
        if not img_rows:
            continue
        img_dir = os.path.dirname(os.path.join(sensor_root, img_rows[0].filename_jpg))
        if not os.path.isdir(img_dir):
            missing_folders.append(img_dir)
            continue

        log_image_tokens[(local_path, cam_name)] = records
        records_by_camera[cam_name] = records
        timestamps_by_camera[cam_name] = [ts for _, ts in records]
        valid_cameras.append(cam_name)

    if len(valid_cameras) != len(cameras):
        return log_image_tokens, samples, missing_folders

    if anchor_stride < 1:
        raise ValueError(f"anchor_stride must be >= 1, got {anchor_stride}")
    primary_cam = primary_camera or valid_cameras[0]
    if primary_cam not in records_by_camera:
        raise ValueError(
            f"nuplan_primary_camera={primary_cam!r} is not available in cameras={cameras}"
        )
    primary_records = records_by_camera[primary_cam]
    for anchor_image_idx in range(
        0, len(primary_records), anchor_stride
    ):
        anchor_timestamp = primary_records[anchor_image_idx][1]
        hist_ts, fut_ts = _sample_target_timestamps(
            anchor_timestamp,
            num_history,
            num_future,
            interval_us,
        )
        if _all_cameras_have_targets(
            timestamps_by_camera,
            hist_ts + fut_ts,
            camera_time_tolerance_us,
        ):
            samples.append((local_path, anchor_image_idx, primary_cam))

    return log_image_tokens, samples, missing_folders


class NuplanDevkitDataset(Dataset):
    """
    从 NuPlan log 加载相机序列与 ego 轨迹。

    - **trajectory_frame_rate_hz**：轨迹点时间间隔为 ``1/trajectory_frame_rate_hz`` 秒（默认 2Hz → 0.5s）。
    - **num_future_frames**：未来轨迹点数；与 ``future_images`` 行数一一对应（每行来自同一 image 的 ``ego_pose``）。
    - **num_history_frames**：仅作条件、不参与预测的帧数（≥1，含锚点帧）；与 ``history_images`` / ``history_trajectory`` 行数一一对应。
    - 以 primary camera 的锚点时间戳为基准，按 ``trajectory_interval_s`` 生成历史/未来目标时间戳；
      其他 camera 使用 nearest timestamp 对齐到同一目标时刻。
    - **anchor_stride**：沿时间滑窗时，相邻两个样本的 **锚点下标** 在 image token 序列上相隔多少个索引（≥1；1 表示枚举所有合法锚点）。
    - 轨迹为 ego 后轴在 **当前时刻（末帧历史）** 位姿下的局部 SE2 ``(x,y,heading)``。
    - 训练时由模型 ``forward`` 将 ``history_images`` 与 **全部** ``future_images`` 沿时间维拼接后送入 V-JEPA encoder（batch 内字段保持分开）。
    """
    def __init__(
        self,
        data_root: str,
        split: str = "mini",
        cameras: Optional[List[str]] = None,
        primary_camera: Optional[str] = None,
        train: bool = True,
        resize_to: Tuple[int, int] = (128, 128),
        anchor_stride: int = 1,
        split_ratio: float = 0.9,
        num_future_frames: int = 8,
        trajectory_frame_rate_hz: float = 2.0,
        num_history_frames: int = 1,
        num_history_image_frames: int = 0,
        num_history_trajectory_steps: int = 0,
        map_root: Optional[str] = None,
        map_version: str = "nuplan-maps-v1.1",
        verbose: bool = True,
        index_build_workers: int = 1,
        camera_time_tolerance_us: int = DEFAULT_CAMERA_TIME_TOLERANCE_US,
        nav_command_infer_mode: Optional[str] = None,
        nav_command_forward_m: Optional[float] = None,
        nav_command_lateral_m: Optional[float] = None,
        nav_command_heading_rad: Optional[float] = None,
        nav_command_combine: Optional[str] = None,
    ) -> None:
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
        self.data_root = os.path.expanduser(data_root)
        self.split = split
        self.cameras = cameras if cameras else ["CAM_F0"]
        self.primary_camera = primary_camera or ("CAM_F0" if "CAM_F0" in self.cameras else self.cameras[0])
        if self.primary_camera not in self.cameras:
            raise ValueError(
                f"primary_camera={self.primary_camera!r} must be included in cameras={self.cameras}"
            )
        self.train = train
        self.resize_to = resize_to
        self.num_future_frames = int(num_future_frames)
        if self.num_future_frames < 1:
            raise ValueError("num_future_frames 须 >= 1")
        tr = float(trajectory_frame_rate_hz)
        if tr <= 0:
            raise ValueError("trajectory_frame_rate_hz 须 > 0")
        self.trajectory_frame_rate_hz = tr
        self.trajectory_interval_s = 1.0 / tr
        self._trajectory_interval_us = int(round(self.trajectory_interval_s * 1_000_000))
        self._camera_time_tolerance_us = int(camera_time_tolerance_us)
        if self._camera_time_tolerance_us < 0:
            raise ValueError("camera_time_tolerance_us must be >= 0")

        nh_img = int(num_history_image_frames) if num_history_image_frames > 0 else int(num_history_frames)
        nh_traj = int(num_history_trajectory_steps) if num_history_trajectory_steps > 0 else int(num_history_frames)
        if nh_img < 1:
            nh_img = 1
        if nh_traj < 1:
            nh_traj = 1
        self._num_history_images = nh_img
        self._num_history_traj = nh_traj
        self._num_history = max(nh_img, nh_traj)

        self.anchor_stride = int(anchor_stride)
        if self.anchor_stride < 1:
            raise ValueError(f"anchor_stride must be >= 1, got {anchor_stride}")
        self.split_ratio = split_ratio
        self.map_version = map_version

        nuplan_v11 = os.path.join(self.data_root, "nuplan-v1.1")
        splits_dir = os.path.join(nuplan_v11, "splits", self.split)
        sensor_root = os.path.join(nuplan_v11, "sensor_blobs", self.split)
        if map_root is None:
            map_root = os.path.join(self.data_root, "maps")
        self.map_root = map_root
        if not os.path.isdir(splits_dir):
            raise FileNotFoundError(
                f"NuPlan splits 目录不存在: {splits_dir}. "
                "请设置 nuplan_data_root 为 dataset 根目录。"
            )
        if not os.path.isdir(sensor_root):
            raise FileNotFoundError(f"NuPlan sensor_blobs 不存在: {sensor_root}")

        self._sensor_root = sensor_root
        self._data_root = nuplan_v11

        # 发现所有 log 文件
        db_paths = sorted(discover_log_dbs(splits_dir))
        if not db_paths:
            raise RuntimeError(f"未发现任何 .db 文件: {splits_dir}")

        # 按 log 划分 train/val
        n_logs = len(db_paths)
        idx = int(n_logs * split_ratio)
        idx = max(1, min(n_logs - 1, idx))  # 保证 val 至少 1 个 log（n_logs>=2 时）
        if train:
            selected_paths = db_paths[:idx]
        else:
            selected_paths = db_paths[idx:]

        # 直接从 image 表按时间顺序构建 token 序列，不依赖 lidar
        self.log_image_tokens: dict = {}  # (log_path, cam_name) -> List[(image_token, timestamp_us)]
        self.log_image_timestamps: dict = {}
        self.samples: List[Tuple[str, int, str]] = []  # (log_path, anchor_image_idx, cam_name)
        self.missing_folders: List[str] = []  # 记录缺少的文件夹路径

        desc = "Building NuPlan index (image-only, per-log)"
        n_workers = max(1, index_build_workers)

        def _process_args():
            for log_path in selected_paths:
                yield (
                    log_path,
                    self._data_root,
                    self._sensor_root,
                    self.cameras,
                    self.primary_camera,
                    self.anchor_stride,
                    verbose,
                    self._num_history,
                    self.num_future_frames,
                    self._trajectory_interval_us,
                    self._camera_time_tolerance_us,
                )

        if n_workers <= 1:
            for args in tqdm(
                list(_process_args()),
                desc=desc,
                disable=not verbose,
                total=len(selected_paths),
            ):
                tokens_dict, samples_list, missing = _process_one_log(*args)
                self.log_image_tokens.update(tokens_dict)
                self.samples.extend(samples_list)
                self.missing_folders.extend(missing)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_process_one_log, *args): args[0]
                    for args in _process_args()
                }
                for future in tqdm(
                    as_completed(futures),
                    desc=desc,
                    disable=not verbose,
                    total=len(futures),
                ):
                    tokens_dict, samples_list, missing = future.result()
                    self.log_image_tokens.update(tokens_dict)
                    self.samples.extend(samples_list)
                    self.missing_folders.extend(missing)

        # Blob store（用于 load_image，与 nuplan-devkit load_image 一致）
        self._local_store = LocalStore(self._sensor_root)
        missing_info = ""
        if self.missing_folders:
            unique_missing = sorted(set(self.missing_folders))
            missing_info = f" (skipped {len(self.missing_folders)} with missing folders)"
            missing_info += f"\n  Missing folders ({len(unique_missing)}): {unique_missing[:2]}{'...' if len(unique_missing) > 2 else ''}"
        print(f"Loaded {len(self.samples)} samples" + missing_info)
        self.log_image_timestamps = {
            key: [ts for _, ts in records]
            for key, records in self.log_image_tokens.items()
        }

    def __len__(self) -> int:
        return len(self.samples)

    def preprocess_for_base(self, img):
        arr = img.as_numpy
        if arr is None:
            raise RuntimeError("图像为空")
        return cv2.resize(arr, (self.resize_to[1], self.resize_to[0]), interpolation=cv2.INTER_AREA)

    def _rgb_tensor_from_image_row(self, img_row: Any) -> torch.Tensor:
        img = load_image(img_row, self._local_store, None)
        if img is None:
            raise RuntimeError("无法加载图像")
        arr = self.preprocess_for_base(img)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    def _image_row_from_camera_time(self, log_path: str, cam_name: str, target_timestamp: int) -> Any:
        records = self.log_image_tokens[(log_path, cam_name)]
        timestamps = self.log_image_timestamps[(log_path, cam_name)]
        frame_idx = _nearest_timestamp_index(
            timestamps,
            target_timestamp,
            self._camera_time_tolerance_us,
        )
        if frame_idx is None:
            raise RuntimeError(
                f"NuPlan camera {cam_name} has no image within "
                f"{self._camera_time_tolerance_us}us of timestamp {target_timestamp}"
            )
        sensor_source = get_camera_channel_sensor_data(cam_name)
        token, _timestamp = records[frame_idx]
        rows = list(
            get_sensor_data_from_sensor_data_tokens_from_db(
                log_path, sensor_source, Image, [token]
            )
        )
        if not rows or not rows[0].ego_pose_token:
            raise RuntimeError(f"Missing ego_pose_token ({cam_name} timestamp={target_timestamp})")
        return rows[0]

    def _multicamera_tensor_at_time(self, log_path: str, target_timestamp: int, primary_cam: str) -> Tuple[torch.Tensor, Any]:
        view_tensors: List[torch.Tensor] = []
        primary_row = None
        for cam_name in self.cameras:
            row = self._image_row_from_camera_time(log_path, cam_name, target_timestamp)
            view_tensors.append(self._rgb_tensor_from_image_row(row))
            if cam_name == primary_cam:
                primary_row = row
        if primary_row is None:
            raise RuntimeError(f"Primary camera {primary_cam} is not present in {self.cameras}")
        return torch.stack(view_tensors, dim=0), primary_row

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds")
        log_path, anchor_image_idx, primary_cam = self.samples[index]

        primary_records = self.log_image_tokens[(log_path, primary_cam)]
        anchor_timestamp = primary_records[anchor_image_idx][1]
        hist_ts, fut_ts = _sample_target_timestamps(
            anchor_timestamp,
            self._num_history,
            self.num_future_frames,
            self._trajectory_interval_us,
        )

        all_hist_tensors: List[torch.Tensor] = []
        fut_tensors: List[torch.Tensor] = []
        global_hist: List[np.ndarray] = []
        global_fut: List[np.ndarray] = []

        current_ego_dynamic = None  # 末帧历史（= 锚点时刻）的瞬时速度/加速度（车身系）
        for target_ts in hist_ts:
            view_tensor, primary_row = self._multicamera_tensor_at_time(log_path, target_ts, primary_cam)
            all_hist_tensors.append(view_tensor)
            ed = ego_dynamic_from_token(log_path, primary_row.ego_pose_token)
            if target_ts == hist_ts[-1]:
                current_ego_dynamic = ed
            ep = ed.pose
            global_hist.append(np.array([ep.x, ep.y, ep.heading], dtype=np.float64))

        for target_ts in fut_ts:
            view_tensor, primary_row = self._multicamera_tensor_at_time(log_path, target_ts, primary_cam)
            fut_tensors.append(view_tensor)
            ep = ego_dynamic_from_token(log_path, primary_row.ego_pose_token).pose
            global_fut.append(np.array([ep.x, ep.y, ep.heading], dtype=np.float64))

        gh = np.stack(global_hist, axis=0)
        gf = np.stack(global_fut, axis=0)
        origin = gh[-1].copy()
        o = StateSE2(float(origin[0]), float(origin[1]), float(origin[2]))
        all_hist_traj = convert_absolute_to_relative_se2_array(o, gh)
        future_trajectory = convert_absolute_to_relative_se2_array(o, gf)

        history_images = torch.stack(all_hist_tensors[-self._num_history_images:], dim=0)
        future_images = torch.stack(fut_tensors, dim=0)
        if len(self.cameras) == 1:
            history_images = history_images[:, 0]
            future_images = future_images[:, 0]
        history_trajectory = all_hist_traj[-self._num_history_traj:]

        nav_cmd = infer_navigation_command(
            future_trajectory,
            self._nav_infer_mode,
            forward_m=self._nav_forward_m,
            lateral_threshold_m=self._nav_lateral_m,
            heading_threshold_rad=self._nav_heading_rad,
            combine=self._nav_combine,
        )
        # NuPlan 没有 unknown 指令：统一用 4 维 one-hot，但第 4 维恒为 0
        driving_cmd_oh = np.zeros(4, dtype=np.float32)
        driving_cmd_oh[min(int(nav_cmd), 2)] = 1.0
        # 瞬时车身系速度/加速度，来自 ego_pose.vx/vy/acceleration_{x,y}，
        # 与 NAVSIM ego_dynamic_state[:2]/[2:4] 同坐标系（车身后轴，前向 x、左向 y）。
        # nuPlan 官方 nuplan_scenario_queries 直接塞入 rear_axle_velocity_2d，不做旋转。
        assert current_ego_dynamic is not None, "last-history ego_dynamic not populated"
        vel = np.array([current_ego_dynamic.vx, current_ego_dynamic.vy], dtype=np.float32)
        acc = np.array([current_ego_dynamic.ax, current_ego_dynamic.ay], dtype=np.float32)
        ego_speed = float(np.linalg.norm(vel))
        ego_status = np.concatenate([driving_cmd_oh, vel, acc])

        return {
            "history_images": history_images,
            "future_images": future_images,
            "camera_names": [c.lower() for c in self.cameras],
            "history_trajectory": torch.from_numpy(history_trajectory.astype(np.float32)),
            "future_trajectory": torch.from_numpy(future_trajectory.astype(np.float32)),
            "trajectory_interval_s": float(self.trajectory_interval_s),
            "navigation_command": nav_cmd,
            "ego_speed": ego_speed,
            "ego_status": torch.from_numpy(ego_status),
            "dataset_id": 0,
            "scene_token": "",
            "index": index,
        }
