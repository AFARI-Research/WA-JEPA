import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets.nuplan_devkit_dataset import NuplanDevkitDataset
from datasets.weighted_mixture_dataset import WeightedMixtureDataset
from utils.utils import cfg_get

VALID_DATASETS = ("NUPLAN", "NAVSIM", "NAVSIM_OFFICIAL")


def _resolve_world_size_and_rank(
    num_processes: Optional[int],
    process_index: Optional[int],
) -> Tuple[int, int]:
    """Return ``(world_size, rank)`` for weighted distributed sampling.

    If ``torch.distributed`` is already initialized (typical after
    ``Accelerator()`` in training), use ``get_world_size()`` / ``get_rank()`` as
    the only source of truth. Relying on a stale ``num_processes`` or ``WORLD_SIZE``
    can make ``ceil(N/world_size)`` wrong and shrink ``len(DataLoader)`` per rank.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ws = int(torch.distributed.get_world_size())
        rk = int(torch.distributed.get_rank())
        if num_processes is not None and int(num_processes) != ws:
            warnings.warn(
                f"[MultiData] num_processes={num_processes} != "
                f"torch.distributed.get_world_size()={ws}; using get_world_size().",
                stacklevel=2,
            )
        if process_index is not None and int(process_index) != rk:
            warnings.warn(
                f"[MultiData] process_index={process_index} != "
                f"torch.distributed.get_rank()={rk}; using get_rank().",
                stacklevel=2,
            )
        return ws, rk

    ws = max(
        1,
        int(num_processes)
        if num_processes is not None
        else int(os.environ.get("WORLD_SIZE", "1")),
    )
    rk = int(
        process_index
        if process_index is not None
        else int(os.environ.get("RANK", "0"))
    )
    return ws, rk


def _navsim_paths_from_config(
    config: Optional[object],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """(train_log, train_sensor_blobs, val_log, val_sensor_blobs)；sensor 缺省回落到 log。"""
    if config is None:
        return None, None, None, None
    log = cfg_get(config, "navsim_log_path", None)
    sensor = cfg_get(config, "sensor_blobs_path", None)
    val_log = cfg_get(config, "navsim_val_log_path", None) or log
    val_sensor = cfg_get(config, "navsim_val_sensor_blobs_path", None) or sensor or log
    train_sensor = sensor or log
    return log, train_sensor, val_log, val_sensor


def _navsim_val_window_stride(config: Optional[object]) -> Any:
    w = cfg_get(config, "navsim_window_stride", None) if config else None
    vw = cfg_get(config, "navsim_val_window_stride", None) if config else None
    return vw if vw is not None else w


def _navsim_val_holdout(
    config: Optional[object],
    val_filter: Optional[str],
) -> Tuple[Optional[float], Tuple[str, str]]:
    """Resolve ``(holdout_ratio, (train_side, val_side))`` for the NAVSIM log split.

    ``navsim_val_holdout_ratio`` splits the *train* filter's logs so validation never
    touches the benchmark's test split. It only applies when train and val read the same
    filter: if a config points val at its own YAML (navtest, navmini) that file already
    defines the split, and splitting it again would silently evaluate a fraction of it.
    """
    ratio = cfg_get(config, "navsim_val_holdout_ratio", None) if config else None
    if ratio is None:
        return None, ("train", "train")
    train_filter = cfg_get(config, "train_scene_filter_yaml", None) if config else None
    if val_filter and val_filter != train_filter:
        raise ValueError(
            "navsim_val_holdout_ratio splits one filter's logs into train and val, so "
            "val_scene_filter_yaml must be empty or equal to train_scene_filter_yaml. "
            f"Got val={val_filter!r} train={train_filter!r}."
        )
    return float(ratio), ("train", "val")


def _shared_or_source(shared: dict, source: dict, key: str, default: Any) -> Any:
    """Like ``shared.get(k, source.get(k, default))`` when *shared* may bind ``k`` to ``None``."""
    if key in shared:
        return shared[key]
    return cfg_get(source, key, default)


def nav_command_kwargs_from_mapping(mapping: Optional[object]) -> Dict[str, Any]:
    """从训练 config 或 multi-dataset ``source_cfg`` 解析 ``nav_command_*`` 字段。"""
    from datasets.nav_command_infer import (
        NAV_COMMAND_COMBINE,
        NAV_COMMAND_FORWARD_M,
        NAV_COMMAND_HEADING_RAD,
        NAV_COMMAND_INFER_MODE,
        NAV_COMMAND_LATERAL_M,
    )

    def pick(key: str, default: Any) -> Any:
        if mapping is None:
            return default
        v = cfg_get(mapping, key, None)
        return default if v is None else v

    return {
        "nav_command_infer_mode": str(pick("nav_command_infer_mode", NAV_COMMAND_INFER_MODE)),
        "nav_command_forward_m": float(pick("nav_command_forward_m", NAV_COMMAND_FORWARD_M)),
        "nav_command_lateral_m": float(pick("nav_command_lateral_m", NAV_COMMAND_LATERAL_M)),
        "nav_command_heading_rad": float(pick("nav_command_heading_rad", NAV_COMMAND_HEADING_RAD)),
        "nav_command_combine": str(pick("nav_command_combine", NAV_COMMAND_COMBINE)),
    }


def _normalize_camera_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    try:
        return [str(v).lower() for v in value]
    except TypeError as exc:
        raise ValueError(f"camera_names must be a string or sequence, got {type(value).__name__}") from exc


def nuplan_collate_fn(batch: list) -> dict:
    """Stack samples; handles mixed scalar/string fields gracefully.

    Tensors with matching shapes are stacked. Variable-length tensors with the
    same trailing dimensions are padded on their first dimension and get a
    ``<key>_mask`` boolean mask. Other irregular values fall back to lists.
    """
    import torch.nn.utils.rnn as rnn_utils

    elem = batch[0]
    result: Dict[str, Any] = {}
    for key in elem:
        vals = [d[key] for d in batch]
        if isinstance(vals[0], torch.Tensor):
            try:
                result[key] = torch.stack(vals, dim=0)
            except RuntimeError:
                can_pad = (
                    vals[0].dim() > 0
                    and all(isinstance(v, torch.Tensor) for v in vals)
                    and all(v.dim() == vals[0].dim() for v in vals)
                    and all(v.shape[1:] == vals[0].shape[1:] for v in vals)
                )
                if can_pad:
                    padded = rnn_utils.pad_sequence(vals, batch_first=True, padding_value=0.0)
                    result[key] = padded
                    lengths = torch.tensor([v.shape[0] for v in vals], dtype=torch.long)
                    max_len = padded.shape[1]
                    result[f"{key}_mask"] = (
                        torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
                    )
                else:
                    result[key] = vals
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        elif isinstance(vals[0], str):
            result[key] = vals
        else:
            result[key] = vals
    return result


DEFAULT_NUM_WORKERS = 6
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_PIN_MEMORY = torch.cuda.is_available()
DEFAULT_PERSISTENT_WORKERS = False


def load_nuplan(
    data_root: str,
    resize_to: Tuple[int, int] = (128, 128),
    cameras: Optional[List[str]] = None,
    primary_camera: Optional[str] = None,
    nuplan_anchor_stride: int = 1,
    split_ratio: float = 0.9,
    nuplan_split: str = "mini",
    nuplan_index_build_workers: int = 1,
    num_future_frames: int = 8,
    trajectory_frame_rate_hz: float = 2.0,
    num_history_image_frames: int = 1,
    num_history_trajectory_steps: int = 4,
    nuplan_camera_time_tolerance_us: int = 50000,
    **nav_command_kwargs,
):
    """
    加载 NuPlan 相机与 ego 轨迹，返回 (train_dataset, val_dataset)。
    必须使用 nuplan-devkit 官方 API。
    额外关键字应仅为 ``nav_command_*``（见 ``nav_command_kwargs_from_mapping``）。
    """

    cameras = cameras or ["CAM_F0"]
    data_root = os.path.expanduser(data_root)
    train = NuplanDevkitDataset(
        data_root=data_root,
        split=nuplan_split,
        cameras=cameras,
        primary_camera=primary_camera,
        train=True,
        resize_to=resize_to,
        num_future_frames=num_future_frames,
        trajectory_frame_rate_hz=trajectory_frame_rate_hz,
        num_history_image_frames=num_history_image_frames,
        num_history_trajectory_steps=num_history_trajectory_steps,
        anchor_stride=nuplan_anchor_stride,
        split_ratio=split_ratio,
        index_build_workers=nuplan_index_build_workers,
        camera_time_tolerance_us=nuplan_camera_time_tolerance_us,
        **nav_command_kwargs,
    )
    val = NuplanDevkitDataset(
        data_root=data_root,
        split=nuplan_split,
        cameras=cameras,
        primary_camera=primary_camera,
        train=False,
        resize_to=resize_to,
        num_future_frames=num_future_frames,
        trajectory_frame_rate_hz=trajectory_frame_rate_hz,
        num_history_image_frames=num_history_image_frames,
        num_history_trajectory_steps=num_history_trajectory_steps,
        anchor_stride=nuplan_anchor_stride,
        split_ratio=split_ratio,
        index_build_workers=nuplan_index_build_workers,
        camera_time_tolerance_us=nuplan_camera_time_tolerance_us,
        **nav_command_kwargs,
    )
    return train, val


def data_loaders(
    train_data,
    val_data,
    batch_size,
    collate_fn=None,
    config: Optional[object] = None,
):
    """Build plain DataLoaders; multi-GPU sharding is applied by ``Accelerator.prepare``."""
    train_num_workers = int(cfg_get(config, "dataloader_num_workers", DEFAULT_NUM_WORKERS))
    val_default_workers = DEFAULT_NUM_WORKERS if len(val_data) > 0 else 0
    val_num_workers = int(
        cfg_get(config, "val_dataloader_num_workers", val_default_workers)
    )
    prefetch_factor = int(
        cfg_get(config, "dataloader_prefetch_factor", DEFAULT_PREFETCH_FACTOR)
    )
    pin_memory = bool(cfg_get(config, "dataloader_pin_memory", DEFAULT_PIN_MEMORY))
    persistent_workers = bool(
        cfg_get(config, "dataloader_persistent_workers", DEFAULT_PERSISTENT_WORKERS)
    )
    if len(val_data) <= 0:
        val_num_workers = 0

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and train_num_workers > 0,
        prefetch_factor=prefetch_factor if train_num_workers > 0 else None,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and val_num_workers > 0,
        prefetch_factor=prefetch_factor if val_num_workers > 0 else None,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def load_data_and_data_loaders(
    dataset: str,
    batch_size: int,
    frame_height: int = 256,
    frame_width: int = 256,
    # NuPlan data settings
    nuplan_data_root: Optional[str] = None,
    nuplan_cameras: Optional[List[str]] = None,
    nuplan_primary_camera: Optional[str] = None,
    nuplan_split: str = "mini",
    nuplan_anchor_stride: int = 1,
    nuplan_split_ratio: float = 0.9,
    # 私有集（预留）
    private_data_root: Optional[str] = None,
    config: Optional[object] = None,
    **kwargs,
):
    """
    统一入口：``dataset in VALID_DATASETS``。
    若传入 config（如训练脚本的 merged namespace），会从中补全未显式传入的数据相关参数。
    """
    nuplan_index_build_workers = 1
    num_future_frames = 8
    trajectory_frame_rate_hz = 2.0
    nuplan_camera_time_tolerance_us = 50000
    nh_img = 1
    nh_traj = 4
    nh_img_paths = None
    if config is not None:
        frame_height = int(cfg_get(config, "frame_height", frame_height))
        frame_width = int(cfg_get(config, "frame_width", frame_width))
        nuplan_data_root = cfg_get(config, "nuplan_data_root", None) or nuplan_data_root
        nuplan_cameras = cfg_get(config, "nuplan_cameras", None) or nuplan_cameras
        nuplan_primary_camera = cfg_get(config, "nuplan_primary_camera", None) or nuplan_primary_camera
        nuplan_split = cfg_get(config, "nuplan_split", nuplan_split)
        nuplan_anchor_stride = int(cfg_get(config, "nuplan_anchor_stride", nuplan_anchor_stride))
        nuplan_split_ratio = float(cfg_get(config, "nuplan_split_ratio", nuplan_split_ratio))
        nuplan_index_build_workers = int(cfg_get(config, "nuplan_index_build_workers", 1))
        nuplan_camera_time_tolerance_us = int(
            cfg_get(config, "nuplan_camera_time_tolerance_us", nuplan_camera_time_tolerance_us)
        )
        num_future_frames = int(cfg_get(config, "num_future_frames", num_future_frames))
        trajectory_frame_rate_hz = float(cfg_get(config, "trajectory_frame_rate_hz", trajectory_frame_rate_hz))
        nh_img = int(cfg_get(config, "num_history_image_frames", cfg_get(config, "num_history_frames", nh_img)))
        nh_traj = int(cfg_get(config, "num_history_trajectory_steps", cfg_get(config, "num_history_frames", nh_traj)))
        nh_img_paths = cfg_get(config, "num_history_image_paths", None)
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Invalid dataset: '{dataset}'. Supported: {', '.join(VALID_DATASETS)}."
        )

    resize_to = (frame_height, frame_width)
    nav_kw = nav_command_kwargs_from_mapping(config)

    if dataset == "NUPLAN":
        if not (nuplan_data_root and os.path.isdir(os.path.expanduser(nuplan_data_root))):
            raise FileNotFoundError(
                "NUPLAN requires nuplan_data_root in config, "
                "pointing to the dataset root (containing nuplan-v1.1/ and maps/)."
            )
        configured_camera_names = _normalize_camera_names(cfg_get(config, "camera_names", None))
        dataset_camera_names = _normalize_camera_names(nuplan_cameras or ["CAM_F0"])
        if configured_camera_names and configured_camera_names != dataset_camera_names:
            raise ValueError(
                "For NUPLAN multi-camera training, data.camera_names must match "
                "nuplan_cameras after lowercasing and in the same order. "
                f"got camera_names={configured_camera_names} nuplan_cameras={dataset_camera_names}"
            )
        training_data, validation_data = load_nuplan(
            data_root=nuplan_data_root,
            resize_to=resize_to,
            cameras=nuplan_cameras or ["CAM_F0"],
            primary_camera=nuplan_primary_camera,
            nuplan_anchor_stride=nuplan_anchor_stride,
            split_ratio=nuplan_split_ratio,
            nuplan_split=nuplan_split,
            nuplan_index_build_workers=nuplan_index_build_workers,
            num_future_frames=num_future_frames,
            trajectory_frame_rate_hz=trajectory_frame_rate_hz,
            num_history_image_frames=nh_img,
            num_history_trajectory_steps=nh_traj,
            nuplan_camera_time_tolerance_us=nuplan_camera_time_tolerance_us,
            **nav_kw,
        )
    elif dataset == "NAVSIM":
        from datasets.navsim_dataset import NavsimDataset

        navsim_log_path, train_sensor, navsim_val_log_path, navsim_val_sensor_blobs_path = (
            _navsim_paths_from_config(config)
        )
        if not navsim_log_path:
            raise FileNotFoundError("NAVSIM requires navsim_log_path in config")
        train_filter = cfg_get(config, "train_scene_filter_yaml", None)
        val_filter = cfg_get(config, "val_scene_filter_yaml", None)
        navsim_window_stride = cfg_get(config, "navsim_window_stride", None) if config else None
        val_navsim_stride = _navsim_val_window_stride(config)
        training_data = NavsimDataset(
            navsim_log_path=navsim_log_path,
            sensor_blobs_path=train_sensor,
            num_history_image_frames=nh_img,
            num_history_trajectory_steps=nh_traj,
            num_future_frames=num_future_frames,
            resize_to=resize_to,
            scene_filter_yaml=train_filter,
            window_stride=navsim_window_stride,
            **nav_kw,
        )
        validation_data = NavsimDataset(
            navsim_log_path=navsim_val_log_path,
            sensor_blobs_path=navsim_val_sensor_blobs_path,
            num_history_image_frames=nh_img,
            num_history_trajectory_steps=nh_traj,
            num_future_frames=num_future_frames,
            resize_to=resize_to,
            scene_filter_yaml=val_filter or train_filter,
            window_stride=val_navsim_stride,
            **nav_kw,
        )
    elif dataset == "NAVSIM_OFFICIAL":
        from datasets.navsim_official_dataset import NavsimOfficialDataset

        navsim_log_path, train_sensor, navsim_val_log_path, navsim_val_sensor_blobs_path = (
            _navsim_paths_from_config(config)
        )
        if not navsim_log_path:
            raise FileNotFoundError("NAVSIM_OFFICIAL requires navsim_log_path in config")
        train_filter = cfg_get(config, "train_scene_filter_yaml", None)
        val_filter = cfg_get(config, "val_scene_filter_yaml", None)
        if not train_filter:
            raise FileNotFoundError(
                "NAVSIM_OFFICIAL requires train_scene_filter_yaml "
                "(NAVSIM devkit SceneFilter YAML, e.g. navtrain.yaml)"
            )
        holdout_ratio, holdout_sides = _navsim_val_holdout(config, val_filter)
        training_data = NavsimOfficialDataset(
            navsim_log_path=navsim_log_path,
            sensor_blobs_path=train_sensor,
            scene_filter_yaml=train_filter,
            original_sensor_path=cfg_get(config, "original_sensor_path", None),
            synthetic_sensor_path=cfg_get(config, "synthetic_sensor_path", None),
            synthetic_scenes_path=cfg_get(config, "synthetic_scenes_path", None),
            num_history_image_frames=nh_img,
            num_history_image_paths=nh_img_paths,
            num_history_trajectory_steps=nh_traj,
            num_future_frames=num_future_frames,
            resize_to=resize_to,
            camera_name=str(cfg_get(config, "camera_name", "cam_f0")),
            camera_names=cfg_get(config, "camera_names", None),
            log_holdout_ratio=holdout_ratio,
            log_holdout_side=holdout_sides[0],
            **nav_kw,
        )
        validation_data = NavsimOfficialDataset(
            navsim_log_path=navsim_val_log_path,
            sensor_blobs_path=navsim_val_sensor_blobs_path,
            scene_filter_yaml=val_filter or train_filter,
            original_sensor_path=cfg_get(
                config,
                "navsim_val_original_sensor_path",
                cfg_get(config, "original_sensor_path", None),
            ),
            synthetic_sensor_path=cfg_get(
                config,
                "navsim_val_synthetic_sensor_path",
                cfg_get(config, "synthetic_sensor_path", None),
            ),
            synthetic_scenes_path=cfg_get(
                config,
                "navsim_val_synthetic_scenes_path",
                cfg_get(config, "synthetic_scenes_path", None),
            ),
            num_history_image_frames=nh_img,
            num_history_image_paths=nh_img_paths,
            num_history_trajectory_steps=nh_traj,
            num_future_frames=num_future_frames,
            resize_to=resize_to,
            camera_name=str(cfg_get(config, "camera_name", "cam_f0")),
            camera_names=cfg_get(config, "camera_names", None),
            log_holdout_ratio=holdout_ratio,
            log_holdout_side=holdout_sides[1],
            **nav_kw,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size,
        collate_fn=nuplan_collate_fn,
        config=config,
    )

    return training_data, validation_data, training_loader, validation_loader


# ---------------------------------------------------------------------------
# Multi-dataset infrastructure
# ---------------------------------------------------------------------------

def _build_single_source(source_cfg: dict, shared_cfg: dict) -> Tuple[Any, Any]:
    """Build (train_dataset, val_dataset) for one data source config."""
    src_type = str(cfg_get(source_cfg, "type", "NUPLAN")).upper()
    h = int(_shared_or_source(shared_cfg, source_cfg, "frame_height", 256))
    w = int(_shared_or_source(shared_cfg, source_cfg, "frame_width", 512))
    nf = int(_shared_or_source(shared_cfg, source_cfg, "num_future_frames", 8))
    nh_img = int(_shared_or_source(shared_cfg, source_cfg, "num_history_image_frames", 1))
    nh_traj = int(_shared_or_source(shared_cfg, source_cfg, "num_history_trajectory_steps", 4))
    nh_img_paths = _shared_or_source(shared_cfg, source_cfg, "num_history_image_paths", None)
    hz = float(_shared_or_source(shared_cfg, source_cfg, "trajectory_frame_rate_hz", 2.0))
    nuplan_cameras = _shared_or_source(shared_cfg, source_cfg, "nuplan_cameras", ["CAM_F0"])
    nuplan_primary_camera = _shared_or_source(shared_cfg, source_cfg, "nuplan_primary_camera", None)
    nuplan_camera_time_tolerance_us = int(
        _shared_or_source(shared_cfg, source_cfg, "nuplan_camera_time_tolerance_us", 50000)
    )

    if src_type == "NUPLAN":
        nuplan_anchor_stride = int(cfg_get(source_cfg, "nuplan_anchor_stride", 1))
        nav_kw = nav_command_kwargs_from_mapping(source_cfg)
        train_ds = NuplanDevkitDataset(
            data_root=source_cfg["nuplan_data_root"],
            split=cfg_get(source_cfg, "nuplan_split", "train"),
            cameras=nuplan_cameras,
            primary_camera=nuplan_primary_camera,
            train=True,
            resize_to=(h, w),
            num_future_frames=nf,
            trajectory_frame_rate_hz=hz,
            num_history_image_frames=nh_img,
            num_history_trajectory_steps=nh_traj,
            anchor_stride=nuplan_anchor_stride,
            split_ratio=float(cfg_get(source_cfg, "split_ratio", 0.9)),
            index_build_workers=int(cfg_get(source_cfg, "index_build_workers", 1)),
            camera_time_tolerance_us=nuplan_camera_time_tolerance_us,
            **nav_kw,
        )
        val_ds = NuplanDevkitDataset(
            data_root=source_cfg["nuplan_data_root"],
            split=cfg_get(source_cfg, "nuplan_split", "train"),
            cameras=nuplan_cameras,
            primary_camera=nuplan_primary_camera,
            train=False,
            resize_to=(h, w),
            num_future_frames=nf,
            trajectory_frame_rate_hz=hz,
            num_history_image_frames=nh_img,
            num_history_trajectory_steps=nh_traj,
            anchor_stride=nuplan_anchor_stride,
            split_ratio=float(cfg_get(source_cfg, "split_ratio", 0.9)),
            index_build_workers=int(cfg_get(source_cfg, "index_build_workers", 1)),
            camera_time_tolerance_us=nuplan_camera_time_tolerance_us,
            **nav_kw,
        )
        return train_ds, val_ds

    elif src_type == "NAVSIM":
        from datasets.navsim_dataset import NavsimDataset

        log_names = cfg_get(source_cfg, "log_names", None)
        if isinstance(log_names, list):
            log_names = set(log_names)
        scene_tokens = cfg_get(source_cfg, "scene_tokens", None)
        if isinstance(scene_tokens, list):
            scene_tokens = set(scene_tokens)

        train_filter = cfg_get(source_cfg, "train_scene_filter_yaml", None)
        val_filter = cfg_get(source_cfg, "val_scene_filter_yaml", None)
        navsim_window_stride = cfg_get(source_cfg, "navsim_window_stride", None)
        navsim_val_window_stride = cfg_get(source_cfg, "navsim_val_window_stride", None)
        val_navsim_stride = (
            navsim_val_window_stride
            if navsim_val_window_stride is not None
            else navsim_window_stride
        )

        nav_kw = nav_command_kwargs_from_mapping(source_cfg)
        train_ds = NavsimDataset(
            navsim_log_path=source_cfg["navsim_log_path"],
            sensor_blobs_path=source_cfg["sensor_blobs_path"],
            num_history_image_frames=nh_img,
            num_history_trajectory_steps=nh_traj,
            num_future_frames=nf,
            resize_to=(h, w),
            camera_name=cfg_get(source_cfg, "camera_name", "cam_f0"),
            log_names=log_names,
            scene_tokens=scene_tokens,
            scene_filter_yaml=train_filter,
            window_stride=navsim_window_stride,
            **nav_kw,
        )
        val_ds = NavsimDataset(
            navsim_log_path=cfg_get(source_cfg, "navsim_val_log_path", source_cfg["navsim_log_path"]),
            sensor_blobs_path=cfg_get(
                source_cfg,
                "navsim_val_sensor_blobs_path",
                source_cfg["sensor_blobs_path"],
            ),
            num_history_image_frames=nh_img,
            num_history_trajectory_steps=nh_traj,
            num_future_frames=nf,
            resize_to=(h, w),
            camera_name=cfg_get(source_cfg, "camera_name", "cam_f0"),
            scene_filter_yaml=val_filter or train_filter,
            window_stride=val_navsim_stride,
            **nav_kw,
        )
        return train_ds, val_ds

    elif src_type == "NAVSIM_OFFICIAL":
        from datasets.navsim_official_dataset import NavsimOfficialDataset

        train_filter = cfg_get(source_cfg, "train_scene_filter_yaml", None)
        if not train_filter:
            raise ValueError(
                "NAVSIM_OFFICIAL source requires train_scene_filter_yaml "
                "(path to navsim SceneFilter YAML)"
            )
        val_filter = cfg_get(source_cfg, "val_scene_filter_yaml", None)
        nav_kw = nav_command_kwargs_from_mapping(source_cfg)
        train_ds = NavsimOfficialDataset(
            navsim_log_path=source_cfg["navsim_log_path"],
            sensor_blobs_path=source_cfg["sensor_blobs_path"],
            scene_filter_yaml=train_filter,
            original_sensor_path=cfg_get(source_cfg, "original_sensor_path", None),
            synthetic_sensor_path=cfg_get(source_cfg, "synthetic_sensor_path", None),
            synthetic_scenes_path=cfg_get(source_cfg, "synthetic_scenes_path", None),
            num_history_image_frames=nh_img,
            num_history_image_paths=nh_img_paths,
            num_history_trajectory_steps=nh_traj,
            num_future_frames=nf,
            resize_to=(h, w),
            camera_name=cfg_get(source_cfg, "camera_name", "cam_f0"),
            camera_names=cfg_get(source_cfg, "camera_names", None),
            **nav_kw,
        )
        val_ds = NavsimOfficialDataset(
            navsim_log_path=cfg_get(source_cfg, "navsim_val_log_path", source_cfg["navsim_log_path"]),
            sensor_blobs_path=cfg_get(
                source_cfg,
                "navsim_val_sensor_blobs_path",
                source_cfg["sensor_blobs_path"],
            ),
            scene_filter_yaml=val_filter or train_filter,
            original_sensor_path=cfg_get(
                source_cfg,
                "navsim_val_original_sensor_path",
                cfg_get(source_cfg, "original_sensor_path", None),
            ),
            synthetic_sensor_path=cfg_get(
                source_cfg,
                "navsim_val_synthetic_sensor_path",
                cfg_get(source_cfg, "synthetic_sensor_path", None),
            ),
            synthetic_scenes_path=cfg_get(
                source_cfg,
                "navsim_val_synthetic_scenes_path",
                cfg_get(source_cfg, "synthetic_scenes_path", None),
            ),
            num_history_image_frames=nh_img,
            num_history_image_paths=nh_img_paths,
            num_history_trajectory_steps=nh_traj,
            num_future_frames=nf,
            resize_to=(h, w),
            camera_name=cfg_get(source_cfg, "camera_name", "cam_f0"),
            camera_names=cfg_get(source_cfg, "camera_names", None),
            **nav_kw,
        )
        return train_ds, val_ds

    else:
        raise ValueError(f"Unknown dataset type: {src_type}")


def _resolve_sampling_weights(
    train_datasets: List[Any],
    source_names: List[str],
    sampling_cfg: dict,
) -> Dict[str, float]:
    mode = str(cfg_get(sampling_cfg, "mode", "proportional")).strip().lower()
    explicit_weights = cfg_get(sampling_cfg, "weights", {}) or {}

    if mode == "uniform":
        weights = {name: 1.0 for name in source_names}
    elif mode == "weights":
        missing = [name for name in source_names if name not in explicit_weights]
        if missing:
            raise ValueError(
                "sampling.mode=weights requires entries for every source. "
                f"Missing: {missing}"
            )
        weights = {name: float(explicit_weights[name]) for name in source_names}
    elif mode == "proportional":
        weights = {
            name: max(float(len(ds)), 1.0)
            for name, ds in zip(source_names, train_datasets)
        }
    else:
        raise ValueError(
            f"Unknown sampling.mode: {mode}. Supported: proportional, uniform, weights"
        )

    if any(weight <= 0 for weight in weights.values()):
        raise ValueError(f"Sampling weights must be positive, got: {weights}")
    return weights


def _make_mixture_loader(
    datasets: List[Any],
    weight_names: List[str],
    normalized_weights: Dict[str, float],
    batch_size: int,
    *,
    drop_last: bool,
    samples_per_epoch: Optional[int],
    num_processes: Optional[int],
    process_index: Optional[int],
    loader_cfg: Optional[object] = None,
    seed: int = 42,
) -> Tuple[WeightedMixtureDataset, DataLoader, int]:
    """``WeightedMixtureDataset``：权重抽样在 Dataset 内完成；DataLoader 可与 ``Accelerator.prepare`` 正常配合。"""
    if len(datasets) != len(weight_names):
        raise ValueError("datasets and weight_names length mismatch")
    world_size, rank = _resolve_world_size_and_rank(num_processes, process_index)

    total_concat_len = sum(len(d) for d in datasets)
    if total_concat_len == 0:
        raise ValueError("Combined dataset is empty")

    target_samples = max(int(samples_per_epoch or total_concat_len), batch_size)
    num_samples_per_rank = max(1, int(math.ceil(target_samples / max(world_size, 1))))

    mixture = WeightedMixtureDataset(
        datasets=datasets,
        source_names=weight_names,
        normalized_weights=normalized_weights,
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        total_concat_len=total_concat_len,
        world_size=world_size,
        seed=int(seed),
    )

    nw = int(cfg_get(loader_cfg, "dataloader_num_workers", DEFAULT_NUM_WORKERS))
    prefetch_factor = int(
        cfg_get(loader_cfg, "dataloader_prefetch_factor", DEFAULT_PREFETCH_FACTOR)
    )
    pin_memory = bool(cfg_get(loader_cfg, "dataloader_pin_memory", DEFAULT_PIN_MEMORY))
    persistent_workers = bool(
        cfg_get(loader_cfg, "dataloader_persistent_workers", DEFAULT_PERSISTENT_WORKERS)
    )
    loader = DataLoader(
        mixture,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and nw > 0,
        prefetch_factor=prefetch_factor if nw > 0 else None,
        drop_last=drop_last,
        collate_fn=nuplan_collate_fn,
    )
    if rank == 0:
        dist_ok = torch.distributed.is_available() and torch.distributed.is_initialized()
        ws_env = os.environ.get("WORLD_SIZE", "")
        lw_env = os.environ.get("LOCAL_WORLD_SIZE", "")
        approx_loader_len = num_samples_per_rank // max(batch_size, 1)
        print(
            "  [MultiData] mixture_loader: "
            f"world_size={world_size} (dist_init={dist_ok}, env WORLD_SIZE={ws_env!r} LOCAL_WORLD_SIZE={lw_env!r}) "
            f"concat_len={total_concat_len} target_samples={target_samples} "
            f"virtual_len={len(mixture)} num_samples_per_rank={num_samples_per_rank} "
            f"loader_len={len(loader)} (formula_len={approx_loader_len}) "
            f"batch_size={batch_size} drop_last={drop_last}"
        )
    return mixture, loader, num_samples_per_rank


def load_mixed_datasets(
    data_cfg: dict,
    batch_size_per_gpu: int = 2,
    *,
    num_processes: Optional[int] = None,
    process_index: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Build multi-dataset training objects for weighted mixed-dataset training.

    Train and val both use ``all_train`` / ``all_val`` with the same ``source_names``
    and ``normalized_weights``, via :func:`_make_mixture_loader`
    (train ``drop_last=True``, val ``False``). Combined validation must be non-empty.
    Config: ``sampling.samples_per_epoch`` / ``val_samples_per_epoch``.
    *seed* feeds :class:`~datasets.weighted_mixture_dataset.WeightedMixtureDataset`
    (use the same value as training config ``seed``).

    *batch_size_per_gpu* applies to train and val loaders.

    Pass ``num_processes`` / ``process_index`` from :class:`accelerate.Accelerator` for
    logging; after ``torch.distributed`` is initialized, :func:`_resolve_world_size_and_rank`
    uses ``get_world_size()`` / ``get_rank()`` only (see :func:`_make_mixture_loader`)。

    加权抽样在 :class:`~datasets.weighted_mixture_dataset.WeightedMixtureDataset`
    内部完成；``DataLoader`` 可走 ``Accelerator.prepare``，由 Accelerate 注入DDP用的 ``DistributedSampler``
    （仅划分 ``idx``，与各 rank 上有放回 mixture 语义相容）。
    """
    sources_cfg = cfg_get(data_cfg, "sources", {})
    if not sources_cfg:
        raise ValueError("data.sources is empty — define at least one source")

    shared_keys = {
        "frame_height", "frame_width", "num_future_frames",
        "num_history_image_frames", "num_history_trajectory_steps",
        "trajectory_frame_rate_hz",
        "nuplan_cameras", "nuplan_primary_camera", "nuplan_camera_time_tolerance_us",
    }
    shared_cfg = {k: v for k, v in data_cfg.items() if k in shared_keys}

    sampling_cfg = cfg_get(data_cfg, "sampling", {}) or {}

    all_train, all_val = [], []
    source_names: List[str] = []

    bs = int(batch_size_per_gpu)

    for name, src_cfg in sources_cfg.items():
        source_names.append(name)
        train_ds, val_ds = _build_single_source(src_cfg, shared_cfg)
        all_train.append(train_ds)
        all_val.append(val_ds)

        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"  [MultiData] source={name} type={cfg_get(src_cfg, 'type', '?')} "
                f"train={len(train_ds)} val={len(val_ds)} bs={bs}"
            )

    source_weights = _resolve_sampling_weights(
        train_datasets=all_train,
        source_names=source_names,
        sampling_cfg=sampling_cfg,
    )
    total_weight = sum(source_weights.values())
    normalized_weights = {
        name: weight / total_weight for name, weight in source_weights.items()
    }

    _, train_loader, _ = _make_mixture_loader(
        all_train,
        source_names,
        normalized_weights,
        bs,
        drop_last=True,
        samples_per_epoch=cfg_get(sampling_cfg, "samples_per_epoch", None),
        num_processes=num_processes,
        process_index=process_index,
        loader_cfg=data_cfg,
        seed=seed,
    )

    _, val_loader, _ = _make_mixture_loader(
        all_val,
        source_names,
        normalized_weights,
        bs,
        drop_last=False,
        samples_per_epoch=cfg_get(sampling_cfg, "val_samples_per_epoch", None),
        num_processes=num_processes,
        process_index=process_index,
        loader_cfg=data_cfg,
        seed=seed,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "source_names": source_names,
        "source_weights": normalized_weights,
    }


def visualize_reconstruction(
    input_frames: torch.Tensor,
    pred_frames: torch.Tensor,
    save_path: Optional[str] = None,
    num_samples: int = 4,
    resolution: Optional[Tuple[int, int]] = None,
    sample_ids: Optional[object] = None,  # list[int] or tensor
    text_labels: Optional[List[str]] = None,
    future_label: str = "pred",
    gt_frames: Optional[torch.Tensor] = None,
    compare_label: str = "gt",
) -> None:
    """
    对每个样本单独出一张图：**自上而下** 排列：先全部输入帧，再未来帧。
    若未传 ``gt_frames``，未来帧每行一幅（仅 ``pred_frames``）。
    若传入 ``gt_frames``，未来段每一时间步 **一行两列**：左 ``pred``、右 ``gt``，时间步一一对应。
    数据为 [-1,1]，显示前转到 [0,1]。文件名由 ``save_path`` 推导：
    ``<stem>_id<dataset_index>.<ext>``；若无 ``sample_ids`` 则用 ``<stem>_sample<j>.<ext>``。

    Args:
        input_frames: [B, T, C, H, W]，T>=1。
        pred_frames: [B, Tp, C, H, W]，Tp>=1。
        save_path: 保存路径模板（含扩展名）；为 None 时对前 ``num_samples`` 个样本依次 ``plt.show()``。
        num_samples: 最多导出 batch 内前多少个样本。
        resolution: 若为 ``(宽, 高)`` 像素，则按与原先相同的规则换算 inch；若为 ``None``（默认），则按
            张量中的 H、W 与行数自适应画布，不依赖配置。
        sample_ids: 各样本在 Dataset 中的 index，与 batch 维对齐。
        text_labels: ??????????????????
        future_label: 未来帧（或对比模式下左侧 pred）行的标题前缀。
        gt_frames: 可选，[B, Tp, C, H, W]，与 ``pred_frames`` 时间步数相同；传入则 pred/gt 左右同排。
        compare_label: 对比模式下右侧列标题前缀（默认 ``gt``）。
    """
    inp = input_frames.detach()
    prd = pred_frames.detach()
    if inp.dim() == 4:
        inp = inp.unsqueeze(1)
    if prd.dim() == 4:
        prd = prd.unsqueeze(1)

    B, T_in, _, _, _ = inp.shape
    if T_in < 1:
        raise ValueError(f"input_frames needs T>=1, got T={T_in}")
    if prd.shape[0] != B:
        raise ValueError(f"pred_frames batch {prd.shape[0]} != input_frames batch {B}")
    T_pred = prd.shape[1]
    if T_pred < 1:
        raise ValueError("pred_frames must have at least 1 frame")

    gw: Optional[torch.Tensor] = None
    if gt_frames is not None:
        gw = gt_frames.detach()
        if gw.dim() == 4:
            gw = gw.unsqueeze(1)
        if gw.shape[0] != B:
            raise ValueError(f"gt_frames batch {gw.shape[0]} != input_frames batch {B}")
        T_gt = gw.shape[1]
        if T_gt != T_pred:
            raise ValueError(f"gt_frames T={T_gt} must match pred_frames T={T_pred}")
    compare_mode = gw is not None

    n_export = min(int(num_samples), B)
    if n_export <= 0:
        return

    inp = inp[:n_export]
    prd = prd[:n_export]

    input_frames = inp.to("cpu", dtype=torch.float32)
    pred_frames = prd.to("cpu", dtype=torch.float32)
    gt_t: Optional[torch.Tensor] = None
    if gw is not None:
        gt_t = gw[:n_export].to("cpu", dtype=torch.float32)
        gt_t = gt_t.mul(0.5).add(0.5).clamp(0.0, 1.0)

    input_frames = input_frames.mul(0.5).add(0.5).clamp(0.0, 1.0)
    pred_frames = pred_frames.mul(0.5).add(0.5).clamp(0.0, 1.0)

    ids_list: Optional[list] = None
    if sample_ids is not None:
        if isinstance(sample_ids, torch.Tensor):
            ids_list = sample_ids.detach()[:n_export].flatten().cpu().tolist()
        elif isinstance(sample_ids, (list, tuple)):
            ids_list = list(sample_ids[:n_export])

    n_panels = T_in + T_pred
    dpi = 100
    ih, iw = int(input_frames.shape[-2]), int(input_frames.shape[-1])

    has_text = text_labels is not None and len(text_labels) > 0
    text_height_in = 1.2 if has_text else 0.0

    if resolution is not None:
        w_px, h_px = int(resolution[0]), int(resolution[1])
        ref_rows = 3
        fig_w_in = max(w_px / dpi, 0.1)
        if compare_mode:
            fig_w_in = max(fig_w_in * 1.9, 3.2)
        fig_h_in = (h_px / dpi) * max(n_panels / ref_rows, 0.45) + text_height_in
    else:
        row_img_in = max(ih / dpi, 0.75)
        row_title_in = 0.36
        fig_w_in = max(iw / dpi + 0.2, 2.0)
        if compare_mode:
            fig_w_in = max(fig_w_in * 1.95, 3.2)
        fig_h_in = (row_img_in + row_title_in) * n_panels + 0.45 + text_height_in
        max_side_in = 40.0
        if fig_w_in > max_side_in or fig_h_in > max_side_in:
            s = min(max_side_in / fig_w_in, max_side_in / fig_h_in)
            fig_w_in *= s
            fig_h_in *= s

    titles: List[str] = [f"frame {t + 1}" for t in range(T_in)]
    if not compare_mode:
        if T_pred == 1:
            titles.append(future_label)
        else:
            titles.extend([f"{future_label} {k + 1}" for k in range(T_pred)])

    def _pred_title(k: int) -> str:
        if T_pred == 1:
            return future_label
        return f"{future_label} {k + 1}"

    def _gt_title(k: int) -> str:
        if T_pred == 1:
            return compare_label
        return f"{compare_label} {k + 1}"

    for j in range(n_export):
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), layout="constrained")
        gs = fig.add_gridspec(n_panels, 2, hspace=0.22, wspace=0.06)

        for idx in range(n_panels):
            if idx < T_in:
                ax = fig.add_subplot(gs[idx, :])
                img_t = input_frames[j, idx]
                ax.imshow(img_t.permute(1, 2, 0).contiguous().numpy(), aspect="auto")
                ax.axis("off")
                ax.set_title(
                    titles[idx],
                    loc="left",
                    fontsize=11,
                    fontweight="medium",
                    pad=8,
                )
            elif gt_t is not None:
                k = idx - T_in
                ax_l = fig.add_subplot(gs[idx, 0])
                ax_r = fig.add_subplot(gs[idx, 1])
                ax_l.imshow(pred_frames[j, k].permute(1, 2, 0).contiguous().numpy(), aspect="auto")
                ax_l.axis("off")
                ax_l.set_title(_pred_title(k), loc="left", fontsize=11, fontweight="medium", pad=8)
                ax_r.imshow(gt_t[j, k].permute(1, 2, 0).contiguous().numpy(), aspect="auto")
                ax_r.axis("off")
                ax_r.set_title(_gt_title(k), loc="left", fontsize=11, fontweight="medium", pad=8)
            else:
                ax = fig.add_subplot(gs[idx, :])
                img_t = pred_frames[j, idx - T_in]
                ax.imshow(img_t.permute(1, 2, 0).contiguous().numpy(), aspect="auto")
                ax.axis("off")
                ax.set_title(
                    titles[idx],
                    loc="left",
                    fontsize=11,
                    fontweight="medium",
                    pad=8,
                )

        id_part = ids_list[j] if ids_list is not None and j < len(ids_list) else None
        if id_part is not None:
            fig.suptitle(
                f"dataset id = {id_part}",
                fontsize=12,
                fontweight="bold",
            )

        if (
            text_labels is not None
            and j < len(text_labels)
            and text_labels[j]
        ):
            label = text_labels[j]
            if len(label) > 300:
                label = label[:297] + "..."
            fig.text(
                0.02, 0.005, label,
                fontsize=6, family="monospace",
                verticalalignment="bottom",
                wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.85),
            )
        try:
            if save_path:
                root, ext = os.path.splitext(save_path)
                if not ext:
                    root, ext = save_path, ".png"
                out_dir = os.path.dirname(root)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                if id_part is not None:
                    out_path = f"{root}_id{id_part}{ext}"
                else:
                    out_path = f"{root}_sample{j}{ext}"
                plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
            else:
                plt.show()
        finally:
            plt.close(fig)
