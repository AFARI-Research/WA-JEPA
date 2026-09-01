"""
Ego 轨迹：全局 StateSE2 → 以当前时刻（末帧历史）为原点的局部 (x, y, heading)。

与常见 nuPlan/NavSim 训练中的相对位姿约定一致。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.database.nuplan_db.query_session import execute_one
from pyquaternion import Quaternion


def convert_absolute_to_relative_se2_array(
    origin: StateSE2, state_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """全局 SE2 数组 → 以 origin 为参考的局部坐标。"""
    if state_se2_array.shape[-1] != 3:
        raise ValueError(f"expected last dim 3, got shape {state_se2_array.shape}")
    theta = -origin.heading
    origin_array = np.array([[origin.x, origin.y, origin.heading]], dtype=np.float64)

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points_rel = state_se2_array - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ rot.T
    points_rel[..., 2] = np.arctan2(np.sin(points_rel[..., 2]), np.cos(points_rel[..., 2]))
    return points_rel


class EgoDynamic(NamedTuple):
    """Ego 位姿 + 车身后轴坐标系下的瞬时 (vx, vy, ax, ay)。

    坐标系与 NAVSIM ``ego_dynamic_state[:2]/[2:4]`` 严格一致（均为车身后轴系，
    前向 x、左向 y）。nuPlan ``ego_pose`` 表的 ``vx/vy/acceleration_x/acceleration_y``
    本身就是车身系，无需旋转 —— 官方 ``nuplan_scenario_queries`` 直接把它们塞入
    ``rear_axle_velocity_2d`` / ``rear_axle_acceleration_2d``。
    """

    pose: StateSE2
    vx: float
    vy: float
    ax: float
    ay: float


def ego_dynamic_from_token(log_file: str, ego_pose_token_hex: str) -> EgoDynamic:
    """从 ego_pose 表一次性读取 (x, y, yaw, vx, vy, ax, ay)。"""
    if not ego_pose_token_hex:
        raise ValueError("ego_pose_token 为空")
    row = execute_one(
        "SELECT x, y, qw, qx, qy, qz, vx, vy, acceleration_x, acceleration_y "
        "FROM ego_pose WHERE token = ?",
        (bytearray.fromhex(ego_pose_token_hex),),
        log_file,
    )
    if row is None:
        raise RuntimeError(f"ego_pose 表中无 token: {ego_pose_token_hex[:16]}...")
    q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
    yaw = float(q.yaw_pitch_roll[0])
    return EgoDynamic(
        pose=StateSE2(float(row["x"]), float(row["y"]), yaw),
        vx=float(row["vx"]),
        vy=float(row["vy"]),
        ax=float(row["acceleration_x"]),
        ay=float(row["acceleration_y"]),
    )
