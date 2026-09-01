"""从 ego 相对 future 轨迹几何推断 navigation command（左 / 直 / 右）。

与 ``scripts/evaluation/verify_navsim_command_geometry.py`` 使用同一套规则；
训练管线（``NavsimDataset`` / ``NuplanDevkitDataset``）在缺少 GT 标签时用本模块推断条件。

**默认超参**（``NAV_COMMAND_*``）来自对 NAVSIM 的离线网格搜索与 navtest 复验，见仓库根目录 ``README.md`` 中「Navigation command 几何启发式」一节。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# --- 与 README 声明一致的默认（navtrain 网格 + navtest Top-K 复验后采用） ---
NAV_COMMAND_INFER_MODE = "arc_length"
NAV_COMMAND_FORWARD_M = 20.0
NAV_COMMAND_LATERAL_M = 2.5
NAV_COMMAND_HEADING_RAD = 0.15
NAV_COMMAND_COMBINE = "and"


def decide_command(
    y: float,
    h: float,
    lateral_m: float,
    head_rad: float,
    combine: str,
) -> int:
    """由 (横向 y, 航向 h) 与阈值得到 0 左 / 1 直 / 2 右。"""
    c = str(combine).strip().lower()
    if c == "or":
        if h > head_rad or y > lateral_m:
            return 0
        if h < -head_rad or y < -lateral_m:
            return 2
        return 1
    if c == "and":
        if h > head_rad and y > lateral_m:
            return 0
        if h < -head_rad and y < -lateral_m:
            return 2
        return 1
    if c == "lat_only":
        if y > lateral_m:
            return 0
        if y < -lateral_m:
            return 2
        return 1
    if c == "heading_only":
        if h > head_rad:
            return 0
        if h < -head_rad:
            return 2
        return 1
    raise ValueError(
        f"Unknown combine: {combine!r} (use or, and, lat_only, heading_only)"
    )


def infer_nav_command_endpoint(
    future_trajectory: np.ndarray,
    lateral_threshold_m: float = NAV_COMMAND_LATERAL_M,
    heading_threshold_rad: float = NAV_COMMAND_HEADING_RAD,
    combine: str = NAV_COMMAND_COMBINE,
) -> int:
    ft = np.asarray(future_trajectory, dtype=np.float64)
    if ft.size == 0:
        return 1
    final_heading = float(ft[-1, 2])
    final_y = float(ft[-1, 1])
    return decide_command(
        final_y, final_heading, lateral_threshold_m, heading_threshold_rad, combine
    )


def infer_nav_command_arc_length(
    future_trajectory: np.ndarray,
    forward_m: float = NAV_COMMAND_FORWARD_M,
    lateral_threshold_m: float = NAV_COMMAND_LATERAL_M,
    heading_threshold_rad: float = NAV_COMMAND_HEADING_RAD,
    combine: str = NAV_COMMAND_COMBINE,
) -> int:
    ft = np.asarray(future_trajectory, dtype=np.float64)
    if ft.size == 0:
        return 1
    tlen = int(ft.shape[0])
    if tlen < 1:
        return 1

    xy = np.zeros((tlen + 1, 2), dtype=np.float64)
    xy[1:] = ft[:, :2]
    h_vert = np.zeros(tlen + 1, dtype=np.float64)
    h_vert[1:] = ft[:, 2]

    diffs = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    total_arc = float(seg_len.sum())

    if total_arc < 1e-9:
        return 1

    target = float(forward_m)

    def _interp_y_h(s_target: float) -> Tuple[float, float]:
        if s_target <= 0.0:
            return float(xy[0, 1]), float(h_vert[0])
        cum = 0.0
        for i in range(len(seg_len)):
            L = float(seg_len[i])
            if L < 1e-12:
                continue
            if cum + L >= s_target - 1e-9:
                t = (s_target - cum) / L
                pos = xy[i] + t * diffs[i]
                h0 = float(h_vert[i])
                h1 = float(h_vert[i + 1])
                dh = h1 - h0
                dh = (dh + np.pi) % (2 * np.pi) - np.pi
                h = h0 + t * dh
                return float(pos[1]), float(h)
            cum += L
        return float(xy[-1, 1]), float(h_vert[-1])

    if total_arc < target:
        if float(seg_len[-1]) > 1e-9:
            direc = diffs[-1] / seg_len[-1]
        else:
            hl = float(h_vert[-1])
            direc = np.array([np.cos(hl), np.sin(hl)], dtype=np.float64)
        extra = target - total_arc
        pos = xy[-1] + direc * extra
        y_lat = float(pos[1])
        h_at = float(h_vert[-1])
    else:
        y_lat, h_at = _interp_y_h(target)

    return decide_command(
        y_lat, h_at, lateral_threshold_m, heading_threshold_rad, combine
    )


def infer_navigation_command(
    future_trajectory: np.ndarray,
    mode: str,
    *,
    forward_m: float = NAV_COMMAND_FORWARD_M,
    lateral_threshold_m: float = NAV_COMMAND_LATERAL_M,
    heading_threshold_rad: float = NAV_COMMAND_HEADING_RAD,
    combine: str = NAV_COMMAND_COMBINE,
) -> int:
    """由相对 future 轨迹 (T,3) 推断 0=左 / 1=直 / 2=右。

    ``mode``:
      - ``endpoint``: 仅末点 (y, heading)
      - ``arc_length``: 沿折线累积弧长至 ``forward_m`` 米处取 (y, heading)
    """
    m = str(mode).strip().lower()
    if m == "endpoint":
        return infer_nav_command_endpoint(
            future_trajectory,
            lateral_threshold_m=lateral_threshold_m,
            heading_threshold_rad=heading_threshold_rad,
            combine=combine,
        )
    if m == "arc_length":
        return infer_nav_command_arc_length(
            future_trajectory,
            forward_m=forward_m,
            lateral_threshold_m=lateral_threshold_m,
            heading_threshold_rad=heading_threshold_rad,
            combine=combine,
        )
    raise ValueError(f"Unknown infer mode: {mode!r}")
