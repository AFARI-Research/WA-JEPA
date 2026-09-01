"""Evaluation metrics for world model and trajectory prediction."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Compute PSNR between pred and target images.

    Both tensors should be in the same range (e.g. [-1, 1] → data_range=2.0).
    Returns a scalar tensor.
    """
    mse = (pred.float() - target.float()).pow(2).mean()
    if mse < 1e-10:
        return torch.tensor(100.0, device=pred.device)
    return 10.0 * torch.log10(torch.tensor(data_range ** 2, device=pred.device) / mse)


def trajectory_ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Displacement Error: mean L2 over all waypoints.

    Args:
        pred: ``[N, 3]`` (x, y, heading)
        gt:   ``[N, 3]``
    """
    assert pred.shape[0] == gt.shape[0], (pred.shape, gt.shape)
    return float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1).mean())


def trajectory_fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final Displacement Error: L2 at the last waypoint."""
    assert pred.shape[0] == gt.shape[0], (pred.shape, gt.shape)
    return float(np.linalg.norm(pred[-1, :2] - gt[-1, :2]))


def trajectory_heading_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute heading error (radians)."""
    assert pred.shape[0] == gt.shape[0], (pred.shape, gt.shape)
    diff = pred[:, 2] - gt[:, 2]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return float(np.abs(diff).mean())


def parse_trajectory_text(text: str) -> Optional[np.ndarray]:
    """Parse trajectory text in ``[PT, (x, y, h), ...]`` format.

    First tries ReCogDrive-style strict match: exactly 8 triples with decimal
    components (comma-space inside parentheses). If that fails, falls back to a
    lenient scan of ``(...)`` segments as three comma-separated floats.

    Returns ``[N, 3]`` numpy array, or ``None`` if no waypoint could be parsed.
    """
    strict = re.search(
        r"\[PT(?:, )?("
        r"(?:\([-+]?\d*\.\d+,\s*[-+]?\d*\.\d+,\s*[-+]?\d*\.\d+\)(?:,\s*)?){8})"
        r"\]",
        text,
    )
    if strict is not None:
        inner = strict.group(1)
        coords = re.findall(
            r"\(([-+]?\d*\.\d+),\s*([-+]?\d*\.\d+),\s*([-+]?\d*\.\d+)\)",
            inner,
        )
        if len(coords) == 8:
            arr = np.array(
                [[float(a), float(b), float(c)] for a, b, c in coords],
                dtype=np.float32,
            )
            return arr

    pattern = r"\(([^)]+)\)"
    matches = re.findall(pattern, text)
    if not matches:
        return None

    def _try_segment(inner: str) -> Optional[list[float]]:
        parts = inner.split(",")
        if len(parts) < 3:
            return None
        try:
            return [float(parts[0]), float(parts[1]), float(parts[2])]
        except ValueError:
            return None

    parsed = [_try_segment(m) for m in matches]
    first_ok = next((i for i, p in enumerate(parsed) if p is not None), None)
    if first_ok is None:
        return None

    trimmed = parsed[first_ok:]
    out_rows: list[np.ndarray] = []
    last: np.ndarray | None = None

    for i, p in enumerate(trimmed):
        if p is not None:
            last = np.array(p, dtype=np.float32)
            out_rows.append(last.copy())
        else:
            if last is None:
                break
            out_rows.append(last.copy())
            for _ in range(i + 1, len(trimmed)):
                out_rows.append(last.copy())
            break

    if not out_rows:
        return None
    return np.stack(out_rows, axis=0)
