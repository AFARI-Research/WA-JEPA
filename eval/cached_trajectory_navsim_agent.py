"""NAVSIM agent that replays precomputed trajectories from a pickle cache."""

from __future__ import annotations

import inspect
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory


def _load_trajectories(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "trajectories" in payload:
        payload = payload["trajectories"]
    if not isinstance(payload, dict):
        raise TypeError(f"Trajectory cache must be a dict, got {type(payload)!r}")
    return {
        str(token): np.asarray(poses, dtype=np.float32)
        for token, poses in payload.items()
    }


def _scene_token(scene: Any) -> str:
    metadata = getattr(scene, "scene_metadata", None)
    for name in ("initial_token", "scene_token"):
        token = getattr(metadata, name, None)
        if token:
            return str(token)
    frames = getattr(scene, "frames", None) or []
    if frames:
        center = len(frames) // 2
        token = getattr(frames[center], "token", None)
        if token:
            return str(token)
    raise RuntimeError("Could not infer NAVSIM token from scene.")


class CachedTrajectoryNavsimAgent(AbstractAgent):
    """Replay cached worldmodel trajectories while NAVSIM performs scoring."""

    def __init__(
        self,
        trajectory_cache_path: str,
        trajectory_sampling=None,
        **_unused_kwargs,
    ):
        if "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters:
            super().__init__(trajectory_sampling, requires_scene=True)
        else:
            super().__init__()
            self.requires_scene = True
        self.trajectory_cache_path = trajectory_cache_path
        self._trajectories: dict[str, np.ndarray] = {}

    def name(self) -> str:
        return "cached_worldmodel_trajectory"

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def initialize(self) -> None:
        self._trajectories = _load_trajectories(self.trajectory_cache_path)
        if not self._trajectories:
            raise RuntimeError(f"Empty trajectory cache: {self.trajectory_cache_path}")

    def compute_trajectory(self, agent_input, scene=None) -> Trajectory:
        if scene is None:
            raise RuntimeError("CachedTrajectoryNavsimAgent requires NAVSIM scene input.")
        token = _scene_token(scene)
        try:
            poses = self._trajectories[token]
        except KeyError as exc:
            raise KeyError(f"Token {token} missing from {self.trajectory_cache_path}") from exc
        try:
            return Trajectory(poses)
        except TypeError:
            return Trajectory(poses, self._trajectory_sampling)
