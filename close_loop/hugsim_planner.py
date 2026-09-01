"""Adapter that drives a WA-JEPA checkpoint inside the HUGSIM closed-loop simulator.

HUGSIM itself is not modified. Upstream already supports an external driving stack:
``sim/utils/launch_ad.py`` spawns a subprocess and exchanges observations and plans
over two FIFOs, so this package only implements the client side. ``closed_loop.py``,
``traj2control`` and ``hugsim_evaluate`` stay byte-identical, which is what makes the
resulting scores comparable to the published HUGSIM baselines.

Every conversion below was measured against the simulator, not inferred from
documentation. All of them fail silently -- a wrong camera slot or an inverted
command mapping produces a lower score with no error anywhere:

1. Camera slots. The model identifies a camera by a learned embedding indexed by its
   position in ``model.camera_names``, so ``camera_map`` in the config must line the
   two name lists up entry by entry.

2. Sample rate. HUGSIM steps at 0.25 s (4 Hz) while the model's history and waypoints
   are 0.5 s apart (2 Hz), so history is strided by 2 simulator frames. Feeding every
   frame makes the model observe a world running at half speed.

3. World -> ego frame. ``sim/hug_sim.py`` builds ``ego_box`` such that
   ``(x, y, yaw) = (ego_box[0], ego_box[1], ego_box[6])`` is a right-handed frame with
   x forward and y left -- the same convention the model was trained on.

4. Command encoding. HUGSIM uses 0 = right / 1 = left / 2 = straight; NAVSIM uses
   0 = left / 1 = straight / 2 = right. See ``hugsim.command_map`` in the config.

5. Output frame. ``closed_loop.py`` applies ``plan[:, [1, 0]]`` and negates the second
   column to read the plan as a lidar-frame (x right, y forward) trajectory, so this
   module emits ``(x, y)_lidar = (-y, x)_ego``.

6. Precision. fp32 weights under ``torch.autocast(bfloat16)``. This matches the bf16
   autocast the model was trained under (``trainer.mixed_precision: "bf16"``). Casting
   the weights to bfloat16 instead is a different thing and does not reproduce the
   reported numbers.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np


DEFAULT_CAMERA_MAP = {
    "cam_l0": "CAM_FRONT_LEFT",
    "cam_f0": "CAM_FRONT",
    "cam_r0": "CAM_FRONT_RIGHT",
    "cam_b0": "CAM_BACK",
}
DEFAULT_COMMAND_MAP = [2, 0, 1]
DEFAULT_SIM_DT_S = 0.25
DEFAULT_TRAJECTORY_FRAME_RATE_HZ = 2.0


def convert_absolute_to_relative_se2_array(origin_xyh, state_se2_array):
    """Convert a global SE(2) array into coordinates relative to ``origin_xyh``.

    Line-for-line equivalent to ``datasets.ego_trajectory_utils`` so that the relative
    pose convention matches training exactly. That module is not imported here because
    it pulls in ``nuplan.database``; a HUGSIM client should not need the nuPlan
    database stack to compute a rotation.
    """
    state_se2_array = np.asarray(state_se2_array, dtype=np.float64).copy()
    if state_se2_array.shape[-1] != 3:
        raise ValueError(f"Expected last dim 3, got shape {state_se2_array.shape}")
    ox, oy, oh = float(origin_xyh[0]), float(origin_xyh[1]), float(origin_xyh[2])
    theta = -oh
    origin_array = np.array([[ox, oy, oh]], dtype=np.float64)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points_rel = state_se2_array - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ rot.T
    points_rel[..., 2] = np.arctan2(np.sin(points_rel[..., 2]), np.cos(points_rel[..., 2]))
    return points_rel


class HugsimPlanner:
    """Loads a checkpoint, keeps the history buffer, turns ``(obs, info)`` into waypoints.

    The client (``close_loop/hugsim_client.py``) only relies on ``plan``, ``horizon``,
    ``hugsim_cams``, ``traj_log`` and ``n_warmup_pad``.
    """

    def __init__(
        self,
        repo_root: str,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        log=print,
    ):
        import torch
        from omegaconf import OmegaConf

        self.torch = torch
        self.log = log
        self.device = device
        # Autocast is tied to the device rather than hardcoded to "cuda", so `--device cpu`
        # stays usable for offline debugging. Scenarios always run on a GPU.
        self.autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

        repo_root = str(Path(repo_root).resolve())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        cfg = OmegaConf.load(str(config_path))
        OmegaConf.set_struct(cfg, False)
        # Closed-loop inference restores the trained state dict, so neither the V-JEPA
        # pretrained weights nor `init_from_checkpoint` are needed. The shipped preset
        # already disables them; forcing it here keeps a training config usable as-is.
        cfg.model.require_pretrained = False
        cfg.model.vjepa2_ckpt = None
        cfg.model.init_from_checkpoint = None
        cfg.model.require_init_from_checkpoint = False
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.cfg = cfg_dict

        model_cfg = cfg_dict["model"]
        data_cfg = cfg_dict.get("data") or {}
        sim_cfg = cfg_dict.get("hugsim") or {}

        camera_map = {
            str(k).lower(): str(v) for k, v in (sim_cfg.get("camera_map") or DEFAULT_CAMERA_MAP).items()
        }
        self.model_cams = [str(c).lower() for c in model_cfg["camera_names"]]
        unmapped = [c for c in self.model_cams if c not in camera_map]
        if unmapped:
            raise RuntimeError(
                f"model.camera_names contains {unmapped}, which have no HUGSIM camera in "
                f"hugsim.camera_map (known: {sorted(camera_map)})"
            )
        self.hugsim_cams = [camera_map[c] for c in self.model_cams]

        self.command_map = [int(v) for v in (sim_cfg.get("command_map") or DEFAULT_COMMAND_MAP)]
        self.in_h, self.in_w = int(model_cfg["input_hw"][0]), int(model_cfg["input_hw"][1])
        self.n_hist = int(model_cfg["num_history_frames"])
        self.ego_dim = int(model_cfg["ego_status_dim"])
        self.horizon = int(model_cfg["trajectory_horizon"])

        sim_dt = float(sim_cfg.get("sim_dt_s", DEFAULT_SIM_DT_S))
        rate_hz = float(data_cfg.get("trajectory_frame_rate_hz", DEFAULT_TRAJECTORY_FRAME_RATE_HZ))
        self.model_interval_s = 1.0 / rate_hz
        self.history_stride = int(round(self.model_interval_s / sim_dt))
        if self.history_stride < 1:
            raise RuntimeError(
                f"history stride resolved to {self.history_stride} from sim_dt_s={sim_dt} and "
                f"trajectory_frame_rate_hz={rate_hz}; the simulator must step at least as "
                "often as the model's trajectory interval"
            )

        from models.factory import build_world_model
        from training.checkpoint import load_checkpoint

        self.log(
            f"[planner] building {model_cfg.get('variant')} "
            f"cams={self.model_cams} stride={self.history_stride}"
        )
        model = build_world_model(cfg_dict).to(device)
        # `load_checkpoint` is strict by default; a partially restored trajectory head
        # would still produce plausible-looking waypoints.
        load_checkpoint(checkpoint_path, model, map_location=device, log_fn=self.log)
        self.model = model.eval()  # fp32 weights, bfloat16 comes from autocast
        mem = (f" (cuda mem {torch.cuda.memory_allocated() / 1e9:.2f} GB)"
               if self.autocast_device == "cuda" else "")
        self.log(f"[planner] weights loaded from {checkpoint_path}{mem}")

        # Oldest frame we need is stride * (n_hist - 1) steps back.
        self.buf_len = self.history_stride * (self.n_hist - 1) + 1
        self.buf = deque(maxlen=self.buf_len)
        self.n_calls = 0
        self.n_warmup_pad = 0  # steps where history was padded by repeating the oldest frame
        self.traj_log: list[dict] = []

    # ---------- obs / info -> tensors ----------

    def _image_tensor(self, rgb, cam_name: str):
        """HUGSIM render -> model input tensor ``[3, H, W]`` in ``[-1, 1]``.

        Matches ``datasets/navsim_official_dataset.py``. ``obs['rgb'][cam]`` is already
        RGB uint8 HWC, so do not run ``cvtColor`` on it -- swapping R and B produces no
        error at all.
        """
        import cv2

        if rgb is None:
            raise RuntimeError(f"obs['rgb']['{cam_name}'] is None")
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            raise RuntimeError(f"{cam_name} has unexpected image shape {img.shape}")
        img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
        t = self.torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return t.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    @staticmethod
    def _pose_from_info(info) -> np.ndarray:
        """``(x, y, yaw)`` in the HUGSIM world frame: x forward, y left, right-handed."""
        box = info["ego_box"]
        return np.array([float(box[0]), float(box[1]), float(box[6])], dtype=np.float64)

    def _ego_status(self, info):
        """NAVSIM ``ego_status = [command_onehot(4), vx, vy, ax, ay]``.

        Matches ``eval/navsim_agent.py``. HUGSIM reports a scalar speed and a
        longitudinal acceleration with no lateral component, so vy and ay are zero.
        """
        raw_cmd = int(np.asarray(info["command"]).ravel()[0])
        if not 0 <= raw_cmd < len(self.command_map):
            raise RuntimeError(
                f"HUGSIM command={raw_cmd} is outside hugsim.command_map {self.command_map}"
            )
        nav_cmd = self.command_map[raw_cmd]

        st = np.zeros(self.ego_dim, dtype=np.float32)
        st[min(nav_cmd, 3)] = 1.0
        st[4] = float(np.asarray(info["ego_velo"]).ravel()[0])  # vx
        st[5] = 0.0                                             # vy
        st[6] = float(np.asarray(info["accelerate"]).ravel()[0])  # ax
        st[7] = 0.0                                             # ay
        return st, nav_cmd, raw_cmd

    def _history_indices(self) -> list[int]:
        """Pick ``n_hist`` buffer indices spaced by the stride, ending at the current frame.

        While the buffer is still filling, indices clamp to 0, i.e. the oldest frame is
        repeated. That is counted and reported because it means the model saw a frozen
        past rather than real history.
        """
        last = len(self.buf) - 1
        stride = self.history_stride
        return [max(0, last - stride * (self.n_hist - 1 - k)) for k in range(self.n_hist)]

    # ---------- public interface ----------

    def reset(self) -> None:
        self.buf.clear()
        self.n_calls = 0
        self.n_warmup_pad = 0
        self.traj_log = []

    def plan(self, obs, info) -> np.ndarray:
        """``(obs, info)`` -> waypoints ``[horizon, 2]`` in the lidar frame (x right, y forward).

        The return value is pickled straight back to ``closed_loop.py``, which converts
        it into controls with ``traj2control``.
        """
        torch = self.torch

        imgs = torch.stack(
            [self._image_tensor(obs["rgb"][c], c) for c in self.hugsim_cams], dim=0
        )  # [V, 3, H, W]
        self.buf.append({"imgs": imgs, "pose": self._pose_from_info(info)})

        idx = self._history_indices()
        padded = len(self.buf) < self.buf_len
        if padded:
            self.n_warmup_pad += 1

        hist_imgs = torch.stack([self.buf[i]["imgs"] for i in idx], dim=0)  # [T, V, 3, H, W]
        poses = np.stack([self.buf[i]["pose"] for i in idx], axis=0)        # [T, 3] world frame
        hist_traj = convert_absolute_to_relative_se2_array(poses[-1], poses).astype(np.float32)
        ego_status, nav_cmd, raw_cmd = self._ego_status(info)

        features = {
            "history_images": hist_imgs.unsqueeze(0).to(self.device),
            "ego_status": torch.from_numpy(ego_status).unsqueeze(0).to(self.device),
            "history_trajectory": torch.from_numpy(hist_traj).unsqueeze(0).to(self.device),
        }

        with torch.no_grad(), torch.autocast(self.autocast_device, dtype=torch.bfloat16):
            pred = self.model.predict_trajectory(features)
        traj = pred.squeeze(0).float().cpu().numpy().astype(np.float64)  # [horizon, 3] x fwd, y left

        # ego (x forward, y left) -> lidar (x right, y forward); see module docstring item 5.
        lidar = np.stack([-traj[:, 1], traj[:, 0]], axis=1)

        self.n_calls += 1
        self.traj_log.append(
            {
                "step": self.n_calls - 1,
                "hugsim_command": raw_cmd,
                "navsim_command": nav_cmd,
                "ego_velo": float(ego_status[4]),
                "accelerate": float(ego_status[6]),
                "hist_pad": padded,
                "traj_ego": traj.tolist(),
            }
        )
        return np.ascontiguousarray(lidar, dtype=np.float64)
