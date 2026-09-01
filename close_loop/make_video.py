"""Render one finished HUGSIM scenario into a driving video.

Reads only what a run already archived with ``SAVE_VIZ=1`` -- nothing is re-simulated and
no checkpoint is loaded, so this is safe to run long after the benchmark finished, on a
machine without a GPU.

Each frame is::

    +-------------------------------+-------------------+
    | CAM_FRONT + the planned path  | BEV: plan, other  |
    | projected onto the ground     | agents, path so far|
    +-------------------------------+-------------------+
    | step / time / speed / nav command / scenario score |
    +----------------------------------------------------+

Why this is worth having: `HD-Score 0.31` does not say whether the model steered into a
wall, stopped for nothing, or drove well and lost points to the controller. The overlay
does. It is also the only check that catches a plan which is numerically plausible and
geometrically wrong -- the audit section of the report can only count failures that
raised.

The coordinate chain, from HUGSIM's source rather than from documentation:

1. ``sim/hugsim_env/envs/hug_sim.py`` builds the ego box from ``whl = [1.6, 1.5, 3.0]``
   with vertices at ``y in [0, 1.5]``, so the vehicle frame is ``(x right, y down,
   z forward)`` and the ground plane is ``y = 1.5``.
2. ``_get_obs`` renders every camera from ``c2w = ego @ v2front @ inv(v2c) @ cam_rect``.
   A point already in the vehicle frame therefore lands in the rendered camera's frame at
   ``p_c = inv(cam_rect) @ v2c @ inv(v2front) @ p_v``. For ``CAM_FRONT`` the ``v2c`` terms
   cancel and only ``cam_rect`` is left.
3. ``create_cam`` turns the intrinsics into ``fx = W / (2 tan(fovx / 2))``,
   ``fy = H / (2 tan(fovy / 2))`` with principal point ``(cx, cy)``: a plain OpenCV
   pinhole, no distortion.
4. The model predicts in the ego frame ``(x forward, y left)``, so a waypoint on the
   ground is ``p_v = (-y, 1.5, x)``.

**The four splits do not share a camera configuration**, which is why this script reads it
instead of hardcoding it:

| split | CAM_FRONT fovx | ``cam_rect`` rotation | ``cam_rect`` translation |
| --- | --- | --- | --- |
| nuScenes | 65.12 deg | none | ``[0, 0.3, 0]`` |
| KITTI-360 | 65.12 deg | ``[5, 0, 0]`` deg | none |
| Waymo | 58.12 deg | none | ``[0, 0.3, 0]`` |
| PandaSet | 58.12 deg | none | ``[0, 0.7, 0]`` |

Using nuScenes' values on PandaSet puts the camera 0.4 m off the ground truth and widens
the field of view by 7 degrees; the overlay still looks like a plausible trajectory, just
one that no longer matches the pixels. That is the failure this module is built to
prevent, so it cross-checks the config against ``viz/camera.json``, which the client wrote
during the run, and refuses to draw on a mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

DATASETS = ["nuscenes", "kitti360", "waymo", "pandaset"]

# Vehicle frame, from hug_sim.py's whl = [1.6, 1.5, 3.0]: y points down and the ground is
# at the bottom of the ego box.
GROUND_Y_V = 1.5
EGO_WIDTH_M = 1.6
EGO_LENGTH_M = 3.0

# Points closer than this along the optical axis project to nonsense (and behind the
# camera the projection flips sign), so they are dropped rather than clamped.
MIN_DEPTH_M = 0.5

INTRINSIC_KEYS = ("H", "W", "fovx", "fovy", "cx", "cy")

# HUGSIM's command encoding, not NAVSIM's; `info['command']` is archived unconverted.
CMD_NAME = {0: "RIGHT", 1: "LEFT", 2: "STRAIGHT"}

METRICS = ["nc", "dac", "ttc", "c", "pdms", "rc", "hdscore"]


# --------------------------------------------------------------------------------------
# camera model
# --------------------------------------------------------------------------------------


def fov2focal(fov_rad: float, pixels: int) -> float:
    return pixels / (2.0 * math.tan(fov_rad / 2.0))


def fov_to_radians(fov: float) -> float:
    """Accept either unit.

    The camera YAML stores degrees and ``load_camera_cfg`` converts them in place, so
    ``info['cam_params']`` -- and therefore ``viz/camera.json`` -- holds radians. The two
    sources are read side by side here. The test is unambiguous for real optics: a field
    of view in radians is below pi, in degrees above 3.14 unless the lens is narrower than
    3.14 degrees, which no camera in this benchmark is.
    """
    return math.radians(fov) if fov > math.pi else float(fov)


def euler_xyz_matrix(rot_deg) -> np.ndarray:
    """``scipy.spatial.transform.Rotation.from_euler('XYZ', rot, degrees=True)``.

    Reimplemented rather than imported: SciPy is not a dependency of this repository, and
    pulling one in for three rotation matrices in a visualisation script is not worth it.
    Capital ``XYZ`` is SciPy's *intrinsic* convention, which composes as ``Rx @ Ry @ Rz``;
    the extrinsic reading (lowercase ``xyz``, ``Rz @ Ry @ Rx``) gives a different matrix
    for the same angles. ``tests/test_hugsim_video.py`` pins the composition.
    """
    a, b, c = (math.radians(float(v)) for v in rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]], dtype=np.float64)
    return rx @ ry @ rz


def rt2pose(rot_deg, trans) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = euler_xyz_matrix(rot_deg)
    pose[:3, 3] = np.asarray(trans, dtype=np.float64)
    return pose


def load_camera_yaml(path: Path) -> tuple[dict, np.ndarray]:
    """HUGSIM's ``configs/sim/<dataset>_camera.yaml`` -> per-camera params and ``cam_rect``.

    ``cam_rect`` is the reason this file has to be read at all: it is not part of
    ``info``, so a run cannot archive it, and it differs between all four splits.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - environment problem, not a code path
        raise SystemExit("pyyaml is required to read the camera config: pip install pyyaml")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cams = {}
    for name, cam in cfg["cams"].items():
        intr = dict(cam["intrinsics"])
        intr["fovx"] = fov_to_radians(intr["fovx"])
        intr["fovy"] = fov_to_radians(intr["fovy"])
        ext = cam["extrinsics"]
        cams[name] = {"intrinsic": intr, "v2c": rt2pose(ext["v2c_rot"], ext["v2c_trans"])}

    rect = np.eye(4, dtype=np.float64)
    if "cam_rect" in cfg:
        rect = rt2pose(cfg["cam_rect"]["rot"], cfg["cam_rect"]["trans"])
    return cams, rect


def load_archived_camera(scenario_dir: Path) -> tuple[dict | None, str]:
    """``viz/camera.json`` as written by ``hugsim_client.py``, or ``None`` with a reason.

    Keys starting with an underscore are the record's own metadata (which config the run
    used), not cameras.
    """
    path = Path(scenario_dir) / "viz" / "camera.json"
    if not path.is_file():
        return None, f"{path} is absent (run archived before camera.json existed)"
    raw = json.loads(path.read_text(encoding="utf-8"))
    cams = {}
    for name, cam in raw.items():
        if name.startswith("_"):
            continue
        intr = dict(cam["intrinsic"])
        intr["fovx"] = fov_to_radians(intr["fovx"])
        intr["fovy"] = fov_to_radians(intr["fovy"])
        cams[name] = {"intrinsic": intr, "v2c": np.asarray(cam["v2c"], dtype=np.float64)}
    return cams, str(path)


def compare_cameras(from_yaml: dict, from_run: dict, cam: str) -> list[str]:
    """Differences between the config being used and the one the run actually rendered with.

    Reach of this check, stated plainly because a check that is trusted beyond its reach is
    worse than none: it compares intrinsics and ``v2c``, which separates the 65.12-degree
    splits (nuScenes, KITTI-360) from the 58.12-degree ones (Waymo, PandaSet). It cannot
    separate nuScenes from KITTI-360, or Waymo from PandaSet, because those pairs differ
    only in ``cam_rect`` -- which is absent from the archive. ``camera.json`` also records
    the dataset name for exactly that reason; see ``resolve_camera``.
    """
    diffs = []
    if cam not in from_yaml:
        return [f"camera config has no {cam} (cameras: {sorted(from_yaml)})"]
    if cam not in from_run:
        return [f"the run archived no {cam} (cameras: {sorted(from_run)})"]
    a, b = from_yaml[cam], from_run[cam]
    for key in INTRINSIC_KEYS:
        va, vb = float(a["intrinsic"][key]), float(b["intrinsic"][key])
        if abs(va - vb) > 1e-4 * max(1.0, abs(vb)):
            diffs.append(f"{cam}.{key}: config {va:.6f}, run {vb:.6f}")
    dv = float(np.abs(a["v2c"] - b["v2c"]).max())
    if dv > 1e-6:
        diffs.append(f"{cam}.v2c: max element difference {dv:.3e}")
    return diffs


class CameraModel:
    """Ego-frame ground points -> pixels of one rendered camera.

    Holds the full ``inv(cam_rect) @ v2c @ inv(v2front)`` chain even though the archived
    camera is always ``CAM_FRONT``, where it collapses to ``inv(cam_rect)``. Keeping the
    general form costs two lines and means pointing this at another camera cannot silently
    produce a front-camera projection.
    """

    def __init__(self, name: str, intrinsic: dict, v2cam: np.ndarray, source: str = ""):
        self.name = name
        self.source = source
        self.H = int(intrinsic["H"])
        self.W = int(intrinsic["W"])
        self.fx = fov2focal(float(intrinsic["fovx"]), self.W)
        self.fy = fov2focal(float(intrinsic["fovy"]), self.H)
        self.cx = float(intrinsic["cx"])
        self.cy = float(intrinsic["cy"])
        self.v2cam = np.asarray(v2cam, dtype=np.float64)

    @classmethod
    def build(cls, cams: dict, cam_rect: np.ndarray, cam: str = "CAM_FRONT",
              source: str = "") -> "CameraModel":
        if cam not in cams:
            raise KeyError(f"{cam} is not in the camera config (have {sorted(cams)})")
        v2front = cams["CAM_FRONT"]["v2c"]
        v2cam = np.linalg.inv(cam_rect) @ cams[cam]["v2c"] @ np.linalg.inv(v2front)
        return cls(cam, cams[cam]["intrinsic"], v2cam, source=source)

    def project_ground(self, pts_ego) -> tuple[np.ndarray, np.ndarray]:
        """``[(x forward, y left), ...]`` on the ground -> ``(uv, visible)``."""
        pts = np.asarray(pts_ego, dtype=np.float64).reshape(-1, 2)
        p_v = np.stack(
            [
                -pts[:, 1],                              # right = -left
                np.full(len(pts), GROUND_Y_V),            # on the ground
                pts[:, 0],                               # forward
                np.ones(len(pts)),
            ],
            axis=1,
        )
        p_c = p_v @ self.v2cam.T
        z = p_c[:, 2]
        ahead = z > MIN_DEPTH_M
        zc = np.maximum(z, 1e-6)
        u = np.where(ahead, self.fx * p_c[:, 0] / zc + self.cx, -1.0)
        v = np.where(ahead, self.fy * p_c[:, 1] / zc + self.cy, -1.0)
        # Points somewhat outside the frame are kept: a segment that leaves the image
        # should still be drawn up to the border, and cv2 clips it. Only absurd
        # coordinates are dropped, so a near-horizon point does not become a line across
        # the whole frame.
        visible = ahead & (u > -self.W) & (u < 2 * self.W) & (v > -self.H) & (v < 2 * self.H)
        return np.stack([u, v], axis=1), visible


def dataset_from_path(path: Path) -> str | None:
    """Infer the split from a results path such as ``pandaset_ltf``.

    Ambiguity is reported as "unknown" rather than resolved by picking the first match: a
    path containing two split names is exactly the case where guessing draws the wrong
    camera.
    """
    text = str(Path(path).resolve()).lower()
    hits = [d for d in DATASETS if d in text]
    return hits[0] if len(hits) == 1 else None


def resolve_camera(scenario_dir: Path, dataset: str | None, camera_yaml: Path | None,
                   hugsim_root: Path | None, cam: str = "CAM_FRONT",
                   strict: bool = True, log=print) -> CameraModel:
    """Pick the camera config for this scenario and verify it against the run's own record."""
    scenario_dir = Path(scenario_dir)
    archived, archived_src = load_archived_camera(scenario_dir)

    recorded_ds = None
    if archived is not None:
        raw = json.loads((scenario_dir / "viz" / "camera.json").read_text(encoding="utf-8"))
        recorded_ds = raw.get("_dataset") or None

    if dataset is None:
        dataset = recorded_ds or dataset_from_path(scenario_dir.parent)
    if recorded_ds and dataset and recorded_ds != dataset:
        raise SystemExit(
            f"[fatal] the run recorded dataset={recorded_ds!r} but {dataset!r} was requested. "
            f"The camera configs differ per split, so the overlay would be drawn with the "
            f"wrong geometry while still looking like a trajectory. Fix --dataset."
        )

    if camera_yaml is None:
        if dataset is None:
            raise SystemExit(
                "[fatal] cannot tell which split this is. Pass --dataset, or --camera-config "
                f"directly. (Looked at the run's camera.json and at {scenario_dir.parent.name}.)"
            )
        root = hugsim_root or Path(os.environ.get("HUGSIM_ROOT", ""))
        if not str(root):
            raise SystemExit(
                "[fatal] set HUGSIM_ROOT or pass --hugsim-root / --camera-config; the "
                "per-split camera geometry lives in HUGSIM's configs/sim/<dataset>_camera.yaml "
                "and is not part of the archive."
            )
        camera_yaml = Path(root) / "configs" / "sim" / f"{dataset}_camera.yaml"
    camera_yaml = Path(camera_yaml)
    if not camera_yaml.is_file():
        raise SystemExit(f"[fatal] camera config not found: {camera_yaml}")

    cams, cam_rect = load_camera_yaml(camera_yaml)
    if archived is None:
        # Not fatal: runs archived before the client recorded camera.json are still worth
        # rendering. Say so, because the cross-check is the thing standing between a
        # correct overlay and a convincing wrong one.
        log(f"[warn] {archived_src}; the camera config cannot be cross-checked. Confirm "
            f"--dataset matches the run that produced {scenario_dir.name}.")
    else:
        diffs = compare_cameras(cams, archived, cam)
        if diffs:
            msg = (f"[fatal] {camera_yaml} does not match the camera the run rendered with:\n  "
                   + "\n  ".join(diffs)
                   + "\nThe usual cause is the wrong split's camera config (nuScenes and "
                     "KITTI-360 use fovx 65.12 deg, Waymo and PandaSet 58.12 deg).")
            if strict:
                raise SystemExit(msg)
            log(msg.replace("[fatal]", "[warn]"))

    return CameraModel.build(cams, cam_rect, cam=cam, source=str(camera_yaml))


# --------------------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------------------


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    import cv2

    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_traj_on_front(img, traj_ego, cam: CameraModel):
    """Draw the plan as a ribbon on the ground: a bare polyline hides the heading."""
    import cv2

    out = img.copy()
    half = EGO_WIDTH_M / 2.0
    traj = [(float(p[0]), float(p[1])) for p in traj_ego]
    # Start the ribbon at the bumper so it visibly leaves the ego vehicle rather than
    # floating at the first waypoint half a second ahead.
    left = [(0.0, half)] + [(x, y + half) for x, y in traj]
    right = [(0.0, -half)] + [(x, y - half) for x, y in traj]

    pl, vl = cam.project_ground(left)
    pr, vr = cam.project_ground(right)

    overlay = out.copy()
    n = len(pl) - 1
    for i in range(n):
        if not (vl[i] and vl[i + 1] and vr[i] and vr[i + 1]):
            continue
        quad = np.array([pl[i], pl[i + 1], pr[i + 1], pr[i]], dtype=np.int32)
        t = i / max(1, n - 1)
        # BGR, green near the vehicle fading to cyan far away, so distance is readable.
        cv2.fillConvexPoly(overlay, quad, (int(60 + 60 * t), int(230 - 60 * t), int(60 + 20 * t)))
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    pc, vc = cam.project_ground([(0.0, 0.0)] + traj)
    line = [tuple(np.int32(p)) for p, ok in zip(pc, vc) if ok]
    for i in range(len(line) - 1):
        cv2.line(out, line[i], line[i + 1], (0, 255, 255), 2, cv2.LINE_AA)
    for i, (p, ok) in enumerate(zip(pc, vc)):
        if ok and i > 0:
            cv2.circle(out, tuple(np.int32(p)), 4, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def rot2d(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def draw_bev(entry: dict, past_xy, size: int = 450, span: float = 40.0):
    """Bird's-eye panel, ego at the centre facing up. ``span`` is the visible depth in metres."""
    import cv2

    img = np.full((size, size, 3), 28, np.uint8)
    scale = size / span

    def to_px(pt):
        x, y = float(pt[0]), float(pt[1])
        # ego (x forward, y left) -> image (forward is up, left is left); the vehicle sits
        # low in the frame so most of the panel shows what is ahead of it.
        return int(round(size / 2 - y * scale)), int(round(size * 0.75 - x * scale))

    for d in range(0, int(span) + 1, 10):
        _, v = to_px((d, 0))
        if 0 <= v < size:
            cv2.line(img, (0, v), (size, v), (48, 48, 48), 1)
            _text(img, f"{d}m", (4, max(12, v - 4)), 0.35, (110, 110, 110))
    for d in (-10, 0, 10):
        u, _ = to_px((0, d))
        if 0 <= u < size:
            cv2.line(img, (u, 0), (u, size), (48, 48, 48), 1)

    box = entry["ego_box"]
    ego_xy = np.array([box[0], box[1]], dtype=np.float64)
    ego_yaw = float(box[6])
    w2e = rot2d(-ego_yaw)

    if len(past_xy) > 1:
        rel = (np.asarray(past_xy, dtype=np.float64) - ego_xy) @ w2e.T
        pts = [to_px(p) for p in rel]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (110, 110, 110), 1, cv2.LINE_AA)

    # obj_box = [x, y, z, w, l, h, yaw] in the world frame, same convention as ego_box.
    for b in entry.get("obj_boxes") or []:
        centre = (np.array([b[0], b[1]], dtype=np.float64) - ego_xy) @ w2e.T
        w, l = float(b[3]), float(b[4])
        corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        corners = corners @ rot2d(float(b[6]) - ego_yaw).T + centre
        poly = np.array([to_px(p) for p in corners], np.int32)
        cv2.fillConvexPoly(img, poly, (60, 120, 220))
        cv2.polylines(img, [poly], True, (120, 190, 255), 1, cv2.LINE_AA)

    ec = np.array([[EGO_LENGTH_M / 2, EGO_WIDTH_M / 2], [EGO_LENGTH_M / 2, -EGO_WIDTH_M / 2],
                   [-EGO_LENGTH_M / 2, -EGO_WIDTH_M / 2], [-EGO_LENGTH_M / 2, EGO_WIDTH_M / 2]])
    poly = np.array([to_px(p) for p in ec], np.int32)
    cv2.fillConvexPoly(img, poly, (80, 200, 80))
    cv2.polylines(img, [poly], True, (200, 255, 200), 1, cv2.LINE_AA)

    traj = entry_traj(entry)
    if traj is not None:
        pts = [to_px((0.0, 0.0))] + [to_px(p) for p in traj]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (0, 255, 255), 2, cv2.LINE_AA)
        for p in pts[1:]:
            cv2.circle(img, p, 3, (0, 0, 255), -1, cv2.LINE_AA)

    _text(img, "BEV (plan / objects)", (8, size - 10), 0.42, (170, 170, 170))
    return img


def entry_traj(entry: dict) -> np.ndarray | None:
    """The planned trajectory in the ego frame, ``[horizon, 2]``.

    ``hugsim_client.py`` nests the planner's own record under ``planner``; the top-level
    ``traj_lidar`` is the same trajectory after the swap to the simulator's frame, and
    drawing that one flips forward and lateral without raising anything.
    """
    planner = entry.get("planner") or {}
    traj = planner.get("traj_ego")
    if not traj:
        return None
    arr = np.asarray(traj, dtype=np.float64)
    return arr[:, :2] if arr.ndim == 2 and arr.shape[1] >= 2 else None


def make_status_bar(entry: dict, width: int, scores: dict, n_total: int,
                    flags: tuple[str, ...] = ()):
    bar = np.full((78, width, 3), 20, np.uint8)
    velo = float(entry["ego_velo"])
    _text(bar,
          f"step {entry['step'] + 1}/{n_total}   t={float(entry['timestamp']):.2f}s   "
          f"v={velo:.2f} m/s ({velo * 3.6:.1f} km/h)   "
          f"nav={CMD_NAME.get(entry['command'], entry['command'])}",
          (12, 28), 0.56, (235, 235, 235))
    if scores:
        line = "  ".join(f"{m.upper()}={scores[m]:.3f}" for m in METRICS if m in scores)
        _text(bar, "scenario: " + line, (12, 58), 0.48, (150, 220, 150))
    if flags:
        # Red, top right: these frames are not the model driving, and a video without the
        # label looks exactly like one where it was.
        _text(bar, "  ".join(flags), (max(12, width - 330), 28), 0.5, (90, 90, 255), 1)
    return bar


# --------------------------------------------------------------------------------------
# assembling a scenario
# --------------------------------------------------------------------------------------


def read_scores(scenario_dir: Path) -> dict:
    path = Path(scenario_dir) / "eval.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def read_failed_steps(scenario_dir: Path) -> set[int]:
    """Steps where inference raised and the client substituted a braking plan."""
    path = Path(scenario_dir) / "planner_stats.json"
    if not path.is_file():
        return set()
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return {int(f["step"]) for f in stats.get("failures", []) if "step" in f}


def iter_frames(scenario_dir: Path, cam: CameraModel, max_frames: int | None = None):
    """Yield finished frames one at a time.

    A generator rather than a list on purpose: a 400-step scenario is about 4 MB per
    composed frame, so materialising one costs 1.6 GB and eight parallel workers exhaust
    the machine. ``render_videos.py`` relies on this.
    """
    import cv2

    scenario_dir = Path(scenario_dir)
    with open(scenario_dir / "plan_log.pkl", "rb") as f:
        log = pickle.load(f)
    if max_frames:
        log = log[:max_frames]
    scores = read_scores(scenario_dir)
    failed = read_failed_steps(scenario_dir)

    past: list[list[float]] = []
    for entry in log:
        step = int(entry["step"])
        image_path = scenario_dir / "viz" / "front" / f"{step:04d}.jpg"
        if not image_path.is_file():
            continue
        front = cv2.imread(str(image_path))
        if front is None:
            continue
        past.append([entry["ego_box"][0], entry["ego_box"][1]])

        traj = entry_traj(entry)
        if traj is not None:
            front = draw_traj_on_front(front, traj, cam)
        _text(front, "CAM_FRONT + planned trajectory", (10, 24), 0.55)

        bev = draw_bev(entry, past)
        height = max(front.shape[0], bev.shape[0])
        if front.shape[0] != height:
            front = cv2.resize(front, (int(front.shape[1] * height / front.shape[0]), height))
        if bev.shape[0] != height:
            bev = cv2.resize(bev, (int(bev.shape[1] * height / bev.shape[0]), height))
        top = np.concatenate([front, bev], axis=1)

        flags = []
        if step in failed:
            flags.append("FALLBACK: inference failed")
        elif (entry.get("planner") or {}).get("hist_pad"):
            flags.append("history padded")
        yield np.concatenate(
            [top, make_status_bar(entry, top.shape[1], scores, len(log), tuple(flags))], axis=0
        )


def encode(frames, out_path: Path, fps: int = 4) -> tuple[bool, str]:
    """Stream frames into an mp4. Returns ``(ok, message)`` and never raises on encoder issues.

    Writes ``<name>.part.mp4`` and renames on success, so an interrupted run leaves no
    half-written file that ``--skip-existing`` would then treat as done.
    """
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".part.mp4")
    writer, w, h, n = None, 0, 0, 0
    try:
        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                # Odd dimensions are rejected by some H.264 encoders.
                w -= w % 2
                h -= h % 2
                for fourcc in ("avc1", "mp4v"):
                    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    if writer.isOpened():
                        break
                    writer.release()
                    writer = None
                if writer is None:
                    return False, "no usable encoder (avc1 and mp4v both failed)"
            writer.write(frame[:h, :w])
            n += 1
    finally:
        if writer is not None:
            writer.release()

    if n == 0:
        tmp.unlink(missing_ok=True)
        return False, "no usable frames (missing viz/front images?)"
    # cv2.VideoWriter reports nothing when the codec silently drops everything, so check
    # the file rather than trusting the absence of an exception.
    if not tmp.is_file() or tmp.stat().st_size < 10000:
        size = tmp.stat().st_size if tmp.is_file() else -1
        tmp.unlink(missing_ok=True)
        return False, f"encoded file is implausibly small ({size} B)"
    tmp.replace(out_path)
    return True, f"{n} frames, {w}x{h}, {out_path.stat().st_size / 1e6:.1f} MB"


def build(scenario_dir: Path, out_path: Path, cam: CameraModel, fps: int = 4,
          max_frames: int | None = None) -> tuple[bool, str]:
    scenario_dir = Path(scenario_dir)
    if not (scenario_dir / "plan_log.pkl").is_file():
        return False, "no plan_log.pkl (the run was not made with SAVE_VIZ=1)"
    return encode(iter_frames(scenario_dir, cam, max_frames), out_path, fps=fps)


def pick_representative(summary_json: Path, per_tier: int = 2) -> list[str]:
    """Per difficulty tier, the best **and worst** HD-Score.

    Not a highlight reel by design. Rendering all 436 scenarios is slow and nobody watches
    them; hand-picking a few ends up picking the good ones, which is dressing up the
    result. Best and worst per tier lets a reader see the spread.
    """
    summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    by_tier: dict[str, list[dict]] = {}
    for row in summary["scenarios"]:
        if row.get("hdscore") is None:
            continue
        by_tier.setdefault(row["tier"], []).append(row)
    picked: list[str] = []
    for tier in sorted(by_tier):
        rows = sorted(by_tier[tier], key=lambda r: r["hdscore"])
        take = [rows[-1]] if per_tier == 1 else [rows[-1], rows[0]]
        for row in take:
            if row["dir"] not in picked:
                picked.append(row["dir"])
    return picked


def add_camera_args(ap: argparse.ArgumentParser, with_dataset: bool = True) -> None:
    """The camera options shared with ``render_videos.py``.

    ``with_dataset=False`` is for callers that already take a list of splits: adding a
    singular ``--dataset`` next to a plural ``--datasets`` accepts the flag and then ignores
    it, which is the silent kind of wrong. Left out, argparse resolves ``--dataset`` as an
    abbreviation of ``--datasets`` instead.
    """
    if with_dataset:
        ap.add_argument("--dataset", choices=DATASETS, default=None,
                        help="split, for the camera config; inferred from the results path "
                             "and from the run's camera.json when omitted")
    ap.add_argument("--camera-config", default=None,
                    help="HUGSIM <dataset>_camera.yaml, overriding --dataset")
    ap.add_argument("--hugsim-root", default=None,
                    help="HUGSIM checkout; defaults to $HUGSIM_ROOT")
    ap.add_argument("--allow-camera-mismatch", action="store_true",
                    help="warn instead of stopping when the camera config disagrees with the "
                         "run's own record; the overlay geometry will be wrong")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True,
                    help="results root, e.g. outputs/hugsim/nuscenes_ltf")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenes", default=None,
                    help="comma-separated scenario directory names; default is all of --root")
    ap.add_argument("--pick", default=None,
                    help="results_<dataset>.json: take the best and worst HD-Score per tier")
    ap.add_argument("--per-tier", type=int, default=2,
                    help="scenarios per tier for --pick (1 keeps only the best)")
    ap.add_argument("--fps", type=int, default=4,
                    help="the closed loop steps at 0.25 s, so 4 fps is real time")
    ap.add_argument("--max-frames", type=int, default=None)
    add_camera_args(ap)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] results root does not exist: {root}")
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "plan_log.pkl").is_file())
    if not dirs:
        print(f"[ERROR] no scenario under {root} has plan_log.pkl. Re-run with SAVE_VIZ=1.")
        return 1

    want = None
    if args.scenes:
        want = [s.strip() for s in args.scenes.split(",") if s.strip()]
    elif args.pick:
        want = pick_representative(Path(args.pick), args.per_tier)
        print(f"picked {len(want)} scenarios by tier: {want}")
    if want is not None:
        index = {p.name: p for p in dirs}
        missing = [w for w in want if w not in index]
        if missing:
            print(f"[warn] no plan_log.pkl for {missing}, skipping those")
        dirs = [index[w] for w in want if w in index]
    if not dirs:
        print("[ERROR] nothing left to render")
        return 1

    cam = resolve_camera(dirs[0], args.dataset,
                         Path(args.camera_config) if args.camera_config else None,
                         Path(args.hugsim_root) if args.hugsim_root else None,
                         strict=not args.allow_camera_mismatch)
    print(f"camera: {cam.name} from {cam.source} "
          f"(fx={cam.fx:.1f} fy={cam.fy:.1f} cx={cam.cx:.1f} cy={cam.cy:.1f})")
    print(f"rendering {len(dirs)} scenarios -> {out_dir}")

    n_ok = 0
    for scenario in dirs:
        ok, msg = build(scenario, out_dir / f"{scenario.name}.mp4", cam,
                        fps=args.fps, max_frames=args.max_frames)
        print(f"  [{'ok' if ok else 'FAIL'}] {scenario.name}: {msg}")
        n_ok += int(ok)
    print(f"{n_ok}/{len(dirs)} rendered")
    # Non-zero on any failure: a wrapper that only checks the exit code must not read a
    # partially rendered batch as a complete one.
    return 0 if n_ok == len(dirs) else 1


if __name__ == "__main__":
    sys.exit(main())
