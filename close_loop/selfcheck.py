#!/usr/bin/env python3
"""Offline assertion battery for the HUGSIM adapter.

Runs on CPU, without a checkpoint and without the simulator. Every check here covers a
conversion that fails silently -- a camera in the wrong slot, an inverted command mapping,
a history stride of 1, a flipped output axis. None of those raise; they only lower the
score, and a lower score on a closed-loop benchmark is indistinguishable from a weaker
model. Checking them offline costs seconds, whereas discovering them from a finished
436-scenario run costs the whole run.

Usage::

    python close_loop/selfcheck.py                                   # config + pure logic
    python close_loop/selfcheck.py --dump <outdir>/dumped_obs.pkl    # also check real frames

To produce a dump, run one scenario with ``HUGSIM_DUMP_OBS=5`` in the environment; the
client writes the first five real ``(obs, info)`` frames to ``dumped_obs.pkl`` in the
scenario's output directory. Real frames matter: a black rear camera, a command channel
that never changes and a render resolution below the model input are all invisible on
synthetic input.

Exit code is 1 if any check FAILs. WARNs are reported but do not fail the run, because
some of them are properties of the benchmark rather than defects (the KITTI-360 and Waymo
splits genuinely render no rear view).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from close_loop.hugsim_planner import (  # noqa: E402
    DEFAULT_CAMERA_MAP,
    DEFAULT_COMMAND_MAP,
    DEFAULT_SIM_DT_S,
    DEFAULT_TRAJECTORY_FRAME_RATE_HZ,
    HugsimPlanner,
    convert_absolute_to_relative_se2_array,
)

N_FAIL = 0
N_WARN = 0


def ok(msg: str) -> None:
    print(f"  ok   {msg}")


def warn(msg: str) -> None:
    global N_WARN
    N_WARN += 1
    print(f"  WARN {msg}")


def fail(msg: str) -> None:
    global N_FAIL
    N_FAIL += 1
    print(f"  FAIL {msg}")


def check(cond: bool, msg: str, level=fail) -> bool:
    (ok if cond else level)(msg)
    return bool(cond)


def load_config(path: Path) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(path))
    return OmegaConf.to_container(cfg, resolve=True)


def bare_planner(cfg: dict) -> HugsimPlanner:
    """A planner with the derived fields set but no model.

    ``__init__`` builds the network and restores a checkpoint, which needs a GPU and
    several gigabytes of weights. The conversions checked below are pure functions of the
    configuration, so constructing the object through ``__new__`` exercises exactly the
    code that runs in the simulator without any of that.
    """
    p = HugsimPlanner.__new__(HugsimPlanner)
    model_cfg = cfg["model"]
    data_cfg = cfg.get("data") or {}
    sim_cfg = cfg.get("hugsim") or {}
    camera_map = {
        str(k).lower(): str(v)
        for k, v in (sim_cfg.get("camera_map") or DEFAULT_CAMERA_MAP).items()
    }
    p.model_cams = [str(c).lower() for c in model_cfg["camera_names"]]
    p.hugsim_cams = [camera_map.get(c, "") for c in p.model_cams]
    p.command_map = [int(v) for v in (sim_cfg.get("command_map") or DEFAULT_COMMAND_MAP)]
    p.in_h, p.in_w = int(model_cfg["input_hw"][0]), int(model_cfg["input_hw"][1])
    p.n_hist = int(model_cfg["num_history_frames"])
    p.ego_dim = int(model_cfg["ego_status_dim"])
    p.horizon = int(model_cfg["trajectory_horizon"])
    sim_dt = float(sim_cfg.get("sim_dt_s", DEFAULT_SIM_DT_S))
    rate_hz = float(data_cfg.get("trajectory_frame_rate_hz", DEFAULT_TRAJECTORY_FRAME_RATE_HZ))
    p.model_interval_s = 1.0 / rate_hz
    p.history_stride = int(round(p.model_interval_s / sim_dt))
    p.buf_len = p.history_stride * (p.n_hist - 1) + 1
    p.sim_dt = sim_dt
    return p


# ---------------------------------------------------------------- config


def check_config(cfg_path: Path, cfg: dict, p: HugsimPlanner) -> None:
    print(f"[config] {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    check("/mnt/" not in text and "/data/" not in text,
          "config contains no site-specific absolute paths")

    check(all(p.hugsim_cams), f"every model camera is mapped: "
                              f"{dict(zip(p.model_cams, p.hugsim_cams))}")
    check(len(set(p.hugsim_cams)) == len(p.hugsim_cams),
          "the mapped HUGSIM cameras are distinct (two model slots reading the same "
          "render would waste a quarter of the input)")

    check(sorted(p.command_map) == [0, 1, 2],
          f"command_map {p.command_map} is a permutation of [0, 1, 2]")
    check(p.command_map == DEFAULT_COMMAND_MAP,
          f"command_map is {DEFAULT_COMMAND_MAP} (HUGSIM 0=right/1=left/2=straight vs "
          f"NAVSIM 0=left/1=straight/2=right)",
          level=warn)

    exact = p.model_interval_s / p.sim_dt
    check(abs(exact - round(exact)) < 1e-9,
          f"the model interval {p.model_interval_s:.3f}s is an exact multiple of the "
          f"simulator step {p.sim_dt:.3f}s (ratio {exact})")
    check(p.history_stride >= 1, f"history stride is {p.history_stride}")
    check(p.history_stride > 1,
          f"history stride is {p.history_stride}; with a stride of 1 the model observes a "
          f"world running at {p.model_interval_s / p.sim_dt:.0f}x slow motion",
          level=warn)

    check(p.ego_dim >= 8, f"ego_status_dim is {p.ego_dim} (slots 4..7 carry vx, vy, ax, ay)")
    check(p.horizon >= 1, f"trajectory_horizon is {p.horizon}")
    step_limit = int((cfg.get("hugsim") or {}).get("step_limit", 400))
    check(step_limit == 400,
          f"step_limit is {step_limit}; closed_loop.py ends an episode at cnt > 400, so any "
          f"other value makes the recorded StepLimit inference wrong",
          level=warn)
    print(f"  buffer length {p.buf_len} frames "
          f"({p.n_hist} history frames, stride {p.history_stride})")


# ---------------------------------------------------------------- pure logic


def check_commands(p: HugsimPlanner) -> None:
    print("[commands] HUGSIM -> NAVSIM one-hot")
    names = {0: "right", 1: "left", 2: "straight"}
    navsim_names = {0: "left", 1: "straight", 2: "right"}
    for raw in (0, 1, 2):
        st, nav, back = p._ego_status(
            {"command": raw, "ego_velo": 3.0, "accelerate": 0.5}
        )
        onehot = int(np.argmax(st[:4]))
        good = (back == raw and nav == p.command_map[raw] and onehot == nav
                and float(st[4]) == 3.0 and float(st[6]) == 0.5
                and float(st[5]) == 0.0 and float(st[7]) == 0.0)
        check(good,
              f"HUGSIM {raw} ({names[raw]}) -> NAVSIM {nav} ({navsim_names.get(nav, '?')}), "
              f"one-hot slot {onehot}, vx={st[4]:.1f}, ax={st[6]:.1f}")
    try:
        p._ego_status({"command": 7, "ego_velo": 0.0, "accelerate": 0.0})
        fail("an out-of-range command was accepted; it must raise so the client counts the "
             "failure instead of silently steering somewhere")
    except RuntimeError:
        ok("an out-of-range command raises")


def check_history(p: HugsimPlanner) -> None:
    print("[history] warm-up padding and stride")
    p.buf = [None]
    idx = p._history_indices()
    check(idx == [0] * p.n_hist,
          f"the first frame repeats the only frame available: {idx}")
    p.buf = [None] * p.buf_len
    idx = p._history_indices()
    want = [i * p.history_stride for i in range(p.n_hist)]
    check(idx == want, f"a full buffer is strided by {p.history_stride}: {idx}")
    check(idx[-1] == p.buf_len - 1, "the last history frame is the current frame")
    p.buf = [None] * (p.buf_len + 5)
    check(p._history_indices()[-1] == p.buf_len + 4,
          "a longer buffer still ends at the current frame")


def check_se2() -> None:
    print("[frames] world -> ego SE(2)")
    origin = np.array([10.0, -4.0, 0.7])
    rel = convert_absolute_to_relative_se2_array(origin, origin[None, :])
    check(np.allclose(rel, 0.0, atol=1e-12),
          "the origin maps to (0, 0, 0) in its own frame")

    # Heading 0 = world x forward. A point 5 m ahead has to land at ego (5, 0).
    rel = convert_absolute_to_relative_se2_array(
        np.array([0.0, 0.0, 0.0]), np.array([[5.0, 0.0, 0.0]])
    )
    check(np.allclose(rel[0], [5.0, 0.0, 0.0], atol=1e-9),
          f"5 m ahead with heading 0 -> ego {rel[0].round(3).tolist()}")

    # Facing +90 degrees (world +y), the same world point is now 5 m to the *right*, i.e.
    # ego y = -5. Getting this sign backwards mirrors every trajectory.
    rel = convert_absolute_to_relative_se2_array(
        np.array([0.0, 0.0, np.pi / 2]), np.array([[5.0, 0.0, 0.0]])
    )
    check(np.allclose(rel[0], [0.0, -5.0, -np.pi / 2], atol=1e-9),
          f"the same point with heading +90 deg -> ego {rel[0].round(3).tolist()} "
          f"(x forward, y left)")

    # Angles must wrap: a raw difference of 350 degrees is really -10.
    rel = convert_absolute_to_relative_se2_array(
        np.array([0.0, 0.0, -np.pi + 0.05]), np.array([[0.0, 0.0, np.pi - 0.05]])
    )
    check(abs(rel[0, 2] - (-0.1)) < 1e-9,
          f"heading difference wraps into [-pi, pi]: {rel[0, 2]:.4f} rad")


def ego_to_lidar(traj: np.ndarray) -> np.ndarray:
    """The conversion applied at the end of ``HugsimPlanner.plan``."""
    return np.stack([-traj[:, 1], traj[:, 0]], axis=1)


def check_output_frame() -> None:
    print("[frames] ego -> lidar output")
    straight = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    out = ego_to_lidar(straight)
    check(np.allclose(out, [[0.0, 1.0], [0.0, 2.0]]),
          f"driving straight ahead -> lidar {out.tolist()} (x right, y forward)")

    left = np.array([[1.0, 0.5, 0.0]])
    out = ego_to_lidar(left)
    check(out[0, 0] < 0 and out[0, 1] > 0,
          f"a left turn (ego y > 0) -> lidar x < 0: {out[0].round(3).tolist()}")

    # closed_loop.py runs plan_traj[:, [1, 0]] to get back (forward, right). Round-tripping
    # here is what proves the two conventions agree; a single missing negation turns every
    # left turn into a right turn with no error anywhere.
    back = out[:, [1, 0]]
    check(np.allclose(back[0], [left[0, 0], -left[0, 1]]),
          f"the simulator's plan_traj[:, [1, 0]] recovers (forward, right) = "
          f"{back[0].round(3).tolist()}")


# ---------------------------------------------------------------- real frames


def check_dump(dump_path: Path, p: HugsimPlanner) -> None:
    print(f"[dump] {dump_path}")
    with open(dump_path, "rb") as f:
        frames = pickle.load(f)
    if not check(len(frames) > 0, f"{len(frames)} frames in the dump"):
        return

    obs, info = frames[0]
    rgb = obs.get("rgb") or {}
    missing = [c for c in p.hugsim_cams if c not in rgb]
    check(not missing,
          f"every mapped camera is present in obs['rgb'] "
          f"(missing: {missing}; available: {sorted(rgb)})")

    for key in ("ego_box", "command", "ego_velo", "accelerate", "timestamp"):
        check(key in info, f"info['{key}'] exists")
    if "ego_box" in info:
        check(len(info["ego_box"]) >= 7,
              f"ego_box has {len(info['ego_box'])} fields (x, y, yaw are 0, 1, 6)")

    means = {}
    for cam in p.hugsim_cams:
        img = rgb.get(cam)
        if img is None:
            continue
        img = np.asarray(img)
        good_shape = img.ndim == 3 and img.shape[2] == 3
        check(good_shape, f"{cam}: shape {img.shape}")
        if not good_shape:
            continue
        check(img.dtype == np.uint8,
              f"{cam}: dtype {img.dtype} (the adapter divides by 255, so a float render "
              f"would be scaled twice)")
        check(img.shape[0] >= p.in_h and img.shape[1] >= p.in_w,
              f"{cam}: render {img.shape[1]}x{img.shape[0]} is at least the model input "
              f"{p.in_w}x{p.in_h}",
              level=warn)
        means[cam] = float(img.mean())

    if means:
        black = [c for c, m in means.items() if m < 1.0]
        print("  camera brightness: "
              + ", ".join(f"{c}={m:.1f}" for c, m in means.items()))
        check(len(black) < len(means),
              f"not every camera is black (black: {black})")
        if black:
            warn(f"{black} render as black. The KITTI-360 and Waymo splits render only the "
                 f"three front cameras, so this is expected there and has to be stated in "
                 f"the report; on nuScenes or PandaSet it means the camera mapping is wrong")
        else:
            ok("no camera renders as black")

    cmds = [int(np.asarray(i["command"]).ravel()[0]) for _, i in frames if "command" in i]
    if cmds:
        check(all(0 <= c < len(p.command_map) for c in cmds),
              f"every command is inside command_map: {sorted(set(cmds))}")
    ts = [float(i["timestamp"]) for _, i in frames if "timestamp" in i]
    if len(ts) > 1:
        dts = np.diff(ts)
        check(bool((dts > 0).all()), f"timestamps increase: {np.round(dts, 3).tolist()}")
        check(bool(np.allclose(dts, p.sim_dt, atol=1e-3)),
              f"the step is {p.sim_dt}s as configured; measured {np.round(dts, 4).tolist()}",
              level=warn)
    for _, i in frames:
        for key in ("ego_velo", "accelerate"):
            if key in i and not np.isfinite(np.asarray(i[key], dtype=np.float64)).all():
                fail(f"info['{key}'] is not finite: {i[key]!r}")
                break

    # The pure conversions again, on real values this time.
    try:
        st, nav, raw = p._ego_status(info)
        ok(f"a real frame converts: command {raw} -> {nav}, vx={st[4]:.2f}, ax={st[6]:.2f}")
    except Exception as exc:  # noqa: BLE001
        fail(f"converting a real frame raised: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(REPO / "configs" / "wa_jepa_hugsim.yaml"))
    ap.add_argument("--dump", default=None,
                    help="dumped_obs.pkl written by hugsim_client.py under HUGSIM_DUMP_OBS")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"[ERROR] config not found: {cfg_path}")
        return 1
    cfg = load_config(cfg_path)
    p = bare_planner(cfg)

    check_config(cfg_path, cfg, p)
    check_commands(p)
    check_history(p)
    check_se2()
    check_output_frame()
    if args.dump:
        dump_path = Path(args.dump)
        if dump_path.is_file():
            check_dump(dump_path, p)
        else:
            print(f"[dump] {dump_path} not found; run one scenario with HUGSIM_DUMP_OBS=5 to "
                  f"produce it")
    else:
        print("[dump] skipped. The checks above use synthetic values only; a black camera, a "
              "constant command channel and a too-small render can only be seen on real "
              "frames (--dump).")

    print(f"\n{N_FAIL} failures, {N_WARN} warnings")
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
