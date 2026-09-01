"""HUGSIM AD client: talks to ``closed_loop.py`` over FIFOs and answers with waypoints.

Upstream protocol (``closed_loop.py``, verified line by line):

  * ``closed_loop.py`` launches this process *first* and only then builds the
    environment and the two FIFOs, so the client has to wait for ``obs_pipe`` and
    ``plan_pipe`` to appear. Loading the model during that window overlaps with the
    3DGS load and costs nothing.
  * Each step: the simulator writes ``pickle((obs, info))`` to ``obs_pipe`` and reads
    ``pickle(plan)`` from ``plan_pipe``.
  * Teardown: the simulator writes ``pickle('Done')`` to ``obs_pipe``.
  * ``closed_loop.py`` runs ``plan_traj[:, [1, 0]]`` unconditionally, so a ``None`` plan
    raises a ``TypeError`` that takes the whole scenario down. This client never
    returns ``None``.

Failure policy: never degrade silently. If a single inference step raises, a zeroed
(braking) plan keeps the process alive, but the failure is counted in
``planner_stats.json`` so ``aggregate_results.py`` can mark that scenario's score
untrustworthy. Reporting "this scenario is void" is strictly better than handing back
a score that looks fine.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    # stdout/stderr are redirected by launch_ad.py into <output>/output.txt; flush keeps
    # the interleaving with the simulator's own log readable.
    print(f"[ad {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_pipes(obs_pipe: str, plan_pipe: str, timeout: float = 3600.0) -> bool:
    """Wait for the simulator to create both FIFOs. Loading 3DGS can take minutes."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(obs_pipe) and os.path.exists(plan_pipe):
            log(f"FIFOs ready after {time.time() - t0:.1f}s")
            return True
        time.sleep(0.5)
    return False


def _mosaic(tiles, cols: int = 2):
    """Tile the camera images the model actually reads, for eyeball verification.

    The grid must not assume a full last row: with an odd camera count the final row
    holds one tile and ``concatenate`` on axis 0 raises on the width mismatch. That
    exception would be swallowed by the visualisation ``try`` block below, leaving the
    input archive empty for the entire scenario -- discovered only when the archive is
    needed. Pad the last row instead.
    """
    import cv2

    h, w = tiles[0].shape[:2]
    small = [cv2.resize(t, (w // 2, h // 2)) for t in tiles]
    sh, sw = small[0].shape[:2]
    rows = []
    for i in range(0, len(small), cols):
        chunk = small[i:i + cols]
        if len(chunk) < cols:
            pad = np.zeros((sh, sw, 3), dtype=small[0].dtype)
            chunk = chunk + [pad] * (cols - len(chunk))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


def _drain_with_stop(obs_pipe: str, plan_pipe: str) -> None:
    """Walk the protocol to its end after a fatal error.

    Without this, ``closed_loop.py`` blocks forever in ``open(plan_pipe)`` and the
    scenario occupies its GPU slot until the whole run is killed.
    """
    while True:
        try:
            with open(obs_pipe, "rb") as p:
                raw = p.read()
            if not raw:
                return
            if isinstance(pickle.loads(raw), str):
                return
            with open(plan_pipe, "wb") as p:
                p.write(pickle.dumps(np.zeros((8, 2), dtype=np.float64)))
        except Exception:
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Scenario output directory (FIFOs live here)")
    ap.add_argument("--repo", required=True, help="Repository root")
    ap.add_argument("--config", required=True, help="Inference config, e.g. configs/wa_jepa_hugsim.yaml")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint directory or .pt file")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--camera-config", default="",
                    help="HUGSIM <dataset>_camera.yaml in use; recorded in viz/camera.json so "
                         "make_video.py can find the split's geometry again")
    ap.add_argument("--save-viz", type=int, default=0,
                    help="Archive front camera, model inputs and per-step plans")
    args = ap.parse_args()

    output = args.output
    obs_pipe = os.path.join(output, "obs_pipe")
    plan_pipe = os.path.join(output, "plan_pipe")

    # Dump the first N *real* (obs, info) frames so `selfcheck.py` can re-run assertions
    # offline. Real frames matter: black cameras, a constant command channel and a wrong
    # render resolution only show up on real observations. Synthetic input hides them.
    dump_obs_n = int(os.environ.get("HUGSIM_DUMP_OBS", "0"))
    dumped = []

    repo_root = str(Path(args.repo).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    stats = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "n_steps": 0,
        "n_failures": 0,
        "n_hist_pad": 0,
        "failures": [],
        "latency_ms": [],
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    viz_dir = Path(output) / "viz"
    if args.save_viz:
        (viz_dir / "front").mkdir(parents=True, exist_ok=True)
        (viz_dir / "input").mkdir(parents=True, exist_ok=True)
    plan_log = []

    try:
        log("loading model")
        t0 = time.time()
        from close_loop.hugsim_planner import HugsimPlanner

        planner = HugsimPlanner(
            repo_root=repo_root,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            log=log,
        )
        stats["model_load_s"] = round(time.time() - t0, 1)
        log(f"model ready in {stats['model_load_s']}s")
    except Exception:
        log("model failed to load, this scenario cannot run:\n" + traceback.format_exc())
        stats["fatal"] = traceback.format_exc()
        with open(os.path.join(output, "planner_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        if wait_for_pipes(obs_pipe, plan_pipe):
            _drain_with_stop(obs_pipe, plan_pipe)
        return 1

    if not wait_for_pipes(obs_pipe, plan_pipe):
        log("FIFOs never appeared, giving up")
        return 1

    import cv2

    while True:
        with open(obs_pipe, "rb") as p:
            raw = p.read()
        if not raw:
            log("obs_pipe returned no data, treating as end of episode")
            break
        data = pickle.loads(raw)
        if isinstance(data, str):
            log(f"received control signal {data!r}, finishing")
            break
        obs, info = data

        if len(dumped) < dump_obs_n:
            dumped.append((obs, info))
            if len(dumped) == dump_obs_n:
                dump_path = os.path.join(output, "dumped_obs.pkl")
                with open(dump_path, "wb") as f:
                    pickle.dump(dumped, f)
                log(f"dumped {dump_obs_n} real (obs, info) frames -> {dump_path}")

        t0 = time.time()
        try:
            plan = planner.plan(obs, info)
            if not (isinstance(plan, np.ndarray) and plan.ndim == 2 and plan.shape[1] == 2):
                raise RuntimeError(f"illegal waypoint shape: {getattr(plan, 'shape', type(plan))}")
            if not np.isfinite(plan).all():
                raise RuntimeError("waypoints contain NaN/Inf")
        except Exception:
            tb = traceback.format_exc()
            stats["n_failures"] += 1
            stats["failures"].append({"step": stats["n_steps"], "traceback": tb})
            log(f"!!! inference failed at step {stats['n_steps']}, falling back to a braking "
                f"plan (this scenario's score is not trustworthy):\n{tb}")
            # An all-zero plan means "stay put", which traj2control turns into hard braking.
            # The process survives, and the failure is already counted.
            plan = np.zeros((planner.horizon, 2), dtype=np.float64)
        latency_ms = (time.time() - t0) * 1000.0
        stats["latency_ms"].append(round(latency_ms, 1))

        with open(plan_pipe, "wb") as p:
            p.write(pickle.dumps(plan))

        if args.save_viz:
            try:
                step = stats["n_steps"]
                # Written once. `make_video.py` projects the plan onto the render, which
                # needs the camera the simulator actually used; the four splits do not
                # share one. `cam_rect` is not in `info`, so the config path is recorded
                # alongside the parameters that are: with both, the renderer can locate
                # the right config and verify it, instead of being told which split this
                # was and drawing a plausible overlay with the wrong geometry.
                cam_json = viz_dir / "camera.json"
                if not cam_json.is_file():
                    camera_yaml = args.camera_config or os.environ.get("CAMERA_YAML", "")
                    name = Path(camera_yaml).name
                    record = {
                        "_camera_config": camera_yaml,
                        "_dataset": name[:-len("_camera.yaml")]
                        if name.endswith("_camera.yaml") else "",
                    }
                    record.update({
                        name: {
                            "intrinsic": {k: float(v) for k, v in p["intrinsic"].items()},
                            "v2c": np.asarray(p["v2c"], dtype=np.float64).tolist(),
                        }
                        for name, p in info["cam_params"].items()
                    })
                    cam_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
                front = obs["rgb"]["CAM_FRONT"]
                cv2.imwrite(
                    str(viz_dir / "front" / f"{step:04d}.jpg"),
                    cv2.cvtColor(front, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                )
                mosaic = _mosaic([obs["rgb"][c] for c in planner.hugsim_cams])
                cv2.imwrite(
                    str(viz_dir / "input" / f"{step:04d}.jpg"),
                    cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                )
                plan_log.append(
                    {
                        "step": step,
                        "traj_lidar": plan.tolist(),
                        # Whatever the planner recorded for this step. These have to be
                        # chosen before the run: noticing an odd score after 436 scenarios
                        # and having no per-step record means re-running everything.
                        "planner": planner.traj_log[-1] if planner.traj_log else None,
                        "ego_box": list(info["ego_box"]),
                        "obj_boxes": [list(b) for b in info["obj_boxes"]],
                        "command": int(np.asarray(info["command"]).ravel()[0]),
                        "ego_velo": float(np.asarray(info["ego_velo"]).ravel()[0]),
                        "timestamp": float(info["timestamp"]),
                    }
                )
            except Exception:
                # Visualisation must never affect the evaluation; note it and carry on.
                log("visualisation archive failed (evaluation unaffected):\n" + traceback.format_exc())

        stats["n_steps"] += 1
        if stats["n_steps"] % 20 == 0:
            log(f"{stats['n_steps']} steps, last latency {latency_ms:.0f}ms, "
                f"{stats['n_failures']} failures")

    stats["n_hist_pad"] = planner.n_warmup_pad
    stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if stats["latency_ms"]:
        lm = np.asarray(stats["latency_ms"])
        stats["latency_summary"] = {
            "mean_ms": round(float(lm.mean()), 1),
            "p50_ms": round(float(np.percentile(lm, 50)), 1),
            "p95_ms": round(float(np.percentile(lm, 95)), 1),
            "max_ms": round(float(lm.max()), 1),
        }

    # Behaviour summary: the first thing to look at when a score is surprising. All of it
    # is derived from traj_log, no extra inference. It lives in planner_stats.json because
    # plan_log.pkl is tens of MB per scenario while this is tens of bytes, so it can be
    # aggregated across all 436 scenarios.
    try:
        tl = planner.traj_log
        if tl:
            # Command histogram: if one split is almost entirely a single value, the
            # command channel is effectively not driving anything.
            hist: dict[str, int] = {}
            for entry in tl:
                key = str(entry.get("hugsim_command"))
                hist[key] = hist.get(key, 0) + 1
            stats["command_hist"] = hist
            ends = np.asarray(
                [e["traj_ego"][-1] for e in tl if e.get("traj_ego")], dtype=np.float64
            )
            if ends.size:
                stats["behavior"] = {
                    # Longitudinal displacement of the last waypoint should be the same
                    # order as ego_velo x horizon; if it is not, suspect the ego_status
                    # slots or the coordinate frame.
                    "fwd_end_mean_m": round(float(ends[:, 0].mean()), 3),
                    "lat_end_mean_m": round(float(ends[:, 1].mean()), 3),
                    "lat_end_abs_max_m": round(float(np.abs(ends[:, 1]).max()), 3),
                    # A high share of >2 m lateral offsets together with mostly-background
                    # collisions points at a frame or heading mismatch.
                    "frac_lat_gt_2m": round(float((np.abs(ends[:, 1]) > 2.0).mean()), 3),
                }
    except Exception:
        log("behaviour summary failed (evaluation unaffected):\n" + traceback.format_exc())

    with open(os.path.join(output, "planner_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    if plan_log:
        with open(os.path.join(output, "plan_log.pkl"), "wb") as f:
            pickle.dump(plan_log, f)
    log(f"done: {stats['n_steps']} steps, {stats['n_failures']} failures, "
        f"{stats['n_hist_pad']} padded-history steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
