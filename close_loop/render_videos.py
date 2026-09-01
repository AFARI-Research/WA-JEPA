"""Render every archived scenario of a run into videos, in parallel, plus an index.

``make_video.py`` handles one scenario; this drives it across a whole split (or all four)
and writes an ``index.md`` grouping the results by split and difficulty tier with each
scenario's scores, so a reviewer can go from a number in ``RESULTS.md`` to the video of
that exact run.

Design notes worth keeping:

* Frames stream into the encoder one at a time (``make_video.iter_frames`` is a
  generator). A 400-step scenario composes to roughly 4 MB per frame; collecting a list
  first costs 1.6 GB per worker, and eight workers then exhaust the machine.
* One scenario failing must not take the batch down, so every job catches its own
  exceptions and reports them in the summary. Silence would be worse: a missing video is
  invisible in ``index.md`` unless it is listed as a failure.
* The exit code is non-zero if any scenario failed. A wrapper that only checks the exit
  code must not read a partial batch as a complete one.
* The camera config is resolved per split and cross-checked against each run's own
  ``viz/camera.json``, because the four splits do not share camera geometry. See
  ``make_video.py`` for the table.

Rendering is CPU-only and reads nothing but the archive, so it can run on any machine that
has the output directory -- no GPU, no checkpoint, no simulator.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from close_loop.aggregate_results import TIER_BUCKETS, split_dir_name, tier_bucket  # noqa: E402
from close_loop.make_video import (  # noqa: E402
    DATASETS,
    add_camera_args,
    build,
    read_scores,
    resolve_camera,
)

MIN_VIDEO_BYTES = 10000


def render_one(job):
    """Subprocess entry point. Returns a tuple so nothing but plain data crosses processes."""
    dataset, name, scenario_dir, out_path, cam, fps, max_frames = job
    t0 = time.time()
    try:
        ok, msg = build(Path(scenario_dir), Path(out_path), cam, fps=fps, max_frames=max_frames)
    except Exception as exc:  # noqa: BLE001 - one bad scenario must not end the batch
        ok, msg = False, f"{type(exc).__name__}: {exc}"
    return dataset, name, ok, msg, time.time() - t0


def collect_jobs(args, out_root: Path, log=print):
    """Build the job list, resolving one camera model per split."""
    jobs, skipped = [], 0
    for dataset in args.datasets:
        results_root = (Path(args.root) if args.root
                        else Path(args.output_dir) / f"{dataset}_ltf")
        if not results_root.is_dir():
            log(f"[warn] no results directory for {dataset}: {results_root}")
            continue
        scenarios = sorted(p for p in results_root.iterdir()
                           if p.is_dir() and (p / "plan_log.pkl").is_file())
        if not scenarios:
            log(f"[warn] {results_root} has no scenario with plan_log.pkl; the run needs "
                f"SAVE_VIZ=1 for videos")
            continue

        # Resolved once per split, from the first scenario, and reused: it is the same
        # camera for the whole split, and the check prints one message instead of 127.
        cam = resolve_camera(
            scenarios[0],
            dataset,
            Path(args.camera_config) if args.camera_config else None,
            Path(args.hugsim_root) if args.hugsim_root else None,
            strict=not args.allow_camera_mismatch,
            log=log,
        )
        log(f"[{dataset}] camera from {cam.source} (fx={cam.fx:.1f} cy={cam.cy:.1f}), "
            f"{len(scenarios)} scenarios archived")

        for scenario in scenarios:
            _, tier = split_dir_name(scenario.name)
            bucket = tier_bucket(tier)
            out_path = out_root / dataset / bucket / f"{scenario.name}.mp4"
            if args.skip_existing and out_path.is_file() and out_path.stat().st_size > MIN_VIDEO_BYTES:
                skipped += 1
                continue
            jobs.append((dataset, scenario.name, str(scenario), str(out_path), cam,
                         args.fps, args.max_frames))
    if args.limit:
        jobs = jobs[:args.limit]
    return jobs, skipped


def _index_sort_key(item):
    """Best HD-Score first. Unscored scenarios sort last rather than being dropped: a
    video whose ``eval.json`` is missing is exactly the one worth watching."""
    score = item[1].get("hdscore")
    return -(score if score is not None else -1.0)


def write_index(out_root: Path, rendered: list[tuple[str, str]], args) -> Path:
    """``index.md``: every rendered video with its scores, grouped by split and tier."""
    by_dataset: dict[str, dict[str, list[tuple[str, dict]]]] = {}
    for dataset, name in rendered:
        results_root = (Path(args.root) if args.root
                        else Path(args.output_dir) / f"{dataset}_ltf")
        _, tier = split_dir_name(name)
        scores = read_scores(results_root / name)
        by_dataset.setdefault(dataset, {}).setdefault(tier_bucket(tier), []).append((name, scores))

    lines = [
        "# HUGSIM closed-loop videos",
        "",
        f"{len(rendered)} scenarios, {args.fps} fps. The closed loop steps at 0.25 s, so "
        f"{'4 fps is real time' if args.fps == 4 else '4 fps would be real time'}.",
        "",
        "Each frame shows `CAM_FRONT` with the planned trajectory projected onto the ground, "
        "a bird's-eye panel with the plan, the other agents and the path driven so far, and a "
        "status bar with step, time, speed, navigation command and the scenario's scores.",
        "",
        "A frame labelled `FALLBACK: inference failed` is the client's braking plan, not the "
        "model's; `history padded` marks the warm-up steps before the history buffer is full.",
        "",
    ]
    for dataset in args.datasets:
        buckets = by_dataset.get(dataset)
        if not buckets:
            continue
        lines += [f"## {dataset}", ""]
        for bucket in TIER_BUCKETS + sorted(set(buckets) - set(TIER_BUCKETS)):
            rows = buckets.get(bucket)
            if not rows:
                continue
            lines += [f"### {bucket} ({len(rows)})", "",
                      "| Video | HD-Score | PDMS | RC | NC | DAC |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for name, scores in sorted(rows, key=_index_sort_key):
                cells = ["-" if scores.get(m) is None else f"{scores[m]:.3f}"
                         for m in ("hdscore", "pdms", "rc", "nc", "dac")]
                lines.append(f"| [{name}]({dataset}/{bucket}/{name}.mp4) | "
                             + " | ".join(cells) + " |")
            lines.append("")
    path = out_root / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-dir", default="outputs/hugsim",
                    help="benchmark output root holding <dataset>_ltf directories")
    ap.add_argument("--root", default=None,
                    help="one results directory, overriding --output-dir; only "
                         "valid with a single --datasets entry")
    ap.add_argument("--video-dir", default=None,
                    help="output root (default <output-dir>/videos)")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help=">0 renders only the first N (smoke test)")
    ap.add_argument("--skip-existing", action="store_true")
    add_camera_args(ap, with_dataset=False)
    args = ap.parse_args()

    if args.root and len(args.datasets) != 1:
        print("[ERROR] --root names a single results directory, so pass exactly one "
              "--datasets value with it (the camera config is per split).")
        return 1
    if args.camera_config and len(args.datasets) != 1:
        # One config applied to four splits renders three of them with the wrong field of
        # view, camera height or pitch, and every frame still looks like a trajectory
        # overlay. Pass --hugsim-root instead and let each split resolve its own.
        print("[ERROR] --camera-config is one split's camera. Pass a single --datasets "
              "value with it, or use --hugsim-root to resolve one config per split.")
        return 1

    out_root = Path(args.video_dir) if args.video_dir else \
        Path(args.output_dir) / "videos"
    out_root.mkdir(parents=True, exist_ok=True)

    jobs, skipped = collect_jobs(args, out_root)
    if not jobs:
        print(f"[ERROR] nothing to render (skipped {skipped} existing). Check --output-dir "
              f"and that the run used SAVE_VIZ=1.")
        return 1
    print(f"rendering {len(jobs)} scenarios with {args.workers} workers "
          f"(skipped {skipped} existing) -> {out_root}", flush=True)

    t0, done, failed = time.time(), [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render_one, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            dataset, name, ok, msg, dt = future.result()
            (done if ok else failed).append((dataset, name, msg))
            elapsed = time.time() - t0
            print(f"[{n}/{len(futures)}] {'ok' if ok else 'FAIL'} {dataset}/{name}  {msg}  "
                  f"({dt:.0f}s, elapsed {elapsed / 60:.1f}m, "
                  f"eta {elapsed / n * (len(futures) - n) / 60:.1f}m)", flush=True)

    print(f"\n{len(done)}/{len(jobs)} rendered in {(time.time() - t0) / 60:.1f} minutes, "
          f"{len(failed)} failed")
    for dataset, name, msg in failed:
        print(f"  FAIL {dataset}/{name}: {msg}")

    index = write_index(out_root, [(d, n) for d, n, _ in done], args)
    print(f"index: {index}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
