#!/usr/bin/env python3
"""Rewrite the official HUGSIM benchmark scenario YAMLs into the form the simulator reads.

## Why this step exists

The official benchmark ships with the data package (``<data_root>/<dataset>/*.yaml``:
nuscenes 88, kitti360 113, waymo 108, pandaset 127; 436 in total). Field 5 of each
``plan_list`` entry is written as::

    2024_07_05_16_16_38/postprocess/shadow.pth

while ``sim/utils/plan.py`` appends another level to it::

    ckpts[agent] = os.path.join(model_path, 'gs.pth')
    open(os.path.join(model_path, 'wlh.json'))

so the simulator looks for ``<id>/postprocess/shadow.pth/gs.pth``. ``shadow.pth`` is a
file, so that is always a ``FileNotFoundError``; the real 3DRealCar layout is
``<id>/gs.pth`` plus ``<id>/wlh.json``. Upstream's own
``eval_render/export_multiple_scenes.py`` does ``actors[-3].replace("/postprocess/shadow.pth", "")``
(index -3 of the 8-tuple is field 5), which confirms the suffix is meant to be stripped:
the published scenario files and the simulator code disagree, this is not a usage error.

## Why plain text substitution instead of a YAML round-trip

A YAML round-trip reorders keys and rewrites scalars. The result is equivalent to
OmegaConf but no longer auditable at a glance. Text substitution guarantees the output
is byte-identical to the input except for the target substring.

## Stripping the suffix requires adding quotes

3DRealCar ids look like ``2024_07_05_16_16_38``. YAML 1.1 treats ``_`` as a digit
separator, so a bare ``2024_07_05_16_16_38`` parses as the integer ``20240705161638``
and the path silently becomes ``3DRealCar/20240705161638/gs.pth`` -- no YAML error, just
a missing asset much later. The original files get away with it because the
``/postprocess/shadow.pth`` suffix contains slashes and forces a string. Removing the
suffix walks straight into the integer trap, which is why upstream's own example writes
``- "2024_07_05_10_58_02"`` with quotes. This script emits ``"<id>"`` and then asserts
the parsed type is ``str``.

## Scoring

Only agent asset paths change. ``start_*``, ``mode``, the ``plan_list`` structure and
``load_HD_map`` are untouched, and ``closed_loop.py`` / ``traj2control`` /
``hugsim_evaluate`` are not modified at all, so scores stay comparable to the published
baselines.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# `<id>/postprocess/shadow.pth` -> `"<id>"`. The id contains only digits and
# underscores; the quotes are mandatory (see the module docstring).
ASSET_RE = re.compile(r'([0-9_]+)/postprocess/shadow\.pth')

DATASETS = ["nuscenes", "kitti360", "waymo", "pandaset"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="HUGSIM data package root (contains <dataset>/*.yaml, scenes/, 3DRealCar/)")
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parent / "configs" / "benchmark"),
                    help="Where to write the rewritten scenarios")
    ap.add_argument("--realcar", default=None, help="Defaults to <data-root>/3DRealCar")
    ap.add_argument("--scenes-root", default=None, help="Defaults to <data-root>/scenes")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()

    try:
        import yaml
    except ImportError:
        print("pyyaml is required", file=sys.stderr)
        return 1

    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None
        print("! omegaconf missing, skipping the OmegaConf cross-check "
              "(run this inside the simulator environment)", file=sys.stderr)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    realcar = Path(args.realcar) if args.realcar else data_root / "3DRealCar"
    scenes_root = Path(args.scenes_root) if args.scenes_root else data_root / "scenes"
    total_bad = 0

    for ds in args.datasets:
        src_dir = data_root / ds
        if not src_dir.is_dir():
            print(f"[{ds}] source directory missing, skipped: {src_dir}")
            continue
        out_dir = out_root / ds
        out_dir.mkdir(parents=True, exist_ok=True)

        yamls = sorted(src_dir.glob("*.yaml"))
        n_written = n_stripped = 0
        missing_assets: list[str] = []
        missing_scenes: set[str] = set()

        for f in yamls:
            raw = f.read_text()
            new = ASSET_RE.sub(r'"\1"', raw)
            if new != raw:
                n_stripped += 1

            # Every asset referenced after the rewrite must exist. A dangling path is
            # invisible in the YAML and would only surface halfway through a run, so turn
            # it into a countable result here.
            cfg = yaml.safe_load(new)
            for actor in cfg.get("plan_list") or []:
                if not isinstance(actor[5], str):
                    missing_assets.append(
                        f"{f.name}: plan_list[5] parsed as {type(actor[5]).__name__}"
                        f"({actor[5]!r}), expected str -- the quotes were lost"
                    )
                    continue
                asset_dir = realcar / actor[5]
                for need in ("gs.pth", "wlh.json"):
                    if not (asset_dir / need).is_file():
                        missing_assets.append(f"{f.name}: {asset_dir / need}")
            scene_dir = scenes_root / ds / str(cfg["scene_name"])
            if not (scene_dir / "ground_param.pkl").is_file():
                missing_scenes.add(cfg["scene_name"])

            (out_dir / f.name).write_text(new)
            n_written += 1

            # Cross-check with OmegaConf as well. `closed_loop.py` loads scenarios with
            # OmegaConf, whose scalar rules are not guaranteed to match yaml.safe_load;
            # validating only with safe_load does not prove the real consumer agrees.
            if OmegaConf is not None:
                oc = OmegaConf.load(out_dir / f.name)
                for actor in oc.get("plan_list") or []:
                    if not isinstance(actor[5], str):
                        missing_assets.append(
                            f"{f.name}: OmegaConf reads plan_list[5]={actor[5]!r} "
                            f"({type(actor[5]).__name__}), expected str"
                        )

        runnable = [
            f for f in sorted(out_dir.glob("*.yaml"))
            if yaml.safe_load(f.read_text())["scene_name"] not in missing_scenes
        ]
        print(f"[{ds}] wrote {n_written} scenarios ({n_stripped} suffixes stripped) -> {out_dir}")
        if missing_scenes:
            print(f"  ! {len(missing_scenes)} scenes have no 3DGS data: {sorted(missing_scenes)}")
        print(f"  runnable (3DGS data present): {len(runnable)} / {n_written}")
        if missing_assets:
            total_bad += len(missing_assets)
            print(f"  !! {len(missing_assets)} missing agent assets:")
            for m in missing_assets[:5]:
                print(f"     {m}")

        # Persist the runnable list so the benchmark runner does not recompute it.
        (out_dir / "runnable.txt").write_text("\n".join(str(p) for p in runnable) + "\n")

    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
