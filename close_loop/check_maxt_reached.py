#!/usr/bin/env python3
"""Check whether any rollout actually ran long enough for ``max_t`` to matter.

``close_loop/run_upstream_compat.py`` makes fifteen PandaSet scenarios runnable by
accepting the ``max_t`` argument their ``ConstantPlanner`` entry passes and ignoring it.
Ignoring is not a guess about the semantics, because every reading of ``max_t`` ("stop
after max_t", "despawn after max_t", "only plan that far ahead") agrees with the
argument-less constant-velocity behaviour for all ``t < max_t``. A rollout that ends
before ``max_t`` is therefore byte-identical to whatever upstream intended, and the
ignoring is invisible in the score.

That is a statement about the run, not about the code, so it has to be checked against the
run. This script reads the step count of every affected scenario and compares it with
``max_t / dt``:

* no scenario at the limit -- ignoring ``max_t`` was a provable no-op for this run, and the
  fifteen scores can be reported alongside the other 421 without a footnote;
* some scenario at the limit -- those scores depend on the unknown semantics. They are not
  wrong, but they have to be reported separately instead of folded into the mean, and this
  script exits non-zero so a pipeline notices.

``dt`` is read from the same kinematic config the simulator uses (``hug_sim.py`` loads it),
rather than hardcoded, because the step limit is ``max_t / dt`` and a changed ``dt`` moves
the threshold. ``max_t`` is read per scenario from its yaml, so a data package that changes
the value or adds it to more scenarios is picked up without editing this file.

Usage::

    python close_loop/check_maxt_reached.py --results-root <dir>            # all splits
    python close_loop/check_maxt_reached.py --results-root <dir> \\
        --dataset pandaset --kinematic-yaml <hugsim>/configs/sim/kinematic.yaml

``--results-root`` defaults to ``$HUGSIM_OUTPUT_ROOT`` and ``--kinematic-yaml`` to
``$KINEMATIC_YAML`` or ``$HUGSIM_ROOT/configs/sim/kinematic.yaml``, which is what
``scripts/evaluation/run_hugsim_benchmark.sh`` exports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_results import scenario_dir, scenario_yamls, yaml_scalar  # noqa: E402

DATASETS = ["nuscenes", "kitti360", "waymo", "pandaset"]
DEFAULT_DT = 0.25

# `max_t` is a controller argument nested in the plan_list entry, not a top-level scalar:
#
#     plan_list:
#     - - 0.7
#       ...
#       - ConstantPlanner
#       - max_t: 25
#
# so it is matched anywhere in the file rather than through yaml_scalar().
MAX_T_RE = re.compile(r"\bmax_t:\s*([0-9]+(?:\.[0-9]+)?)")


def read_dt(kinematic_yaml: Path | None) -> tuple[float, str]:
    """``dt`` from the simulator's kinematic config, or the upstream default."""
    if kinematic_yaml is not None and kinematic_yaml.is_file():
        value = yaml_scalar(kinematic_yaml.read_text(), "dt")
        if value is not None:
            return float(value), str(kinematic_yaml)
    return DEFAULT_DT, f"upstream default (no readable dt in {kinematic_yaml})"


def affected(scen_dir: Path) -> list[dict]:
    """Scenarios whose yaml passes ``max_t`` to a controller, with their output dir name.

    The output directory is ``<scene_name>_<mode>``, which is not derivable from the yaml
    file name -- see ``aggregate_results.log_index`` for why.
    """
    rows = []
    for y in scenario_yamls(scen_dir):
        if not y.is_file():
            continue
        text = y.read_text()
        m = MAX_T_RE.search(text)
        if m is None:
            continue
        scene, mode = yaml_scalar(text, "scene_name"), yaml_scalar(text, "mode")
        if scene is None or mode is None:
            continue
        rows.append({"yaml": y.name, "dir": f"{scene}_{mode}", "max_t": float(m.group(1))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default=os.environ.get("HUGSIM_OUTPUT_ROOT"),
                    help="directory holding the <scene>_<mode> output dirs "
                         "(default: $HUGSIM_OUTPUT_ROOT)")
    ap.add_argument("--dataset", choices=DATASETS, action="append",
                    help="split to check; repeatable, default all four")
    ap.add_argument("--scenario-dir", help="override the scenario config dir of a single split")
    ap.add_argument("--kinematic-yaml", default=os.environ.get("KINEMATIC_YAML"),
                    help="simulator kinematic config to read dt from "
                         "(default: $KINEMATIC_YAML, else $HUGSIM_ROOT/configs/sim/kinematic.yaml)")
    ap.add_argument("--json-out", help="also write the result as json")
    args = ap.parse_args()

    if not args.results_root:
        ap.error("pass --results-root or set HUGSIM_OUTPUT_ROOT")
    out_root = Path(args.results_root)

    kinematic = args.kinematic_yaml
    if not kinematic and os.environ.get("HUGSIM_ROOT"):
        kinematic = str(Path(os.environ["HUGSIM_ROOT"]) / "configs" / "sim" / "kinematic.yaml")
    dt, dt_source = read_dt(Path(kinematic) if kinematic else None)

    datasets = args.dataset or DATASETS
    rows, reached, missing = [], [], []
    for ds in datasets:
        scen_dir = scenario_dir(ds, args.scenario_dir if len(datasets) == 1 else None)
        if not scen_dir.is_dir():
            continue
        for item in affected(scen_dir):
            d = out_root / item["dir"]
            stats, ev = d / "planner_stats.json", d / "eval.json"
            if not (stats.is_file() and ev.is_file()):
                missing.append({"dataset": ds, **item})
                continue
            st = json.loads(stats.read_text())
            e = json.loads(ev.read_text())
            n_steps = st["n_steps"]
            limit = item["max_t"] / dt
            row = {"dataset": ds, "scenario": item["yaml"], "dir": item["dir"],
                   "max_t": item["max_t"], "step_limit": limit, "n_steps": n_steps,
                   "t_end_s": n_steps * dt, "reached_max_t": n_steps >= limit,
                   "n_failures": st.get("n_failures", 0),
                   "hdscore": e.get("hdscore"), "pdms": e.get("pdms")}
            rows.append(row)
            if row["reached_max_t"]:
                reached.append(row)

    print(f"dt = {dt} s, from {dt_source}")
    print(f"{len(rows)} scenario(s) pass max_t and produced a result; "
          f"{len(missing)} produced none\n")
    if rows:
        print(f"{'scenario':34s} {'max_t':>6s} {'limit':>6s} {'steps':>6s} "
              f"{'t_end':>7s} {'at limit':>9s} {'fallbacks':>10s} {'HD-S':>7s}")
        for r in sorted(rows, key=lambda x: -x["n_steps"]):
            hd = "-" if r["hdscore"] is None else f"{r['hdscore']:.4f}"
            print(f"{r['scenario']:34s} {r['max_t']:6.1f} {r['step_limit']:6.0f} "
                  f"{r['n_steps']:6d} {r['t_end_s']:7.2f} "
                  f"{'YES' if r['reached_max_t'] else 'no':>9s} {r['n_failures']:10d} {hd:>7s}")

    if missing:
        print(f"\n{len(missing)} scenario(s) have no eval.json / planner_stats.json and are "
              f"not counted:")
        for m in missing:
            print(f"    {m['dataset']}/{m['yaml']}")

    print()
    if not rows:
        print("nothing to check: no scenario in the selected splits passes max_t, or none "
              "of them produced results under --results-root.")
        return 0
    if reached:
        print(f"{len(reached)}/{len(rows)} scenario(s) ran to the step limit, so their scores "
              f"DO depend on the max_t semantics upstream never implemented. Ignoring max_t "
              f"is not a no-op for these; report them separately:")
        for r in reached:
            print(f"    {r['scenario']}  {r['n_steps']} steps >= {r['step_limit']:.0f}")
    else:
        print(f"all {len(rows)}/{len(rows)} scenario(s) ended before their step limit, so "
              f"ignoring max_t was a provable no-op for this run: every reading of max_t "
              f"only differs after t > max_t, and no rollout lasted that long.")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"dt": dt, "dt_source": dt_source, "n_scenarios": len(rows),
             "n_reached_max_t": len(reached), "reached": reached, "missing": missing,
             "rows": rows}, indent=2))
        print(f"\nwrote {out}")

    return 1 if reached else 0


if __name__ == "__main__":
    raise SystemExit(main())
