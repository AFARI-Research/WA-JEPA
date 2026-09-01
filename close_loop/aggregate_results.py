#!/usr/bin/env python3
"""Turn HUGSIM closed-loop outputs into an auditable score report.

Two modes:

* single split -- ``--dataset kitti360 --results-root <dir>`` writes
  ``RESULTS_kitti360.md`` and ``results_kitti360.json``. One report per split, so
  there is no need to wait for all 436 scenarios.
* combined -- ``--combined`` reads the per-split ``results_<dataset>.json`` files and
  writes ``RESULTS.md`` with an Average row. The HUGSIM paper reports an Average row
  as well, so aggregating across splits is a published convention rather than an
  invention of this script.

Every report contains:

1. overall and per-tier values for all official metrics (NC, DAC, TTC, C, PDMS, RC,
   HD-Score);
2. the published UniAD / VAD / LTF numbers for the same split, per difficulty tier;
3. a credibility audit -- planner failures, warm-up padding, latency, termination
   reasons, and the scenarios that produced no result at all;
4. the per-scenario table;
5. the measurement caveats that apply to that split.

## Why the audit section is mandatory

``hugsim_client.py`` falls back to a braking trajectory when inference raises, and
counts it. A scenario that fell back on every step still scores a respectable
HD-Score, because standing still collides with nothing and stays inside the drivable
area. Reporting only the mean would present that as a good result even though the
model never contributed. Scenarios with ``n_failures > 0`` are therefore listed
separately and excluded from a second, trusted mean. The same applies when
``eval.json`` exists but ``planner_stats.json`` does not: the driving stack never shut
down cleanly, so the score is not trustworthy.

## Metric definitions (sim/utils/score_calculator.py, unmodified)

Per frame, ``pdms = nc * dac * (5 * ttc + 2 * c) / 7``; per scenario,
``hdscore = mean(pdms) * rc`` with ``rc`` clipped to 1.0. The NC/DAC/TTC/C columns are
per-frame means for that scenario. DAC is judged against the scene's ``ground.ply``
point cloud, not against an HD map.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

METRICS = ["nc", "dac", "ttc", "c", "pdms", "rc", "hdscore"]
METRIC_LABEL = {
    "nc": "NC",
    "dac": "DAC",
    "ttc": "TTC",
    "c": "C",
    "pdms": "PDMS",
    "rc": "RC",
    "hdscore": "**HD-Score**",
}
METRIC_DESC = [
    ("NC", "No at-fault Collision", "fraction of frames without an at-fault collision"),
    ("DAC", "Drivable Area Compliance", "fraction of frames inside the drivable area (from ground.ply, not an HD map)"),
    ("TTC", "Time-to-Collision", "fraction of frames meeting the time-to-collision threshold"),
    ("C", "Comfort", "fraction of frames within the acceleration / jerk / lateral-acceleration limits"),
    ("PDMS", "PDM Score", "mean over frames of nc * dac * (5 * ttc + 2 * c) / 7"),
    ("RC", "Route Completion", "(index of the furthest reached waypoint + 1) / (waypoint count * 0.9), clipped to 1.0"),
    ("HD-S", "HD-Score", "the official headline metric: PDMS * RC"),
]

TERM_PATTERNS = [
    ("Collision with background", "hit static geometry"),
    ("Collision with foreground", "hit a dynamic actor"),
    ("Far from preset trajectory", "left the reference route by more than 10 m"),
    ("Complete", "reached the end of the route"),
    # The fifth entry is not printed by the simulator. `closed_loop.py` ends the episode on
    # `cnt > 400` without logging anything, so it has to be inferred from the step count.
    ("StepLimit", "hit the 400-step limit without colliding or leaving the route"),
]
TERM_RE = re.compile("|".join(re.escape(p) for p, _ in TERM_PATTERNS))
TERM_DESC = dict(TERM_PATTERNS)

# Difficulty tiers of the official benchmark. The display order is fixed rather than
# alphabetical, otherwise "extreme" sorts before "hard". The list only orders tiers, it does
# not define the set: the mode suffixes differ between splits, so any tier not listed here is
# appended at the end instead of being dropped.
TIER_ORDER = ["easy_00", "easy_01", "medium_00", "medium_01",
              "hard_00", "hard_01", "extreme_00", "extreme_01"]
TIER_BUCKETS = ["easy", "medium", "hard", "extreme"]

# HD-Score as published in the HUGSIM paper (arXiv:2412.01718v1), Table XIII, Appendix C.3:
# four splits x three baselines x four difficulty tiers, plus the paper's own Average row.
# The paper reports easy/medium/hard/extreme without the _00/_01 distinction, so our tiers are
# folded with tier_bucket() before comparison.
#
# All three baselines are trained on the split they are evaluated on. A zero-shot cross-domain
# model is not measured the same way, so this table is context, not a leaderboard. The values
# also predate the controller heading fix and were measured on an earlier scenario release, so a
# comparison worth reporting re-scores each baseline under the protocol used here rather than
# quoting these columns.
PAPER_BASELINES = {
    "nuscenes": {
        "UniAD": {"easy": 0.589, "medium": 0.378, "hard": 0.365, "extreme": 0.168},
        "VAD": {"easy": 0.348, "medium": 0.132, "hard": 0.204, "extreme": 0.158},
        "LTF": {"easy": 0.616, "medium": 0.307, "hard": 0.226, "extreme": 0.171},
    },
    "kitti360": {
        "UniAD": {"easy": 0.047, "medium": 0.019, "hard": 0.017, "extreme": 0.006},
        "VAD": {"easy": 0.029, "medium": 0.014, "hard": 0.014, "extreme": 0.008},
        "LTF": {"easy": 0.080, "medium": 0.028, "hard": 0.017, "extreme": 0.003},
    },
    "waymo": {
        "UniAD": {"easy": 0.664, "medium": 0.419, "hard": 0.401, "extreme": 0.171},
        "VAD": {"easy": 0.154, "medium": 0.093, "hard": 0.110, "extreme": 0.085},
        "LTF": {"easy": 0.546, "medium": 0.200, "hard": 0.219, "extreme": 0.071},
    },
    "pandaset": {
        "UniAD": {"easy": 0.647, "medium": 0.366, "hard": 0.308, "extreme": 0.226},
        "VAD": {"easy": 0.442, "medium": 0.158, "hard": 0.087, "extreme": 0.079},
        "LTF": {"easy": 0.871, "medium": 0.450, "hard": 0.331, "extreme": 0.080},
    },
    "AVERAGE": {
        "UniAD": {"easy": 0.487, "medium": 0.295, "hard": 0.273, "extreme": 0.143},
        "VAD": {"easy": 0.243, "medium": 0.099, "hard": 0.104, "extreme": 0.082},
        "LTF": {"easy": 0.528, "medium": 0.246, "hard": 0.198, "extreme": 0.081},
    },
}
BASELINE_NAMES = ["UniAD", "VAD", "LTF"]

# Display name and split-specific caveats. The rear-camera note comes from the paper's
# benchmark section: the KITTI-360 and Waymo splits render only the three front cameras. All
# four *_camera.yaml files list six cameras, so nothing raises -- the rear view is simply fed
# in as a black image. A four-camera model therefore loses 25% of its visual input on those
# two splits, silently.
DATASET_INFO = {
    "nuscenes": {
        "name": "nuScenes",
        "cam_note": "All six cameras are rendered; `CAM_BACK` is valid.",
        "caveats": [],
    },
    "kitti360": {
        "name": "KITTI-360",
        "cam_note": "Only the three front cameras are rendered; `CAM_BACK` is a black image.",
        "caveats": [
            "The rear view is empty. KITTI-360 itself only has two forward-facing perspective "
            "cameras, and the paper's benchmark section states that this split renders the front "
            "three cameras only. WA-JEPA always consumes four views, so a quarter of its visual "
            "input carries no information here. This is a property of the benchmark rather than a "
            "defect, but it penalises methods that use the rear view, and not necessarily to the "
            "same degree as it penalises UniAD/VAD/LTF.",
            "All three published baselines score below 0.09 HD-Score on this split (LTF reaches "
            "0.080 on easy and 0.003 on extreme). The split is hard for every method; read the "
            "absolute values against that scale rather than against nuScenes.",
        ],
    },
    "waymo": {
        "name": "Waymo",
        "cam_note": "Only the three front cameras are rendered; `CAM_BACK` is a black image.",
        "caveats": [
            "The rear view is empty, for the same reason as KITTI-360: none of Waymo's five "
            "cameras points straight back, and this split renders the front three only. A quarter "
            "of the model's visual input carries no information.",
        ],
    },
    "pandaset": {
        "name": "PandaSet",
        "cam_note": "All six cameras are rendered; `CAM_BACK` is valid.",
        "caveats": [
            "Fifteen `medium_02` scenarios need the `HUGSIM_COMPAT` patch to run at all, for a "
            "reason upstream of any driving stack. Their `plan_list` passes "
            "`controller_args: {max_t: 25}` to `ConstantPlanner`, whose `__init__(self)` in "
            "`sim/utils/agent_controller.py` takes no arguments, so `gymnasium.make()` raises "
            "`TypeError` before the first frame is rendered. `max_t` appears only in those fifteen "
            "files; the other 421 scenarios use the argument-less form. This is a data package that "
            "is newer than the code. `close_loop/run_upstream_compat.py` accepts the argument and "
            "ignores it, which is a no-op for any rollout that ends before `max_t` -- every reading "
            "of `max_t` agrees with constant velocity until then. Which rollouts those are is a "
            "fact about the run, so `close_loop/check_maxt_reached.py` verifies it afterwards and "
            "flags any scenario that reached `max_t / dt` steps. With `HUGSIM_COMPAT=0` the fifteen "
            "are reported as missing and PandaSet is scored over 112 of its 127 scenarios instead.",
        ],
    },
}


def tier_bucket(tier: str) -> str:
    """Fold medium_00 / medium_01 into the paper's single "medium" tier."""
    return tier.split("_")[0]


def scenario_dir(dataset: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return REPO / "close_loop" / "configs" / "benchmark" / dataset


def yaml_scalar(text: str, key: str) -> str | None:
    """Read one top-level scalar out of a scenario yaml.

    Deliberately not ``yaml.safe_load``: YAML 1.1 reads the underscores in
    ``10320_10520`` as digit separators, so the kitti360 and waymo ``scene_name`` values
    come back as integers. Only the literal is needed here, and a regex cannot
    reinterpret it.
    """
    m = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", text, re.M)
    if not m:
        return None
    return m.group(1).strip().strip("'\"")


def scenario_yamls(scen_dir: Path) -> list[Path]:
    runnable = scen_dir / "runnable.txt"
    if runnable.is_file():
        return [Path(line.strip()) for line in runnable.read_text().splitlines() if line.strip()]
    return sorted(scen_dir.glob("*.yaml"))


def count_hd_map(scen_dir: Path) -> int:
    """Count the scenarios that switch on load_HD_map. The number differs per split."""
    if not scen_dir.is_dir():
        return 0
    n = 0
    for f in sorted(scen_dir.glob("*.yaml")):
        for line in f.read_text().splitlines():
            if line.startswith("load_HD_map:") and "true" in line.lower():
                n += 1
                break
    return n


def log_index(scen_dir: Path, log_dir: Path | None) -> dict[str, Path]:
    """Map output directory name -> log file, going through the scenario yaml.

    The two names are not related by any string transformation, so the yaml has to be
    read to connect them::

        log name    = <yaml file name>.log     scene-100613054308-easy-00.log
        output name = <scene_name>_<mode>      100613054308_0_200_easy_00

    The waymo ``scene_name`` (``100613054308_0_200``) is truncated in the log name, while
    kitti360 keeps its underscores (``scene-10320_10520-easy-00``). Guessing by replacing
    separators happens to work for nuscenes and for nothing else, which shows up as every
    non-nuscenes termination reason being reported as "unknown".
    """
    if log_dir is None or not log_dir.is_dir():
        return {}
    index: dict[str, Path] = {}
    for y in sorted(scen_dir.glob("*.yaml")):
        log = log_dir / (y.stem + ".log")
        if not log.is_file():
            continue
        text = y.read_text()
        scene, mode = yaml_scalar(text, "scene_name"), yaml_scalar(text, "mode")
        if scene is None or mode is None:
            continue
        index[f"{scene}_{mode}"] = log
    return index


def find_missing(scen_dir: Path, out_root: Path, log_dir: Path | None) -> list[dict]:
    """List which scenarios produced no eval.json, with the real error from their log.

    Reporting only a count and pointing at the logs hands the investigation to the reader,
    and missing scenarios are exactly where a problem hides best (an upstream data/code
    mismatch, for instance).
    """
    missing = []
    for y in scenario_yamls(scen_dir):
        if not y.is_file():
            continue
        text = y.read_text()
        scene, mode = yaml_scalar(text, "scene_name"), yaml_scalar(text, "mode")
        if scene is None or mode is None:
            continue
        if (out_root / f"{scene}_{mode}" / "eval.json").is_file():
            continue
        error = ""
        log = (log_dir / (y.stem + ".log")) if log_dir else None
        if log is not None and log.is_file():
            # Take the last real exception line. The tail of the log is usually gymnasium's
            # re-raise, which dumps the full kwargs and hides the original message.
            for line in reversed(log.read_text(errors="replace").splitlines()):
                m = re.match(r"^(\w*(?:Error|Exception)):\s*(.{0,160})", line.strip())
                if m:
                    error = f"{m.group(1)}: {m.group(2)}"
                    break
        missing.append({"yaml": y.name, "dir": f"{scene}_{mode}", "error": error})
    return missing


def split_dir_name(name: str) -> tuple[str, str]:
    """``<scene id>_<tier>_<nn>`` -> ``(scene id, "<tier>_<nn>")``.

    The scene id itself contains underscores on three of the four splits::

        nuscenes  scene-0013_easy_00          -> scene-0013
        kitti360  0000_10320_10520_easy_00    -> 0000_10320_10520
        waymo     100613054308_0_200_easy_00  -> 100613054308_0_200

    Splitting from the left works only for single-segment ids: kitti360 would yield the
    tier ``10320_10520_easy_00``, which then fails to fold into any of the paper's four
    buckets. The tier and its index are always the last two segments, so split from the
    right.
    """
    head, _, index = name.rpartition("_")
    scene, _, tier = head.rpartition("_")
    return scene, f"{tier}_{index}"


def collect(out_root: Path, logs: dict[str, Path]) -> list[dict]:
    rows = []
    for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
        eval_path = d / "eval.json"
        if not eval_path.is_file():
            continue
        ev = json.loads(eval_path.read_text())
        scene, tier = split_dir_name(d.name)
        row = {"dir": d.name, "scene": scene, "tier": tier}
        row.update({m: ev.get(m) for m in METRICS})

        stats_path = d / "planner_stats.json"
        if stats_path.is_file():
            st = json.loads(stats_path.read_text())
            row["steps"] = st.get("n_steps")
            row["fail"] = st.get("n_failures", 0)
            row["pad"] = st.get("n_hist_pad")
            row["lat_p50"] = (st.get("latency_summary") or {}).get("p50_ms")
            row["load_s"] = st.get("model_load_s")
            row["stats_ok"] = True
        else:
            row["fail"] = None
            row["stats_ok"] = False

        row["term"] = None
        log = logs.get(d.name)
        if log is not None:
            m = TERM_RE.search(log.read_text(errors="replace"))
            if m:
                row["term"] = m.group(0)
        if row["term"] is None and isinstance(row.get("steps"), int) and row["steps"] >= 400:
            row["term"] = "StepLimit"
        rows.append(row)
    return rows


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def fmt(value, nd: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{nd}f}"


def agg(rows) -> dict:
    return {m: mean([r[m] for r in rows]) for m in METRICS}


def md_row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def tiers_present(rows) -> list[str]:
    seen = {r["tier"] for r in rows}
    ordered = [t for t in TIER_ORDER if t in seen]
    return ordered + sorted(seen - set(ordered))


def trusted(rows) -> list[dict]:
    """Rows whose score reflects the model: statistics written, no inference failure."""
    return [r for r in rows if r.get("stats_ok") and not r.get("fail")]


def metric_table(rows, label: str) -> list[str]:
    lines = [md_row(["Subset", "N"] + [METRIC_LABEL[m] for m in METRICS]),
             md_row(["---", "---:"] + ["---:"] * len(METRICS))]
    a = agg(rows)
    lines.append(md_row([label, len(rows)] + [fmt(a[m]) for m in METRICS]))
    for tier in tiers_present(rows):
        sub = [r for r in rows if r["tier"] == tier]
        a = agg(sub)
        lines.append(md_row([f"`{tier}`", len(sub)] + [fmt(a[m]) for m in METRICS]))
    return lines


def baseline_table(rows, dataset: str) -> list[str]:
    baselines = PAPER_BASELINES.get(dataset)
    if not baselines:
        return []
    lines = [md_row(["Tier", "N", "WA-JEPA HD-Score"] + BASELINE_NAMES),
             md_row(["---", "---:", "---:"] + ["---:"] * len(BASELINE_NAMES))]
    for bucket in TIER_BUCKETS:
        sub = [r for r in rows if tier_bucket(r["tier"]) == bucket]
        if not sub:
            continue
        ours = mean([r["hdscore"] for r in sub])
        cells = [bucket, len(sub), fmt(ours, 3)]
        cells += [fmt(baselines[b].get(bucket), 3) for b in BASELINE_NAMES]
        lines.append(md_row(cells))
    ours_all = mean([r["hdscore"] for r in rows])
    avg = [mean([baselines[b][t] for t in TIER_BUCKETS]) for b in BASELINE_NAMES]
    lines.append(md_row(["**all**", len(rows), f"**{fmt(ours_all, 3)}**"] + [fmt(v, 3) for v in avg]))
    lines.append("")
    lines.append("The baseline columns are the published values for this split "
                 "(arXiv:2412.01718v1, Table XIII). Each baseline is trained on the split it is "
                 "evaluated on; a zero-shot model is not measured the same way. They also "
                 "predate the controller heading fix, so they are context rather than a "
                 "comparison: a comparison re-scores every baseline under this protocol. The "
                 "**all** row averages our per-scenario scores but averages the baselines over "
                 "their four tier values, because the paper does not publish their scenario "
                 "counts.")
    return lines


def audit_section(rows, missing, scen_dir: Path, n_expected: int) -> list[str]:
    lines = ["## Credibility audit", ""]
    n_done = len(rows)
    tr = trusted(rows)
    no_stats = [r for r in rows if not r.get("stats_ok")]
    failed = [r for r in rows if r.get("fail")]

    lines.append(f"- scenarios with a result: **{n_done} / {n_expected}**")
    lines.append(f"- fully trustworthy (statistics written, zero inference failures): "
                 f"**{len(tr)}**")
    if len(tr) != n_done:
        a_all, a_tr = agg(rows), agg(tr)
        lines.append(f"- HD-Score over all results: **{fmt(a_all['hdscore'])}**; "
                     f"over trustworthy results only: **{fmt(a_tr['hdscore'])}**")
    if failed:
        lines.append(f"- {plural(len(failed), 'scenario')} fell back to a braking plan on at "
                     f"least one step. A scenario that brakes throughout still scores well "
                     f"(standing still hits nothing), so these are excluded from the "
                     f"trustworthy mean:")
        for r in sorted(failed, key=lambda r: -r["fail"])[:20]:
            lines.append(f"  - `{r['dir']}` {r['fail']}/{r.get('steps')} steps failed, "
                         f"hdscore={fmt(r['hdscore'])}")
    if no_stats:
        lines.append(f"- {plural(len(no_stats), 'scenario')} have `eval.json` but no "
                     f"`planner_stats.json`, so the driving stack did not shut down cleanly: "
                     + ", ".join(f"`{r['dir']}`" for r in no_stats[:20]))

    pads = [r.get("pad") for r in rows if isinstance(r.get("pad"), int)]
    if pads:
        lines.append(f"- warm-up padding: {sum(pads)} steps in total, at most "
                     f"{max(pads)} in one scenario. The first frames of an episode have no "
                     f"history, so the oldest available frame is repeated.")
    lat = [r.get("lat_p50") for r in rows if isinstance(r.get("lat_p50"), (int, float))]
    if lat:
        lines.append(f"- planner latency, median of the per-scenario p50: "
                     f"{sorted(lat)[len(lat) // 2]:.0f} ms "
                     f"(min {min(lat):.0f} ms, max {max(lat):.0f} ms)")
    loads = [r.get("load_s") for r in rows if isinstance(r.get("load_s"), (int, float))]
    if loads:
        lines.append(f"- model load time: {min(loads):.0f}-{max(loads):.0f} s")

    lines.append("")
    lines.append("### Termination reasons")
    lines.append("")
    counts = Counter(r["term"] or "unknown" for r in rows)
    lines.append(md_row(["Reason", "N", "Meaning"]))
    lines.append(md_row(["---", "---:", "---"]))
    for reason, n in counts.most_common():
        lines.append(md_row([f"`{reason}`", n, TERM_DESC.get(reason, "no termination message in the log")]))
    if counts.get("unknown"):
        lines.append("")
        lines.append("`unknown` means no termination message was found in that scenario's log. "
                     "Pass `--log-dir` if it is missing for every scenario.")

    if missing:
        lines.append("")
        lines.append(f"### Scenarios without a result ({len(missing)})")
        lines.append("")
        lines.append(md_row(["Scenario", "Error from the log"]))
        lines.append(md_row(["---", "---"]))
        for m in missing:
            lines.append(md_row([f"`{m['yaml']}`", m["error"] or "(no exception line in the log)"]))

    n_hd = count_hd_map(scen_dir)
    if n_hd:
        lines.append("")
        lines.append(f"{plural(n_hd, 'scenario')} in this split set `load_HD_map: true`. They read "
                     f"`base.HD_map` and need `HD_MAP_PATH` / `HD_MAP_VERSION` to be set; "
                     f"without them those scenarios cannot start.")
    return lines


def scenario_table(rows) -> list[str]:
    lines = [md_row(["Scenario", "Tier"] + [METRIC_LABEL[m] for m in METRICS]
                    + ["Steps", "Fail", "Termination"]),
             md_row(["---", "---"] + ["---:"] * len(METRICS) + ["---:", "---:", "---"])]
    for r in sorted(rows, key=lambda r: (r["tier"], r["scene"])):
        lines.append(md_row(
            [f"`{r['scene']}`", r["tier"]]
            + [fmt(r[m], 3) for m in METRICS]
            + [r.get("steps", "n/a"),
               "n/a" if r.get("fail") is None else r["fail"],
               r["term"] or "unknown"]
        ))
    return lines


def build_md(rows, missing, args, n_expected: int, scen_dir: Path) -> str:
    info = DATASET_INFO.get(args.dataset, {"name": args.dataset, "cam_note": "", "caveats": []})
    out = [f"# HUGSIM closed-loop results -- {info['name']}", ""]
    out.append(f"- config: `{args.config or 'configs/wa_jepa_hugsim.yaml'}`")
    out.append(f"- checkpoint: `{args.checkpoint or '(not recorded)'}`")
    out.append(f"- results: `{args.results_root}`")
    out.append(f"- simulator: HUGSIM, unmodified; scenarios rewritten by "
               f"`close_loop/prep_scenarios.py` (agent asset paths only)")
    if info["cam_note"]:
        out.append(f"- cameras: {info['cam_note']}")
    out.append("")

    out.append("## Scores")
    out.append("")
    out += metric_table(rows, "**all**")
    out.append("")
    tr = trusted(rows)
    if len(tr) != len(rows):
        out.append("Trustworthy subset only (statistics written, zero inference failures):")
        out.append("")
        out += metric_table(tr, "**trusted**")
        out.append("")

    bt = baseline_table(rows, args.dataset)
    if bt:
        out.append("## Comparison with the published baselines")
        out.append("")
        out += bt
        out.append("")

    out += audit_section(rows, missing, scen_dir, n_expected)
    out.append("")

    out.append("## Metric definitions")
    out.append("")
    out.append(md_row(["Metric", "Name", "Definition"]))
    out.append(md_row(["---", "---", "---"]))
    for short, name, desc in METRIC_DESC:
        out.append(md_row([short, name, desc]))
    out.append("")
    out.append("Per frame `pdms = nc * dac * (5 * ttc + 2 * c) / 7`; per scenario "
               "`hdscore = mean(pdms) * rc`. Scoring code is "
               "`sim/utils/score_calculator.py`, used unmodified.")

    if info["caveats"]:
        out.append("")
        out.append("## Caveats for this split")
        out.append("")
        for c in info["caveats"]:
            out.append(f"- {c}")

    out.append("")
    return "\n".join(out) + "\n"


def summarize(rows, missing, args, n_expected: int) -> dict:
    tr = trusted(rows)
    return {
        "dataset": args.dataset,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "results_root": str(args.results_root),
        "n_expected": n_expected,
        "n_scored": len(rows),
        "n_trusted": len(tr),
        "n_missing": len(missing),
        "overall": agg(rows),
        "overall_trusted": agg(tr),
        "per_tier": {t: agg([r for r in rows if r["tier"] == t]) for t in tiers_present(rows)},
        "per_bucket": {
            b: agg([r for r in rows if tier_bucket(r["tier"]) == b])
            for b in TIER_BUCKETS
            if any(tier_bucket(r["tier"]) == b for r in rows)
        },
        "terminations": dict(Counter(r["term"] or "unknown" for r in rows)),
        "missing": missing,
        "scenarios": [
            {k: r.get(k) for k in ["dir", "scene", "tier", *METRICS, "steps", "fail", "pad", "term"]}
            for r in sorted(rows, key=lambda r: (r["tier"], r["scene"]))
        ],
    }


def build_combined_md(summaries: dict, args) -> str:
    out = ["# HUGSIM closed-loop results -- all splits", ""]
    order = [d for d in ["nuscenes", "kitti360", "waymo", "pandaset"] if d in summaries]
    ckpts = {s.get("checkpoint") for s in summaries.values()}
    out.append(f"- checkpoint: `{'`, `'.join(sorted(str(c) for c in ckpts))}`")
    if len(ckpts) > 1:
        out.append("- **The splits below were not all produced by the same checkpoint.** "
                   "They cannot be averaged into one number.")
    out.append("")

    out.append("## Per split")
    out.append("")
    out.append(md_row(["Split", "Scored / total", "Trusted"] + [METRIC_LABEL[m] for m in METRICS]))
    out.append(md_row(["---", "---:", "---:"] + ["---:"] * len(METRICS)))
    for ds in order:
        s = summaries[ds]
        out.append(md_row(
            [DATASET_INFO.get(ds, {}).get("name", ds),
             f"{s['n_scored']} / {s['n_expected']}",
             s["n_trusted"]]
            + [fmt(s["overall"].get(m)) for m in METRICS]
        ))
    avg = {m: mean([summaries[ds]["overall"].get(m) for ds in order]) for m in METRICS}
    out.append(md_row(["**Average**", "", ""] + [f"**{fmt(avg[m])}**" for m in METRICS]))
    out.append("")
    out.append("The Average row is the unweighted mean over splits, which is how the paper's "
               "Table XIII reports its own Average row.")
    out.append("")

    out.append("## HD-Score by difficulty, against the published baselines")
    out.append("")
    out.append(md_row(["Split", "Tier", "WA-JEPA"] + BASELINE_NAMES))
    out.append(md_row(["---", "---", "---:"] + ["---:"] * len(BASELINE_NAMES)))
    for ds in order + ["AVERAGE"]:
        baselines = PAPER_BASELINES.get(ds, {})
        for bucket in TIER_BUCKETS:
            if ds == "AVERAGE":
                ours = mean([
                    (summaries[d]["per_bucket"].get(bucket) or {}).get("hdscore") for d in order
                ])
            else:
                ours = (summaries[ds]["per_bucket"].get(bucket) or {}).get("hdscore")
            cells = [DATASET_INFO.get(ds, {}).get("name", ds), bucket, fmt(ours, 3)]
            cells += [fmt(baselines.get(b, {}).get(bucket), 3) for b in BASELINE_NAMES]
            out.append(md_row(cells))
    out.append("")
    out.append("Our AVERAGE rows average the per-split values, matching how the paper builds its "
               "Average row. Every baseline is trained on the split it is evaluated on.")
    out.append("")

    out.append("## Caveats")
    out.append("")
    for ds in order:
        info = DATASET_INFO.get(ds, {})
        for c in info.get("caveats", []):
            out.append(f"- **{info.get('name', ds)}**: {c}")
    missing_total = sum(s["n_missing"] for s in summaries.values())
    if missing_total:
        out.append(f"- {plural(missing_total, 'scenario')} produced no result in total; see the "
                   f"per-split reports for the scenario names and the errors from their logs.")
    out.append("")
    out.append(f"Per-split reports: "
               + ", ".join(f"`RESULTS_{ds}.md`" for ds in order))
    out.append("")
    return "\n".join(out) + "\n"


def run_one(args) -> int:
    out_root = Path(args.results_root)
    if not out_root.is_dir():
        print(f"[ERROR] results root does not exist: {out_root}")
        return 1
    scen_dir = scenario_dir(args.dataset, args.scenario_dir)
    log_dir = Path(args.log_dir) if args.log_dir else None

    logs = log_index(scen_dir, log_dir)
    rows = collect(out_root, logs)
    if not rows:
        print(f"[ERROR] no eval.json found under {out_root}")
        return 1
    missing = find_missing(scen_dir, out_root, log_dir) if scen_dir.is_dir() else []
    n_expected = args.n_expected or (len(scenario_yamls(scen_dir)) if scen_dir.is_dir() else len(rows))

    out_dir = Path(args.out_dir) if args.out_dir else out_root.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = Path(args.md_out) if args.md_out else out_dir / f"RESULTS_{args.dataset}.md"
    json_path = Path(args.json_out) if args.json_out else out_dir / f"results_{args.dataset}.json"

    md_path.write_text(build_md(rows, missing, args, n_expected, scen_dir), encoding="utf-8")
    summary = summarize(rows, missing, args, n_expected)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    a = summary["overall"]
    print(f"{args.dataset}: {len(rows)}/{n_expected} scored, {summary['n_trusted']} trusted, "
          f"HD-Score {fmt(a['hdscore'])} (PDMS {fmt(a['pdms'])}, RC {fmt(a['rc'])})")
    print(f"  {md_path}")
    print(f"  {json_path}")
    return 0


def run_combined(args) -> int:
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "outputs" / "hugsim"
    summaries = {}
    for ds in ["nuscenes", "kitti360", "waymo", "pandaset"]:
        f = out_dir / f"results_{ds}.json"
        if f.is_file():
            summaries[ds] = json.loads(f.read_text())
    if not summaries:
        print(f"[ERROR] no results_<dataset>.json under {out_dir}. Run the single-split mode "
              f"for each split first.")
        return 1
    md_path = Path(args.md_out) if args.md_out else out_dir / "RESULTS.md"
    md_path.write_text(build_combined_md(summaries, args), encoding="utf-8")
    missing = [ds for ds in ["nuscenes", "kitti360", "waymo", "pandaset"] if ds not in summaries]
    print(f"combined {len(summaries)} splits -> {md_path}")
    if missing:
        print(f"  not included (no results json): {', '.join(missing)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="nuscenes",
                    choices=["nuscenes", "kitti360", "waymo", "pandaset"])
    ap.add_argument("--results-root", default=None,
                    help="Directory holding <scene>_<mode>/eval.json, i.e. the path printed "
                         "by run_hugsim_benchmark.sh")
    ap.add_argument("--log-dir", default=None,
                    help="Per-scenario logs; without them termination reasons stay unknown")
    ap.add_argument("--scenario-dir", default=None,
                    help="Prepared scenarios (default close_loop/configs/benchmark/<dataset>)")
    ap.add_argument("--combined", action="store_true",
                    help="Aggregate the existing results_<dataset>.json files into RESULTS.md")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write the report (default: parent of --results-root)")
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--n-expected", type=int, default=0,
                    help="0 = take the scenario count from runnable.txt")
    ap.add_argument("--config", default=None, help="Recorded in the report for provenance")
    ap.add_argument("--checkpoint", default=None, help="Recorded in the report for provenance")
    args = ap.parse_args()

    if args.combined:
        return run_combined(args)
    if not args.results_root:
        ap.error("--results-root is required unless --combined is given")
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
