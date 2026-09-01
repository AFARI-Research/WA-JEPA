# HUGSIM closed-loop inference and evaluation

**English** | [中文](hugsim_closed_loop.zh-CN.md)

HUGSIM is a photorealistic closed-loop benchmark built on 3D Gaussian Splatting. Unlike
NAVSIM, which scores a trajectory against a recorded log, HUGSIM feeds the model's plan
back into the simulator: the ego vehicle moves, the cameras render observations from the
updated pose, and errors accumulate over time. This closed-loop process provides a more
challenging evaluation of sequential driving behavior.

The official benchmark is 436 scenarios across four splits (nuScenes 88, KITTI-360 113,
Waymo 108, PandaSet 127), each at difficulty tiers easy / medium / hard / extreme.

HUGSIM's own files are never edited: every correction below is a patch applied in the
evaluation process, before `closed_loop` is imported, so a simulator checkout stays
byte-identical to upstream and every correction is a script in `close_loop/` here. There
are two of them — [the controller heading fix](#controller-heading-fix), which every run
applies, and [the upstream compatibility patches](#upstream-compatibility-patches), without
which 16 of the 436 scenarios cannot be scored at all. The published UniAD / VAD / LTF
baselines were produced without either, so they have to be re-scored under this protocol
rather than quoted; see [Comparing with published baselines](#comparing-with-published-baselines).

## How the integration works

Upstream already supports an external driving stack. `sim/utils/launch_ad.py` spawns a
shell script as a subprocess and exchanges data over two FIFOs:

```text
closed_loop.py  --(pickle (obs, info))-->  <output>/obs_pipe   -->  hugsim_client.py
closed_loop.py  <--(pickle plan [8,2])---  <output>/plan_pipe   <--  hugsim_client.py
```

Only the client-side integration is implemented in this repository. It consists of three
files:

| File | Role |
| --- | --- |
| `close_loop/hugsim_planner.py` | loads a checkpoint, keeps the history buffer, converts observations to model inputs and waypoints back to the simulator's frame |
| `close_loop/hugsim_client.py` | the subprocess: opens the FIFOs, loops until the simulator sends `'Done'`, writes per-scenario statistics |
| `scripts/evaluation/run_hugsim_ad.sh` | what the simulator actually launches; fixes up the environment and runs a preflight before `exec`ing the client |

The simulator only accepts `--ad uniad|vad|ltf`, so the `ltf` slot is used for WA-JEPA.
This selection affects only which `base.*_path` key the simulator reads, and the
generated base config points all three at the same launcher.

Around those three sit the offline tools, none of which touch a GPU or the simulator:
`prep_scenarios.py` rewrites the official scenario files, `selfcheck.py` asserts the
conversions, `aggregate_results.py` turns the per-scenario `eval.json` files into a report,
and `make_video.py` / `render_videos.py` render the archive a `SAVE_VIZ=1` run leaves behind.

### The six conversions

Errors in any of these conversions do not raise exceptions. An incorrect camera slot or an
inverted command can still produce scores for all 436 scenarios, but the resulting
degradation is indistinguishable from lower model performance. The conversions were
validated directly against the simulator rather than inferred from documentation.

1. **Camera slots.** The model indexes a learned camera embedding by position in
   `model.camera_names`, so the mapping is ordered:

   | model | HUGSIM |
   | --- | --- |
   | `cam_l0` | `CAM_FRONT_LEFT` |
   | `cam_f0` | `CAM_FRONT` |
   | `cam_r0` | `CAM_FRONT_RIGHT` |
   | `cam_b0` | `CAM_BACK` |

2. **Sample rate.** HUGSIM steps at 0.25 s (4 Hz); the model's history frames and waypoints
   are 0.5 s apart (2 Hz). History is therefore strided by 2 simulator frames. Feeding every
   frame instead makes the observed motion appear to run at half speed.

3. **World to ego.** `(x, y, yaw) = (ego_box[0], ego_box[1], ego_box[6])` is right-handed
   with x forward and y left, matching the training convention.

4. **Command encoding.** HUGSIM uses 0 = right, 1 = left, 2 = straight; NAVSIM uses
   0 = left, 1 = straight, 2 = right. The mapping is `[2, 0, 1]`, indexed by the HUGSIM
   value. The identity mapping would turn every right turn into a left turn.

5. **Output frame.** `closed_loop.py` reads the plan as lidar-frame `(x right, y forward)`
   and applies `plan[:, [1, 0]]`, so the client emits `(-y, x)` from the ego-frame
   prediction.

6. **Precision.** fp32 weights under `torch.autocast(bfloat16)`, matching the execution
   path used to produce the reported NAVSIM results. Casting the weights themselves to
   bfloat16 follows a different path and does not reproduce those results.

`close_loop/selfcheck.py` asserts all of them offline, and `tests/test_hugsim_planner.py`
locks them in the test suite.

## Prerequisites

### 1. Two environments

The simulator needs `gsplat`, `nerfstudio`, `open3d` and CUDA extensions; WA-JEPA needs
`torch` plus this repository. They can share one environment, but separate environments
simplify dependency management. The client is launched as its own process, so `PYTHON_BIN` only needs to point
at an interpreter that can `import torch` and this repo. The client imports `models/`,
`training/checkpoint.py` and `utils/`, none of which touch nuPlan or NAVSIM.

Install HUGSIM by following its own README, then verify the extensions actually run rather
than merely importing:

```bash
python -c "
import torch, gsplat
from simple_knn._C import distCUDA2
print(gsplat.__version__, distCUDA2(torch.rand(100, 3, device='cuda')).shape)
"
```

### 2. The data package

The scenes and agent assets come from the HUGSIM release:

```text
$HUGSIM_DATA_ROOT/
├── scenes/{nuscenes,kitti360,waymo,pandaset}/...   # 3DGS scenes, ground.ply per scene
├── 3DRealCar/<id>/{gs.pth,wlh.json}                # inserted vehicle assets
└── {nuscenes,kitti360,waymo,pandaset}/*.yaml       # 436 official scenarios
```

### 3. Prepare the scenarios

The shipped scenario files and the simulator code disagree about the agent asset path:
field 5 of each `plan_list` entry ends in `/postprocess/shadow.pth`, while `sim/utils/plan.py`
appends `gs.pth` to it and looks for `.../shadow.pth/gs.pth`. Upstream's own export script
strips the same suffix, so this is a mismatch in the release, not a usage error.

```bash
python close_loop/prep_scenarios.py --data-root "$HUGSIM_DATA_ROOT"
```

This writes `close_loop/configs/benchmark/<dataset>/*.yaml` plus a `runnable.txt` listing
the scenarios whose assets are all present. It changes only the agent asset paths:
`start_*`, `mode`, the `plan_list` structure and `load_HD_map` are untouched — and it
verifies that the rewritten ids still parse as strings. That check matters: 3DRealCar ids
look like `2024_07_05_16_16_38`, and YAML 1.1 reads underscores as digit separators, so an
unquoted id becomes the integer `20240705161638` and the asset goes missing much later with
no YAML error.

## Running the benchmark

```bash
export HUGSIM_ROOT=/path/to/HUGSIM
export HUGSIM_DATA_ROOT=/path/to/hugsim_data
export SIM_PYTHON=/path/to/hugsim-env/bin/python
export PYTHON_BIN=/path/to/wa-jepa-env/bin/python
export CHECKPOINT=/path/to/wa-jepa/checkpoint
export OUTPUT_DIR=outputs/hugsim

# One scenario first. It takes 2-20 minutes and exercises the whole chain.
# The scenario argument is the simulator's scenario name, which uses hyphens.
bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes scene-0013-easy-00

# Then the split.
bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes
```

Add `--dry-run` to print the resolved execution plan without running the benchmark.

The default configuration is 8 GPUs with 4 scenarios each, giving 32 concurrent scenarios.
A scenario needs about 8.5 GB (3DGS rendering ~3.5 GB plus the planner ~4.9 GB), so four per
card uses roughly 25 GB — the limit is compute, not memory. Override with
`HUGSIM_NUM_GPUS`, `HUGSIM_PER_GPU` or `JOBS`.

Scenarios are dispatched through `xargs -P` as a work pool rather than sharded up front,
because runtimes differ by more than a factor of ten: an early collision ends in about 80
seconds while a full rollout takes ~19 minutes. Static sharding would leave many GPUs idle
while waiting for the longest-running shard.

**One process per scenario is mandatory.** `sim/utils/plan.py` builds
`self.stats[f'agent_{i}']` with `enumerate(plan_list)` and never clears the previous
scenario's keys, so a second `gymnasium.make` in the same process raises
`KeyError: 'agent_N'` where N is the previous scenario's agent count. The runner already
does this; do not batch scenarios in a Python loop.

Expect roughly 3-5 hours for all 436 scenarios at the default concurrency.

### The remaining splits

```bash
for ds in nuscenes kitti360 waymo pandaset; do
  bash scripts/evaluation/run_hugsim_benchmark.sh "$ds"
done
```

Scenarios with `load_HD_map: true` additionally need `HD_MAP_PATH` and `HD_MAP_VERSION`
pointing at the raw dataset; without them those scenarios cannot start. The runner warns
when a split contains them and the variables are unset.

### Resuming

`SKIP_DONE=1` (the default) skips a scenario only when **both** `eval.json` and
`planner_stats.json` exist. `eval.json` alone means the simulator scored the run but the
driving stack never shut down cleanly, which is exactly the result that must be re-run
rather than kept.

When resuming with a different checkpoint, set `WM_RUN_TAG` to a substring that must appear
in the output path. The preflight refuses to run if it does not, which prevents two
checkpoints from silently merging into one results table. For checkpoint directories that
training overwrites in place (`last`), also set `WM_EXPECT_STEP`: the file size stays
identical across epochs, so only `state.pt` identifies which weights are present.

## Aggregating results

```bash
python close_loop/aggregate_results.py --dataset nuscenes \
  --results-root  outputs/hugsim/nuscenes_ltf \
  --log-dir       outputs/hugsim/nuscenes_logs \
  --checkpoint    "$CHECKPOINT"
```

That writes `RESULTS_nuscenes.md` and `results_nuscenes.json`. Once all four splits are
done:

```bash
python close_loop/aggregate_results.py --combined --out-dir outputs/hugsim
```

`--log-dir` is optional but recommended because termination reasons are available only in
the logs. Without it, every scenario is reported as `unknown`.

### Metrics

Scoring is `sim/utils/score_calculator.py`, unmodified. Per frame:

```text
pdms = nc * dac * (5 * ttc + 2 * c) / 7
```

and per scenario `hdscore = mean(pdms) * rc`, with `rc` clipped to 1.0. HD-Score is the
headline metric.

| Metric | Meaning |
| --- | --- |
| NC | fraction of frames without an at-fault collision |
| DAC | fraction of frames inside the drivable area, judged from the scene's `ground.ply` point cloud rather than an HD map |
| TTC | fraction of frames meeting the time-to-collision threshold |
| C | fraction of frames within the acceleration / jerk / lateral-acceleration limits |
| PDMS | mean of the per-frame formula above |
| RC | route completion, `(furthest waypoint reached + 1) / (waypoint count * 0.9)`, clipped to 1.0 |
| HD-Score | **PDMS × RC** |

### Why the report has an audit section

`hugsim_client.py` falls back to a braking trajectory when inference raises, and counts it.
A scenario that falls back at every step can still obtain a nontrivial HD-Score because a
stationary vehicle avoids collisions and remains inside the drivable area. Reporting only
the overall mean could therefore overstate model performance when inference did not
contribute to the rollout.

The aggregator therefore lists every scenario with `n_failures > 0` and reports a second
mean with them excluded. Same for scenarios that have `eval.json` but no
`planner_stats.json`. Scenarios that produced no result at all are listed by name with the
actual exception from their log, rather than as a count.

Termination reasons come from the log, with one exception: `closed_loop.py` ends an episode
at `cnt > 400` without printing anything, so that case is inferred from the step count and
reported as `StepLimit`.

### Known split-specific caveats

The aggregator writes these into every report; they matter when reading absolute numbers.

- **KITTI-360 and Waymo render only the three front cameras.** Both `*_camera.yaml` files
  still list six cameras, so nothing raises — `CAM_BACK` simply comes back black. A
  four-camera model loses a quarter of its visual input on those splits. This is a property
  of the benchmark, not a defect, but it penalises rear-view methods and not necessarily to
  the same degree as it penalises the baselines.
- **KITTI-360 is challenging for all evaluated methods.** All three published baselines score below 0.09
  HD-Score there. Read absolute values against that scale, not against nuScenes.
- **Fifteen PandaSet `medium_02` scenarios need the compatibility patch to run.** Their
  `plan_list` passes `controller_args: {max_t: 25}` to `ConstantPlanner`, whose
  `__init__(self)` takes no arguments, so `gymnasium.make()` raises `TypeError` before the
  first frame renders. The data package is newer than the code. `HUGSIM_COMPAT=1` (the
  default) accepts and ignores the argument, which is a no-op for every rollout that ends
  before `max_t`; `close_loop/check_maxt_reached.py` verifies that afterwards. With
  `HUGSIM_COMPAT=0` those fifteen are reported as missing and PandaSet is scored over 112 of
  127. See [Upstream compatibility patches](#upstream-compatibility-patches).

### Comparing with published baselines

The report includes UniAD / VAD / LTF HD-Score per tier from the HUGSIM paper
(arXiv:2412.01718v1, Table XIII). Those values were produced under the uncorrected
controller and on an earlier scenario release, and all three baselines are **trained on the
split they are evaluated on**, while a zero-shot cross-domain model is not measured the same
way. Treat the columns as context, not as a leaderboard.

A comparison worth reporting re-runs each baseline under the protocol used here: the same
scenarios, the same heading-corrected controller, the same ground-truth driving commands and
the same aggregation, with one implementation of the metrics for every method. The corrected
evaluation code released alongside DrivoR provides that, and re-scoring under it is what our
paper reports.

The aggregator computes a uniform mean over scenarios within each split, and `--combined`
averages the four splits uniformly. Publications on this benchmark also weight the four
difficulty tiers by their scenario counts (80 / 157 / 96 / 103). The two methods differ in
the third decimal place of HD-Score, so each reported result should identify its
aggregation method.

## Verification and debugging

### Offline checks

```bash
python close_loop/selfcheck.py                    # config plus the pure conversions
python -m unittest tests.test_hugsim_planner      # the same invariants, in the suite
```

Both run on CPU without a checkpoint. They cover the conversions above plus the wiring
between the three shell layers — for instance that the base config's `output_dir` and the
directory the runner scans agree, since a mismatch there ends the run with an empty results
directory and exit code 0.

### Checking real frames

```bash
# The run argument is the scenario name (hyphens); the dump lands in the output
# directory, which is named with underscores.
HUGSIM_DUMP_OBS=5 bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes scene-0013-easy-00
python close_loop/selfcheck.py --dump outputs/hugsim/nuscenes_ltf/scene-0013_easy_00/dumped_obs.pkl
```

The client writes the first five real `(obs, info)` frames. This is what catches a black
camera, a command channel that never changes, and a render smaller than the model input —
none of which are visible on synthetic input.

### Per-step archive

`SAVE_VIZ=1` makes the client store the model's actual inputs and plans at each step. This
archive supports analysis of whether a low score originates from the observations or the
predicted plan. For each scenario, it writes the `CAM_FRONT` render and a
mosaic of the four views the model reads at every step, plus `plan_log.pkl` at the end with
the plan, the ego and object boxes, the navigation command, the speed and the timestamp of
each step. Roughly 40 MB per scenario, no effect on the plan.

The option must be enabled before the run. The archive cannot be reconstructed from
`eval.json`; retrospective analysis therefore requires another simulation, which may not
reproduce the original rollout because closed-loop execution is not deterministic across
restarts.

`viz/camera.json` is written once per scenario and holds the intrinsics and `v2c` of every
camera as the simulator reported them, plus the path of the camera config in use. It exists
because `make_video.py` has to project the plan with the camera the run actually rendered
with, and the four splits do not share one — see below.

### Videos

```bash
python close_loop/make_video.py \
  --root    outputs/hugsim/nuscenes_ltf \
  --out-dir outputs/hugsim/videos/nuscenes \
  --pick    outputs/hugsim/results_nuscenes.json \
  --dataset nuscenes --hugsim-root "$HUGSIM_ROOT"

python close_loop/render_videos.py --workers 8 \
  --hugsim-root "$HUGSIM_ROOT" --skip-existing
```

`make_video.py` renders the scenarios of one results directory; `render_videos.py` drives it
across the splits in parallel and writes an `index.md` grouped by split and tier with each
scenario's scores. Both tools are CPU-only and access only the archive. They can therefore
run on any machine that can access the output directory, without a GPU, checkpoint, or
simulator.

Each frame is the `CAM_FRONT` render with the planned trajectory projected onto the road
surface, a bird's-eye panel with the plan, the other agents and the path driven so far, and
a status bar with step, time, speed, navigation command and the scenario's scores. Two red
flags appear where they apply: `FALLBACK: inference failed` on the steps listed in
`planner_stats.json`, and `history padded` on the warm-up steps before the history buffer is
full. These indicators distinguish model-generated behavior from fallback braking that may
otherwise receive a nontrivial score.

Projection chain, derived from HUGSIM's `sim/utils/sim_utils.py` and `hugsim_env`:

```text
p_v = (-y_ego, 1.5, x_ego)                    vehicle frame: x right, y down, z forward
p_c = inv(cam_rect) @ v2c @ inv(v2front) @ p_v
u   = fx * x_c / z_c + cx,  fy from fovy and H, not from fx
```

`whl = [1.6, 1.5, 3.0]` puts the ground plane at `v_y = 1.5`, which is what makes the
overlay lie on the road rather than float. For `CAM_FRONT` the `v2c` terms cancel and only
`cam_rect` is left; keeping the general form means the same code can project the other
cameras.

The splits differ in exactly the places that produce a plausible-looking wrong overlay:

| Split | fovx | fx, fy | `cam_rect` |
| --- | --- | --- | --- |
| nuScenes | 65.12° | 626.41, 626.41 | translate y 0.3 |
| KITTI-360 | 65.12° | 626.41, 626.41 | rotate x 5°, no translation |
| Waymo | 58.12° | 719.82, 771.55 | translate y 0.3 |
| PandaSet | 58.12° | 719.82, 771.55 | translate y 0.7 |

So `fy` must come from `fovy`, not from `fx` — on Waymo and PandaSet they differ by 7 per
cent — KITTI-360's 5 degree pitch moves the ground by tens of pixels, and PandaSet's camera
sits 0.4 m lower than the others. The renderer reads the config rather than embedding any of
this, and cross-checks it against `viz/camera.json`. That check has a known limit worth
stating: `CAM_FRONT`'s `v2c` is byte-identical across all four splits, so intrinsics alone
only separate the 65.12° pair from the 58.12° pair. The `_dataset` field the client records
closes the gap, and a contradiction between it and `--dataset` stops the render.

The camera YAML stores fields of view in degrees, while `load_camera_cfg` converts them in
place — so `info['cam_params']`, and therefore `camera.json`, holds radians. Both are read,
disambiguated by whether the value exceeds pi.

`tests/test_hugsim_video.py` pins the parts of this that fail silently: the Euler convention
(`from_euler('XYZ')` is intrinsic, `Rx @ Ry @ Rz`), the four splits' focal lengths, that left
of the vehicle is left in the image, that points behind the camera are dropped, and that a
`camera.json` naming another split aborts.

### Where to look when a scenario fails

| Symptom | Where to look |
| --- | --- |
| no `eval.json` | `<LOG_DIR>/<scenario>.log`, the simulator's own stdout |
| `eval.json` but no `planner_stats.json` | the client died; its traceback is in the same log |
| `n_failures > 0` in `planner_stats.json` | the `failures` list records the step and the exception |
| no output files, exit code 0 | `base.output_dir` and the scanned directory disagree; compare the generated `*_base.yaml` against `OUT_ROOT` |

Do not pipe a run through `tail`: `cmd | tail -n 50` reports the exit code from `tail`, so
a failed scenario may appear successful. Determine success from the presence of
`eval.json`, not from the process exit code, because `closed_loop.py` catches exceptions
around `create_gym_env` and still exits with code 0.

## Controller heading fix

Upstream's `traj2control` computes the target heading as
`arctan2(d_forward, d_right)` while its iLQR state expects
`arctan2(d_right, d_forward)`. A straight trajectory is therefore handed a target heading of
90 degrees, and a nearly straight one is worse: lateral values jittering around zero swing
the target heading across most of its range every step, saturating the steering rate.

The penalty is not a constant offset that cancels across methods. It scales with the lateral
jitter of the tracked plan, so rankings obtained with the uncorrected controller reflect
both driving performance and trajectory jitter. The issue is corrected on every run.
`run_hugsim_benchmark.sh` always uses `close_loop/run_fixed_controller.py` as its entry point,
and no setting restores the defect.

The published UniAD / VAD / LTF numbers were produced with the defect in place, so a
comparison against them inherits it — a match against figures that carry a controller defect,
not a like-for-like measurement of planning. A comparison worth reporting re-scores each
baseline under the corrected controller; see
[Comparing with published baselines](#comparing-with-published-baselines).

`close_loop/run_fixed_controller.py` monkey-patches the function before `closed_loop` is
imported, so HUGSIM's files stay byte-identical: the fix lives in this repository, as
`close_loop/run_fixed_controller.py`, instead of modifying the simulator checkout. The same
defect is the subject of upstream PR #57, which is still open; that PR
changes the unpacking order rather than the argument order, and the two implementations agree
elementwise.

## Upstream compatibility patches

Two HUGSIM defects abort a scenario before it can be scored, regardless of the driving
stack. Both result from version differences between the data package and the code. Together
they prevent 16 of the 436 scenarios from being scored, and both are patched by
`close_loop/run_upstream_compat.py`, which
`run_hugsim_benchmark.sh` selects unless `HUGSIM_COMPAT=0`.

The entry patches, then delegates: `HUGSIM_COMPAT_INNER` is the entry that would have run
otherwise, normally `run_fixed_controller.py`, so the
two concerns compose without either file duplicating the other's patches. All patches
are installed before `closed_loop` is imported and live in one process; HUGSIM's files stay
byte-identical.

### Defect 1 — `ConstantPlanner` is handed a `max_t` it never implemented

```text
TypeError: ConstantPlanner.__init__() got an unexpected keyword argument 'max_t'
```

`sim/utils/plan.py:55` instantiates the controller as
`getattr(agent_controller, model.controller)(**model.controller_args)`, and upstream's
`ConstantPlanner.__init__(self)` takes nothing. The patch accepts the argument and ignores
it, which gives those fifteen scenarios the same constant-velocity background traffic as the
other 421.

The intended semantics do not affect behavior before the limit. Plausible interpretations
of `max_t`, such as stopping, despawning, or ending the plan at that time, all match
constant-velocity motion for `t < max_t` and can diverge only afterwards. Because `dt` is
0.25 s (`configs/sim/kinematic.yaml`), `max_t: 25` corresponds to step 100, and a rollout
that ends before step 100 is identical under each interpretation.

That premise is a fact about the run rather than about the code, so it is checked afterwards:

```bash
python close_loop/check_maxt_reached.py \
  --results-root outputs/hugsim/pandaset_ltf --dataset pandaset \
  --json-out outputs/hugsim/maxt_check_pandaset.json
```

The script reads `max_t` from each scenario yaml, `dt` from the kinematic config, and
`n_steps` from `planner_stats.json`. It exits non-zero if any rollout reached
`max_t / dt` steps. Such a score depends on unspecified semantics and must therefore be
reported separately rather than included in the mean.

Run the check after every PandaSet run rather than treating it as settled once: how far a
rollout gets depends on the planner, so the answer changes with the checkpoint. On the
reported one, seven of the fifteen reached the limit and the longest ran 152 steps. That is
the expected direction — a vehicle that holds its lane instead of drifting off the road
survives longer, so the caveat grows as driving improves. Two earlier runs of other driving
stacks had no scenario past the limit but one that stopped at 97 of the 100 steps, so the
remaining margin should not be assumed to be sufficient.

The patch replaces `__init__` on the existing class rather than substituting a subclass,
because `sim/utils/plan.py:111` dispatches on `type(controller) is ConstantPlanner`; a
subclass falls through to `raise NotImplementedError`, turning a crash at setup into a crash
mid-rollout.

### Defect 2 — IDM lead distance with one waypoint left

```text
  File "sim/utils/plan.py", line 104, in plan_traj
  File "sim/utils/agent_controller.py", line 170, in update
IndexError: index is out of bounds for dimension with size 0
```

Inside `IDM.update`, the arc length ahead is built from pairwise differences and then padded
back to full length:

```python
lookahead_path = path[nearest_index + 1:][:self.lookahead_path_length]      # n rows
lookahead_lengths = torch.cumsum(torch.norm(
    lookahead_path[1:, :2] - lookahead_path[:-1, :2], dim=1), dim=0)        # n - 1
lookahead_lengths = torch.cat([lookahead_lengths, lookahead_lengths[-1:]])  # back to n
lead_distance = lookahead_lengths[indices_lead]
```

At `n == 1` the difference is empty, so both the `cumsum` and the slice used to pad it are
empty and `lookahead_lengths` stays empty, while `indices_lead` still holds index 0. It takes
a background vehicle with one waypoint left plus a neighbour inside
`lead_distance_threshold` at the same step, so it is not periodic and varies across
planners. On the reported checkpoint, it prevented one nuScenes scenario from being
scored. The entire rollout terminates, so the missing result is the ego vehicle's score.

The patch appends a copy of the final waypoint to `path` in exactly that state. The arc
length ahead becomes `[0.0, 0.0]` and the lead distance 0: the lead vehicle is on the only
waypoint left. This matches the padding performed by `cat` for every longer path. The
change is limited to this case because:

- it fires only when the lookahead slice has exactly one row, so every other call is the
  upstream call;
- `nearest_index = argmin(dists)` does not move — the appended row duplicates the last one
  and `argmin` returns the first minimum, and the state where the last row *is* the minimum
  gives a slice of length 0, which the patch leaves alone;
- the `target` / `is_end` scan gains one index whose coordinates equal the previous last
  row's, so it either breaks at the same earlier index or falls through to the same
  `path[-1]`;
- the duplicated waypoint can duplicate one column of `is_lead`, and `acceleration.min()`
  absorbs a repeated identical value.

With `lookahead_path_length == 1` the slice would stay one row long however long `path`
becomes, so the patch reports that instead of pretending to have fixed it. Upstream's default
is 100 and no scenario overrides it.

### Reproducing a pre-patch run

`HUGSIM_COMPAT=0` restores the unpatched entry. A run made that way scores PandaSet over 112
of its 127 scenarios; the aggregator's PandaSet caveat records which denominator was used, so
the two are not silently comparable.

## Output layout

```text
outputs/hugsim/
├── nuscenes_base.yaml                    # generated per run, not checked in
├── nuscenes_logs/                        # one log per scenario
├── nuscenes_ltf/                         # results; "ltf" is appended by closed_loop.py
│   └── <scene>_<tier>_<nn>/
│       ├── eval.json                     # the official metrics
│       ├── planner_stats.json            # steps, failures, warm-up padding, latency
│       ├── plan_log.pkl                  # per-step plans
│       ├── dumped_obs.pkl                # only with HUGSIM_DUMP_OBS
│       └── viz/                          # only with SAVE_VIZ=1
│           ├── camera.json               # the camera this run rendered with
│           ├── front/0000.jpg            # CAM_FRONT per step
│           └── input/0000.jpg            # the four views the model reads
├── videos/                               # close_loop/render_videos.py
│   ├── index.md                          # per split and tier, with scores
│   └── nuscenes/easy/<scene>_easy_00.mp4
├── RESULTS_nuscenes.md
├── results_nuscenes.json
└── RESULTS.md                            # --combined
```

## Environment variables

`scripts/evaluation/run_hugsim_benchmark.sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HUGSIM_ROOT` | required | HUGSIM checkout |
| `HUGSIM_DATA_ROOT` | required | data package |
| `SIM_PYTHON` | required | simulator environment python |
| `PYTHON_BIN` | required | WA-JEPA environment python |
| `CHECKPOINT` | required | checkpoint directory or `.pt` |
| `CONFIG` | `configs/wa_jepa_hugsim.yaml` | inference config |
| `OUTPUT_DIR` | `outputs/hugsim` | results root |
| `SCENARIO_DIR` | `close_loop/configs/benchmark/<dataset>` | prepared scenarios |
| `HUGSIM_NUM_GPUS` | detected, else 8 | GPU count |
| `HUGSIM_PER_GPU` | 4 | scenarios per GPU |
| `JOBS` | GPUs × per-GPU | total concurrency |
| `HUGSIM_COMPAT` | 1 | apply the upstream compatibility patches; 0 reproduces a pre-patch run |
| `SKIP_DONE` | 1 | skip finished scenarios |
| `SAVE_VIZ` | 0 | archive per-step inputs and plans |
| `CAMERA_YAML` | `$HUGSIM_ROOT/configs/sim/<dataset>_camera.yaml` | camera config; recorded in `viz/camera.json` |
| `HD_MAP_PATH`, `HD_MAP_VERSION` | unset | only for `load_HD_map: true` scenarios |
| `HUGSIM_ENV_SETUP`, `WM_ENV_SETUP` | unset | scripts to source per environment |
| `WM_RUN_TAG` | unset | substring asserted to be in the output path |
| `WM_EXPECT_STEP` | unset | asserted against `state.pt` |
| `DRY_RUN` | 0 | print the plan and exit |

There is deliberately no default `CHECKPOINT`. The value must pass through three process
layers. If any layer drops it, a `${CHECKPOINT:-<path>}` fallback may score an unintended
checkpoint without raising an error, while still producing results for all 436 scenarios.
