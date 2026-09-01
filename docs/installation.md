<p align="center">
  <img src="docs/wa_jepa_logo.png" alt="WA-JEPA" width="760">
</p>

<p align="center">
  <b>English</b> | <a href="README.zh-CN.md">中文</a>
</p>

WA-JEPA is a multi-camera world-action model for autonomous driving. It uses a
V-JEPA 2.1 visual encoder and a joint flow predictor to generate future scene
representations and ego trajectories from historical camera observations.

This branch covers training, open-loop evaluation on NAVSIM, and closed-loop
evaluation on HUGSIM.

## Model

- Visual encoder: V-JEPA 2.1 ViT-L/16
- Cameras: `CAM_L0`, `CAM_F0`, `CAM_R0`, `CAM_B0`
- Input: 4 historical frames at `256 x 512`
- Output: 8 future trajectory points at 2 Hz
- Joint flow predictor: 12 layers, hidden size 512, 8 attention heads
- Parameters: about 480.9M trainable and 787.7M including the frozen EMA encoder

Main implementation:

```text
models/multiview_causal_future_jepa.py
eval/navsim_agent.py             # NAVSIM open-loop agent
close_loop/hugsim_planner.py     # HUGSIM closed-loop adapter
```

## Installation

Python 3.10 and a CUDA-enabled PyTorch environment are recommended.

```bash
conda create -n wa-jepa python=3.10 -y
conda activate wa-jepa

# Install a PyTorch build compatible with your CUDA version.
# NAVSIM pins its own torch version and is installed after this, so verify the
# result once both are in place (see below).
pip install -r requirements.txt
pip install -e .
```

NuPlan and NAVSIM must be installed separately. WA-JEPA supports two different
NAVSIM evaluation environments; install them in separate directories:

```bash
pip install -e /path/to/nuplan-devkit

# NAVSIM v1.1: used only for PDMS
git clone --branch v1.1 https://github.com/autonomousvision/navsim.git navsim-v1.1
pip install -e navsim-v1.1

# NAVSIM v2.2: used only for navtest EPDMS
git clone --branch v2.2 https://github.com/autonomousvision/navsim.git navsim-v2.2
pip install -e navsim-v2.2
```

NAVSIM pins `torch==2.0.1` and `numpy==1.23.4`, so installing it may replace the
existing PyTorch installation. Verify the resulting environment and reinstall a
compatible build if necessary:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Both NAVSIM versions declare the distribution name `navsim`, so the second
`pip install -e` shadows the first: `import navsim` resolves to whichever was
installed last. The evaluation scripts do not rely on that — each one prepends
its own devkit to `PYTHONPATH` — but an ad-hoc `python -c "import navsim"` will
not tell you which version a script is about to use.

Verify the checkouts before evaluation:

```bash
git -C navsim-v1.1 describe --tags --always   # v1.1-<n>-g<sha>
git -C navsim-v2.2 describe --tags --always   # v2.2
```

Upstream has both a branch and a tag named `v1.1`, and `--branch v1.1` checks
out the branch, which is several commits ahead of the tag. `describe` therefore
reports `v1.1-<n>-g<sha>` rather than a bare `v1.1`; that is the expected result
for a correct checkout. Only `v2.2` reports its tag exactly.

The official NAVSIM repository has a `v1.1` release and several v2 release
tags (`v2.0`, `v2.1`, `v2.1.1`, `v2.1.2`, and `v2.2`). The official repository
README treats its `main` branch as NAVSIM v2. This project pins `v2.2` for the
public EPDMS command so that the devkit version is explicit and reproducible.

The two NAVSIM versions are not interchangeable. NAVSIM v1.1 and v2 have
different dataclasses, scoring configuration, scene filters, and metric-cache
formats.

## Datasets

All data is public. NAVSIM ships the download scripts, so use those rather than
reconstructing the URLs:

```bash
cd navsim-v1.1/download

./download_maps.sh          # nuPlan maps v1.1
./download_trainval.sh      # OpenScene v1.1 trainval, used for training
./download_test.sh          # OpenScene v1.1 test, used for navtest scoring

# Only for the v2 two-stage evaluation:
cd ../../navsim-v2.2/download
./download_navhard_two_stage.sh
```

`download_navtrain_hf.sh` fetches only the `navtrain` subset of `trainval`,
which is what the training configs read; the full `trainval` split is several
terabytes and is not required.

Arrange the result so that the paths below resolve, which is the layout the
official NAVSIM install guide describes:

```text
$OPENSCENE_DATA_ROOT/
├── navsim_logs/{trainval,test}
└── sensor_blobs/{trainval,test}
$NUPLAN_MAPS_ROOT/          # nuplan-maps-v1.1, needed for metric caching
```

## Checkpoints

The trained WA-JEPA weights are published at
[AFARI-Research/WA-JEPA](https://huggingface.co/AFARI-Research/WA-JEPA). That
checkpoint is the stage-2 output described under [Training](#training) and is
what the evaluation commands expect in `CHECKPOINT`.

The V-JEPA 2.1 ViT-L/16 pretrained encoder is released by Meta with the V-JEPA 2
code (`facebookresearch/vjepa2`); download `vjepa2_1_vitl_dist_vitG_384.pt` from
there. The NAVSIM presets set `require_pretrained: true` and load it before the
WA-JEPA state dict, so it is required for both training and NAVSIM evaluation.
`configs/wa_jepa_hugsim.yaml` is the exception: it disables pretrained loading
because closed-loop inference restores the full state dict directly.

A WA-JEPA checkpoint can be either a `model_state_dict.pt` file or a directory
with the following layout:

```text
checkpoint/
├── model_state_dict.pt
└── state.pt                 # optional for inference
```

Set the common paths:

```bash
export VJEPA2_CKPT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
export CHECKPOINT=/path/to/wa-jepa/checkpoint
export OPENSCENE_DATA_ROOT=/path/to/openscene-v1.1
export NUPLAN_MAPS_ROOT=/path/to/nuplan-maps
export PYTHON_BIN=/path/to/python
```

The evaluation scripts accept `OPENSCENE_ROOT` as an alias for
`OPENSCENE_DATA_ROOT`; the training configs read only the latter.

`CHECKPOINT` must match the architecture in the selected evaluation YAML.

## Results

All numbers below come from the published checkpoint, evaluated with the
commands in this README.

NAVSIM `navtest`, open loop:

| Devkit | Metric | NC | DAC | TTC | Comf. | EP | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1.1 | PDMS | 99.5 | 98.3 | 97.7 | 100 | 85.0 | **91.8** |

| Devkit | Metric | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2.2 | EPDMS | 99.4 | 98.2 | 99.7 | 99.9 | 87.8 | 98.9 | 98.3 | 98.3 | 88.1 | **91.7** |

HUGSIM, closed loop, all 436 scenarios, zero-shot (no training stage uses
HUGSIM-rendered observations), with the heading-corrected controller:

| NC | DAC | TTC | C | PDMS | RC | HD-Score |
| --- | --- | --- | --- | --- | --- | --- |
| 0.6856 | 0.9635 | 0.6120 | 0.6620 | 0.5717 | 0.5689 | **0.4462** |

HD-Score by difficulty tier and by source dataset:

| Tier | n | HD-Score | | Dataset | n | HD-Score |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 80 | 0.7977 | | nuScenes | 88 | 0.4725 |
| medium | 157 | 0.5563 | | KITTI-360 | 113 | 0.2963 |
| hard | 96 | 0.3060 | | Waymo | 108 | 0.5542 |
| extreme | 103 | 0.1362 | | PandaSet | 127 | 0.4702 |

The overall HD-Score weights the four tiers by their scenario counts. Averaging
uniformly over the four datasets gives 0.4483 and uniformly over all 436
scenarios gives 0.4464, so the number is not an artifact of one aggregation
rule. `close_loop/aggregate_results.py` reports the per-tier and per-dataset
breakdown for a run.

Reproduced results may vary across environments. The NAVSIM scores above are means over
a fixed set of ten inference seeds: across those seeds EPDMS has a standard
deviation of 0.05 and a range of `[91.63, 91.81]`; a single-seed run should
therefore produce a result close to 91.7, but not necessarily identical to it.
In addition to the seed, the GPU,
driver, devkit version, OpenScene copy, and metric cache all affect the score,
and closed-loop HUGSIM is not deterministic across runs. The values above are a
reference rather than an exact reproduction target. The paper documents the
full protocol, the per-seed statistics, and the baselines used for comparison.

## Training

Training runs in two stages. Stage 1 learns future scene prediction on NuPlan
video with the trajectory head off; stage 2 turns the trajectory head on and
finetunes on NAVSIM `navtrain`, initialized from the stage-1 weights.

| | Stage 1 | Stage 2 |
| --- | --- | --- |
| Config | `configs/causal_future_jepa_mv_video_stage1.yaml` | `configs/causal_future_jepa_mv_navsim.yaml` |
| Data | NuPlan `train` | NAVSIM `navtrain` |
| Trajectory head | off | on |
| Losses | `loss_jepa_pred` 1.0 | `loss_jepa_pred` 0.5, `loss_traj` 1.0 |
| `full_mask_prob` | 0.5 | 1.0 |
| Encoder | frozen for the first 25k steps | trainable from step 0 |
| Initialized from | the V-JEPA 2.1 encoder | the V-JEPA 2.1 encoder, then stage 1 |

Both stages read their paths from the environment:

```bash
export WM_SAVE_ROOT=/path/to/experiments
export VJEPA2_CKPT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
```

`WM_SAVE_ROOT` specifies the base directory. Each run creates a separate
`<user>/<name>/<experiment_name>/<timestamp>` directory under it, and that is
where the resolved `config.yaml`, the checkpoints and the load reports go.

Stage 2 additionally needs a NAVSIM checkout on `PYTHONPATH` for the log
dataclasses and the `navtrain` scene filter; `train.sh` prepends
`NAVSIM_DEVKIT_ROOT` for that. Use one checkout for the whole run: the two NAVSIM
versions have different dataclasses, so switching between them mid-recipe changes
what the loader reads.

Both stages default to DeepSpeed ZeRO-2 with bf16
(`configs/deepspeed/ds_zero2_bf16.json`). `train.sh` derives the process count
from `CUDA_VISIBLE_DEVICES`, or from `nvidia-smi` when that is unset; set
`NPROC_PER_NODE` to override it.

### Stage 1: NuPlan video

```bash
export NUPLAN_DATA_ROOT=/path/to/nuplan

bash scripts/training/train.sh \
  --config configs/causal_future_jepa_mv_video_stage1.yaml
```

The trajectory head is disabled here, so the only loss is `loss_jepa_pred` and
NuPlan needs no trajectory labels beyond the ego poses it already carries.
Checkpoints land in `<save_root>/checkpoints/`: `last` after every validation,
and the five best by `val_active_scene_xpred_mse`. A stage-1 checkpoint is used
only to initialize stage 2.

### Stage 2: NAVSIM finetune

```bash
export OPENSCENE_DATA_ROOT=/path/to/openscene-v1.1
export NAVSIM_DEVKIT_ROOT=$PWD/navsim-v1.1
export STAGE1_CKPT=/path/to/stage1/checkpoints/step_24000

bash scripts/training/train.sh \
  --config configs/causal_future_jepa_mv_navsim.yaml
```

`STAGE1_CKPT` sets `model.init_from_checkpoint`, which the configuration marks
as required. A missing or invalid path therefore causes model construction to
fail instead of starting training with only the pretrained encoder. The path can
also be provided as an override:

```bash
bash scripts/training/train.sh \
  --config configs/causal_future_jepa_mv_navsim.yaml \
  model.init_from_checkpoint=[/path/to/stage1/checkpoints/step_24000]
```

Load the weights through `model.init_from_checkpoint` rather than the top-level
`checkpoint`. `init_from_checkpoint` copies model weights only and tolerates a
partial match, which is exactly what a stage change needs, because the trajectory
head that stage 2 adds has no counterpart in a stage-1 checkpoint. The top-level
`checkpoint` is a trainer resume; it also restores the optimizer, scheduler and
step counter, and so continues stage 1's schedule instead of starting stage 2's.

Every load writes `checkpoint_load_report.jsonl` and `.txt`, along with a
`checkpoint_load_keys/` directory, under the run's save root. They record the
loaded ratio and the missing, unexpected and shape-mismatched keys. Review the
report after stage 2 starts. Because `init_checkpoint_allow_partial` is enabled,
training can proceed even when only a small fraction of checkpoint parameters
match.

Training only validates; it never evaluates on the test split. Stage-2 validation
is a held-out 5% of `navtrain`, selected by a hash of the log name, and the
selection metric is a validation loss the model returns. `navtrain` and `navtest`
share no logs, so neither training nor checkpoint selection touches the test
data. Score `navtest` separately after training, with the evaluation commands
below.

The published weights are the output of this stage; see
[Checkpoints](#checkpoints).

CLI dotlist overrides work on any config key, so a short run needs no new file:

```bash
bash scripts/training/train.sh \
  --config configs/causal_future_jepa_mv_navsim.yaml \
  trainer.batch_size_per_gpu=1 trainer.max_epochs=0 trainer.max_steps=20
```

`max_epochs` takes priority over `max_steps` in this trainer. A step-limited
run must therefore also set `max_epochs=0`.

`configs/CONFIG_GUIDE.md` documents every config field.

## Generate NAVSIM metric caches

Metric caches must be generated separately for NAVSIM v1.1 and v2.2. Use the
same cache directory in the corresponding evaluation command. Both versions
need the OpenScene annotations and the nuPlan map files:

```bash
export PYTHON_BIN=/path/to/python
export OPENSCENE_DATA_ROOT=/path/to/openscene-v1.1
export NUPLAN_MAPS_ROOT=/path/to/nuplan-maps
mkdir -p metric-cache
```

Generate the v1.1 PDMS cache:

```bash
export NAVSIM_V1_DEVKIT_ROOT=$PWD/navsim-v1.1

$PYTHON_BIN "$NAVSIM_V1_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest \
  navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  cache.cache_path="$PWD/metric-cache/navsim-v1.1"
```

Generate the v2.2 EPDMS cache:

```bash
export NAVSIM_V2_DEVKIT_ROOT=$PWD/navsim-v2.2

$PYTHON_BIN "$NAVSIM_V2_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest \
  navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  metric_cache_path="$PWD/metric-cache/navsim-v2.2"
```

To force recomputation of an existing cache, append the matching override:

```bash
# v1.1
cache.force_feature_computation=true

# v2.2
force_feature_computation=true
```

## NAVSIM v1.1 PDMS

This is the original NAVSIM `navtest` evaluation and uses the v1.1 devkit,
v1.1 scene filter, and a v1.1 metric cache. It is **not** the v2 EPDMS
evaluation.

Configuration:

```text
configs/wa_jepa_navsim_pdms.yaml
```

Run:

```bash
export NAVSIM_V1_DEVKIT_ROOT=$PWD/navsim-v1.1
export METRIC_CACHE_PATH=$PWD/metric-cache/navsim-v1.1

bash scripts/evaluation/run_navsim_pdms.sh
```

## NAVSIM v2 EPDMS

This uses the NAVSIM v2 `navtest` one-stage scorer and the v2 metric-cache
format. It is **not** the v1.1 PDMS evaluation. The v2 `navtest` scorer is
different from the v2 `navhard_two_stage` hard pseudo-closed-loop benchmark.

Configuration:

```text
configs/wa_jepa_navsim_epdms.yaml
```

Run:

```bash
export NAVSIM_V2_DEVKIT_ROOT=$PWD/navsim-v2.2
export METRIC_CACHE_PATH=$PWD/metric-cache/navsim-v2.2

bash scripts/evaluation/run_navsim_epdms.sh
```

The EPDMS preset uses `flow_inference_seed: 1`, one of the ten inference seeds
behind the reported 91.7 EPDMS in [Results](#results). It is a NAVSIM v2
`navtest` one-stage evaluation, so a single run reproduces one seed rather than
the ten-seed mean.

PDMS and EPDMS use different YAML files, devkits, and metric caches. Do not mix
the v1.1 and v2 evaluation assets. In particular:

| Item | PDMS | EPDMS |
| --- | --- | --- |
| Devkit | NAVSIM `v1.1` | NAVSIM `v2.2` |
| Script | `run_pdm_score.py` from v1.1 | `run_pdm_score_one_stage.py` from v2 |
| Split | `navtest` | `navtest` |
| Metric | PDM/PDMS | EPDMS-style navtest score |
| Cache | v1.1 metric cache | v2 metric cache |
| YAML | `wa_jepa_navsim_pdms.yaml` | `wa_jepa_navsim_epdms.yaml` |

## Multi-GPU EPDMS

The multi-GPU launcher splits `navtest` logs across visible GPUs and merges the
result CSV files. It reads the same `NAVSIM_V2_DEVKIT_ROOT` and
`METRIC_CACHE_PATH` as the single-GPU EPDMS command above:

```bash
export NAVSIM_V2_DEVKIT_ROOT=$PWD/navsim-v2.2
export METRIC_CACHE_PATH=$PWD/metric-cache/navsim-v2.2

export NAVSIM_NUM_GPUS=8
bash scripts/evaluation/run_navsim_navtest_epdms.sh
```

With `NAVSIM_NUM_GPUS` unset the launcher counts `CUDA_VISIBLE_DEVICES`, or
falls back to `nvidia-smi -L`.

Evaluation results are written under `outputs/` by default. Set `OUTPUT_DIR` to
use another location.

## Dry Run

Use `--dry-run` to inspect the generated NAVSIM command without starting model
inference:

```bash
bash scripts/evaluation/run_navsim_pdms.sh --dry-run
bash scripts/evaluation/run_navsim_epdms.sh --dry-run
```

## HUGSIM closed-loop evaluation

HUGSIM is a 3D Gaussian Splatting closed-loop benchmark: the model's plan is fed back into
the simulator, so the ego vehicle moves, the cameras re-render from the new pose, and
errors compound. NAVSIM, by contrast, scores a trajectory against a recorded log.

The official benchmark is 436 scenarios across four splits (nuScenes 88, KITTI-360 113,
Waymo 108, PandaSet 127) at tiers easy / medium / hard / extreme.

Configuration:

```text
configs/wa_jepa_hugsim.yaml
```

Same architecture as the EPDMS preset, with pretrained-encoder loading disabled because the
trained state dict is restored directly. It adds a `hugsim` block holding the camera
mapping, the simulator step and the command permutation.

Install HUGSIM per its own README, obtain the data package, then rewrite the official
scenario files once:

```bash
export HUGSIM_ROOT=/path/to/HUGSIM
export HUGSIM_DATA_ROOT=/path/to/hugsim_data
export SIM_PYTHON=/path/to/hugsim-env/bin/python

python close_loop/prep_scenarios.py --data-root "$HUGSIM_DATA_ROOT"
```

Run one split:

```bash
export PYTHON_BIN=/path/to/wa-jepa-env/bin/python
export CHECKPOINT=/path/to/wa-jepa/checkpoint

# Single scenario first; it exercises the whole chain in a few minutes.
# Scenario arguments are the simulator's scenario names, which use hyphens; the
# output directories use underscores instead.
bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes scene-0013-easy-00

# Then the split.
bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes
```

The default is 8 GPUs with 4 scenarios each, so 32 concurrent scenarios; a scenario needs
about 8.5 GB. Set `HUGSIM_NUM_GPUS`, `HUGSIM_PER_GPU` or `JOBS` to change that. All 436
scenarios take roughly 3-5 hours at that concurrency.

Aggregate into a report:

```bash
python close_loop/aggregate_results.py --dataset nuscenes \
  --results-root outputs/hugsim/nuscenes_ltf \
  --log-dir      outputs/hugsim/nuscenes_logs

# after all four splits
python close_loop/aggregate_results.py --combined --out-dir outputs/hugsim
```

The report carries the official metrics (NC, DAC, TTC, C, PDMS, RC, HD-Score) overall and
per tier, the HUGSIM paper's published baselines for that split as background, and an audit
section. The audit is not optional: the client falls back to a braking plan when inference
raises, and a scenario that braked throughout still scores well, so those scenarios are
listed and excluded from a second mean.

Means are uniform over scenarios within a split, and the combined report averages the four
splits uniformly. Publications on this benchmark also weight the four difficulty tiers by
their scenario counts (80 / 157 / 96 / 103), which moves HD-Score in the third decimal;
state which rule a number uses.

Before a long run, check the conversions offline. Both run on CPU without a checkpoint:

```bash
python close_loop/selfcheck.py
python -m unittest tests.test_hugsim_planner
```

### Two fixes applied to HUGSIM

HUGSIM's own files are never edited. Both fixes are monkey-patches applied in the
evaluation process before `closed_loop` is imported, so a simulator checkout stays
byte-identical to upstream and both patches are entry scripts in `close_loop/` here.

- **Controller heading.** `traj2control` computes the target heading as
  `arctan2(d_forward, d_right)` while the iLQR state it feeds expects
  `arctan2(d_right, d_forward)`, so a straight plan gets a 90 degree target and a nearly
  straight one saturates the steering rate. The resulting score penalty varies with each
  plan's lateral jitter, so it does not cancel across methods. Rankings obtained with the
  uncorrected controller therefore reflect both planning quality and trajectory jitter.
  Every run applies the correction:
  `run_hugsim_benchmark.sh` always enters through `close_loop/run_fixed_controller.py`.
  Upstream PR #57 fixes the same defect and is still open.
- **Two failures that prevent 16 of the 436 scenarios from being scored.**
  `ConstantPlanner.__init__` rejects the
  `max_t` that fifteen PandaSet `medium_02` scenarios pass it, and `IDM.update` indexes an
  empty lookahead when a background vehicle has one waypoint left. Both abort the scenario
  before it can be scored. `close_loop/run_upstream_compat.py` patches both
  (`HUGSIM_COMPAT=1`, the default) without changing the score of any scenario that already
  ran.

Baselines therefore have to be re-scored, not quoted: the UniAD / VAD / LTF values in the
HUGSIM paper were produced under the uncorrected controller and on an earlier scenario
release, so the columns `aggregate_results.py` prints are background rather than a
comparison. Our paper instead re-runs every baseline under this protocol — same scenarios,
same heading-corrected controller, same ground-truth driving commands, same aggregation —
with the corrected evaluation code released alongside DrivoR, so metric computation is
identical across methods.

The `max_t` patch accepts and ignores the argument, so background traffic in those fifteen
scenarios maintains a constant velocity throughout the rollout. The simulator does not
define the intended semantics of `max_t`; however, plausible interpretations such as
stopping, despawning, or ending the plan at that time are equivalent to constant-velocity
motion before `t = max_t`. Because `dt` is 0.25 s,
`max_t: 25` is step 100: a rollout that ends before step 100 is unaffected by the patch, and
one that runs past it is scored under an assumption instead.

How far a rollout gets depends on the planner, not on the code, so this is checked after
every PandaSet run rather than argued once:

```bash
python close_loop/check_maxt_reached.py \
  --results-root outputs/hugsim/pandaset_ltf --dataset pandaset
```

The script exits non-zero if any of the fifteen reached step 100. Seven of them did on the
reported checkpoint, the longest at 152 steps; those scores are reported separately from the
mean.

`HUGSIM_COMPAT=0` skips the two crash patches, which reproduces a run made before they
existed: PandaSet then scores 112 of its 127 scenarios. The aggregator's PandaSet caveat
states which denominator a report used.

Full guide, including the simulator-to-model conversions and the split-specific caveats:

```text
docs/hugsim_closed_loop.md
```

### Videos of the closed loop

A closed-loop score indicates how far the vehicle progressed, but does not describe its
behavior. Set `SAVE_VIZ=1` during a run to archive the `CAM_FRONT` rendering, a mosaic of
the four model inputs, and the predicted plan at every step:

```bash
SAVE_VIZ=1 bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes
```

The archive requires approximately 40 MB per scenario and does not affect planning. It
must be enabled before the run because it cannot be reconstructed afterwards.

Render one scenario, a selection, or every split in parallel:

```bash
# One scenario.
python close_loop/make_video.py \
  --root    outputs/hugsim/nuscenes_ltf \
  --out-dir outputs/hugsim/videos/nuscenes \
  --scenes  scene-0013_easy_00 \
  --dataset nuscenes --hugsim-root "$HUGSIM_ROOT"

# The best and the worst scenario of every difficulty tier.
python close_loop/make_video.py \
  --root    outputs/hugsim/nuscenes_ltf \
  --out-dir outputs/hugsim/videos/nuscenes \
  --pick    outputs/hugsim/results_nuscenes.json \
  --dataset nuscenes --hugsim-root "$HUGSIM_ROOT"

# All four splits, eight scenarios at a time.
python close_loop/render_videos.py --workers 8 \
  --hugsim-root "$HUGSIM_ROOT" --skip-existing
```

Each frame carries the planned trajectory projected onto the road surface, a bird's-eye
panel with the plan, the other agents and the path driven so far, and a status bar with
step, time, speed, navigation command and the scenario's scores. Steps where inference
raised are flagged `FALLBACK: inference failed`, so a video cannot pass off the client's
braking plan as the model's. `render_videos.py` also writes an `index.md` grouped by split
and tier and sorted by HD-Score.

Rendering needs the camera geometry, and the four splits do not share it: the fields of
view differ (65.12 vs 58.12 degrees horizontally), KITTI-360 rectifies with a 5 degree
pitch, and PandaSet mounts the camera 0.4 m lower. Both scripts therefore read the
simulator's `<dataset>_camera.yaml` — via `--hugsim-root` and `--dataset`, or
`--camera-config` directly — and cross-check it against the parameters the client recorded
during the run. A mismatch stops the render rather than drawing a plausible overlay with
the wrong geometry. Rendering is CPU-only and accesses only the archived data.

## Tests

```bash
python -m unittest \
  tests.test_inference_config \
  tests.test_navsim_agent_imports \
  tests.test_hugsim_planner \
  tests.test_hugsim_video
```

## Notes

- The current public presets require four camera views and four history frames.
- The released full model checkpoint is loaded strictly.
- Ablation checkpoints need their own matching YAML configurations.
- NAVSIM v2 one-stage EPDMS should not be compared directly with hard
  pseudo-closed-loop or two-stage evaluation results.

## License

This repository is released under Apache-2.0; see `LICENSE`.

`third_party/vjepa2/` is vendored from Meta's V-JEPA 2 release and keeps its own
licenses (`third_party/vjepa2/LICENSE`, `third_party/vjepa2/APACHE-LICENSE`).
NAVSIM, nuPlan, OpenScene and HUGSIM are not redistributed here — install them
from their own repositories under their own terms, and obtain their datasets from
the original sources.
