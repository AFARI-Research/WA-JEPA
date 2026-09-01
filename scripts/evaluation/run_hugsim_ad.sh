#!/usr/bin/env bash
# HUGSIM launches the external driving stack as `zsh <this script> <cuda_id> <output_dir>`
# (sim/utils/launch_ad.py). The argument order is fixed upstream and there is no room for
# a third argument, so everything else arrives through the environment.
#
# The shebang is ignored because the simulator invokes the script through `zsh`. Keep the
# body compatible with both shells: use `$0` rather than BASH_SOURCE, and no arrays.
#
# Required environment (exported by run_hugsim_benchmark.sh):
#   WM_PYTHON      absolute path to the python of the WA-JEPA environment
#   WM_CONFIG      inference config, e.g. configs/wa_jepa_hugsim.yaml
#   WM_CHECKPOINT  checkpoint directory or .pt file
# Optional:
#   WM_REPO        repository root (defaults to the parent of this script's directory)
#   WM_RUN_TAG     substring that must appear in the output directory (see below)
#   WM_SAVE_VIZ    1 to archive per-step inputs and plans (default 0)
#   CAMERA_YAML    the simulator's <dataset>_camera.yaml; recorded in viz/camera.json so
#                  close_loop/make_video.py can find the split's geometry again
#   WM_ENV_SETUP   script to source before running, for environments that need it
#   WM_DEVICE      torch device string (default cuda)
#   AD_BASE_LD_LIBRARY_PATH  LD_LIBRARY_PATH captured before the simulator's env script ran
#
# There is deliberately no default checkpoint. A `${WM_CHECKPOINT:-<some path>}` fallback
# has to survive three process layers (benchmark script -> closed_loop.py -> launch_ad.py);
# if any layer drops it, the run silently scores an unintended checkpoint while still
# producing 436 scenarios and no error at all. With SKIP_DONE=1 it gets worse: part of the
# run uses one checkpoint and part another, and the two cannot be told apart afterwards.
set -eu

CUDA_ID="${1:?cuda_id is required}"
OUTPUT="${2:?output_dir is required}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${WM_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${WM_PYTHON:?Set WM_PYTHON to the python of the WA-JEPA environment}"
: "${WM_CONFIG:?Set WM_CONFIG to an inference config}"
: "${WM_CHECKPOINT:?Set WM_CHECKPOINT to the checkpoint to evaluate}"
SAVE_VIZ="${WM_SAVE_VIZ:-0}"
DEVICE="${WM_DEVICE:-cuda}"

# Drop the build-time and toolkit variables inherited from the simulator's environment,
# but keep the driver path. The simulator sources its own env script, which can export a
# CUDA_HOME and library paths for a different CUDA minor version than this environment's
# torch was built against; LD_LIBRARY_PATH outranks a wheel's DT_RUNPATH, so inheriting it
# makes torch load the wrong libcudart. Clearing LD_LIBRARY_PATH outright is equally wrong:
# on many cluster images libcuda.so.1 is only reachable through /usr/local/nvidia/lib64 and
# is absent from ldconfig, and losing it yields "Found no NVIDIA driver on your system".
unset CUDA_HOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CPATH LIBRARY_PATH
BASE_LD="${AD_BASE_LD_LIBRARY_PATH:-/usr/local/nvidia/lib64}"
ENV_LIB="$(cd "$(dirname "$(dirname "${WM_PYTHON}")")" && pwd)/lib"
# The environment's own lib must come first: a system libstdc++ older than the one the
# environment was built against fails with a missing CXXABI symbol, and the traceback
# points at sqlite3 rather than at the real cause.
export LD_LIBRARY_PATH="${ENV_LIB}:${BASE_LD}"

# Do not set PYTHONNOUSERSITE here. Some environments keep torch in the user site-packages
# directory, and masking it turns into `ModuleNotFoundError: No module named 'torch'`.

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

if [ -n "${WM_ENV_SETUP:-}" ]; then
  set +u
  # shellcheck disable=SC1090
  . "${WM_ENV_SETUP}"
  set -u
fi

echo "[hugsim-ad] cuda=${CUDA_ID} output=${OUTPUT}"
echo "[hugsim-ad] config=${WM_CONFIG}"
echo "[hugsim-ad] checkpoint=${WM_CHECKPOINT} viz=${SAVE_VIZ}"

# Preflight. Each assertion covers a failure mode that otherwise only lowers the score or
# hangs the run:
#   * torch.cuda unavailable  -> LD_LIBRARY_PATH is wrong. Without this check the failure
#     surfaces deep inside a `.to("cuda")` call after the model is already built.
#   * checkpoint <-> output directory cross-assertion -> see the WM_CHECKPOINT note above.
#   * existence and size of config and checkpoint -> these usually live on a network mount,
#     where the mount point existing does not mean the path below it does, and a wrong path
#     only raises after the model has been constructed (tens of seconds, times 436).
#   * architecture fields -> the model is built from WM_CONFIG, so config and state dict are
#     self-consistent by construction and `strict=True` cannot catch "a different
#     architecture". Comparing against the fields the adapter depends on can.
# The variables are re-stated on the command line rather than relied upon being exported:
# an optional assertion that is silently skipped because its variable never crossed the
# process boundary is worse than no assertion at all.
WM_OUTPUT="${OUTPUT}" \
WM_CONFIG="${WM_CONFIG}" \
WM_CHECKPOINT="${WM_CHECKPOINT}" \
WM_RUN_TAG="${WM_RUN_TAG:-}" \
WM_EXPECT_STATE_DICT_BYTES="${WM_EXPECT_STATE_DICT_BYTES:-}" \
WM_EXPECT_STEP="${WM_EXPECT_STEP:-}" \
"${WM_PYTHON}" - <<'PY' || exit 1
import os
import sys

import torch

print(f"[preflight] python  {sys.version.split()[0]}")
print(f"[preflight] torch   {torch.__version__}  <- {torch.__file__}")
print(f"[preflight] cuda    available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "[preflight] fatal: no CUDA device. Check that LD_LIBRARY_PATH still contains the "
        "driver directory (AD_BASE_LD_LIBRARY_PATH)."
    )
print(f"[preflight] device  {torch.cuda.get_device_name(0)}")

cfg_path = os.environ["WM_CONFIG"]
ckpt_path = os.environ["WM_CHECKPOINT"]
output = os.environ["WM_OUTPUT"]

tag = os.environ.get("WM_RUN_TAG", "")
if tag and tag not in output:
    raise SystemExit(
        f"[preflight] fatal: WM_RUN_TAG={tag!r} does not appear in the output directory "
        f"{output!r}. Writing one checkpoint's scores into another checkpoint's directory "
        "makes SKIP_DONE merge two models into a single column -- still 436 scenarios, "
        "still no error, but part of the table belongs to a different model."
    )
if tag:
    print(f"[preflight] run tag {tag} present in output dir  ok")

if not os.path.isfile(cfg_path):
    raise SystemExit(f"[preflight] fatal: config not found: {cfg_path}")
print(f"[preflight] config     {os.path.getsize(cfg_path)} B  ok")

if os.path.isdir(ckpt_path):
    sd_path = os.path.join(ckpt_path, "model_state_dict.pt")
    if not os.path.isfile(sd_path):
        raise SystemExit(f"[preflight] fatal: {sd_path} not found (mount missing?)")
elif os.path.isfile(ckpt_path):
    sd_path = ckpt_path
else:
    raise SystemExit(f"[preflight] fatal: checkpoint not found: {ckpt_path}")
print(f"[preflight] state_dict {os.path.getsize(sd_path)} B  ok")

expect_size = os.environ.get("WM_EXPECT_STATE_DICT_BYTES", "")
if expect_size:
    got = os.path.getsize(sd_path)
    if got != int(expect_size):
        raise SystemExit(
            f"[preflight] fatal: state dict is {got} B, expected {expect_size} B. The "
            "checkpoint was overwritten, or only half of it has been flushed."
        )
    print("[preflight] state dict size matches WM_EXPECT_STATE_DICT_BYTES  ok")

state_path = os.path.join(ckpt_path, "state.pt") if os.path.isdir(ckpt_path) else ""
expect_step = os.environ.get("WM_EXPECT_STEP", "")
if expect_step:
    # Checkpoint directories that a training job overwrites in place (`last`) keep the same
    # byte count on every epoch, so a size assertion cannot detect that the weights changed.
    # Reading step/epoch out of state.pt can.
    if not state_path or not os.path.isfile(state_path):
        raise SystemExit(
            f"[preflight] fatal: WM_EXPECT_STEP is set but {state_path or '<file>'} does not "
            "exist, so the checkpoint identity cannot be confirmed."
        )
    st = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(st.get("step", -1)) != int(expect_step):
        raise SystemExit(
            f"[preflight] fatal: checkpoint is at step={st.get('step')} "
            f"(epoch={st.get('epoch')}), expected step={expect_step}. If training resumed "
            "and overwrote this directory, the run becomes a mixture of several models "
            "while still reporting 436 scenarios and no error."
        )
    print(f"[preflight] checkpoint step={st.get('step')} epoch={st.get('epoch')}  ok")

from omegaconf import OmegaConf

m = OmegaConf.load(cfg_path).model
expect = {
    "camera_names": ["cam_l0", "cam_f0", "cam_r0", "cam_b0"],
    "input_hw": [256, 512],
    "num_history_frames": 4,
    "ego_status_dim": 8,
    "trajectory_horizon": 8,
}
for key, want in expect.items():
    got = OmegaConf.to_object(m[key]) if OmegaConf.is_config(m[key]) else m[key]
    if got != want:
        raise SystemExit(
            f"[preflight] fatal: model.{key} = {got!r}, the reported configuration uses "
            f"{want!r}. Two different architectures cannot share one results table."
        )
    print(f"[preflight] arch     {key:20s} {got}  ok")
print("[preflight] all checks passed")
PY

# exec so python replaces this shell. launch_ad.py records the pid of the shell it spawned
# and calls process.kill() after check_alive() times out; without exec the shell is killed
# and python is orphaned, holding GPU memory. Across 436 scenarios those accumulate until
# the GPUs are full.
exec "${WM_PYTHON}" -u "${REPO}/close_loop/hugsim_client.py" \
  --output "${OUTPUT}" \
  --repo "${REPO}" \
  --config "${WM_CONFIG}" \
  --checkpoint "${WM_CHECKPOINT}" \
  --device "${DEVICE}" \
  --camera-config "${CAMERA_YAML:-}" \
  --save-viz "${SAVE_VIZ}"
