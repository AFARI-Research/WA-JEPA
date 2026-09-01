#!/usr/bin/env bash
set -euo pipefail

# HUGSIM closed-loop benchmark for WA-JEPA.
#
# HUGSIM's files are never edited. Upstream supports an external driving stack, so this
# script only orchestrates: it generates the simulator's base config, spreads the scenarios
# of one split across the available GPUs, and runs one process per scenario. The two upstream
# defects and the traj2control heading defect are corrected by in-process patches applied
# before closed_loop is imported; see the comments further down.
#
# Usage:
#   bash scripts/evaluation/run_hugsim_benchmark.sh [dataset] [scenario ...]
#     dataset   nuscenes (default) | kitti360 | waymo | pandaset
#     scenario  optional scenario names (without .yaml) to run a subset, e.g. a smoke test
#
# Required environment:
#   HUGSIM_ROOT       HUGSIM checkout
#   HUGSIM_DATA_ROOT  HUGSIM data package (scenes/, 3DRealCar/, <dataset>/*.yaml)
#   SIM_PYTHON        python of the simulator environment
#   PYTHON_BIN        python of the WA-JEPA environment (may be the same interpreter)
#   CHECKPOINT        checkpoint directory or .pt file to evaluate
# Optional:
#   CONFIG            inference config (default configs/wa_jepa_hugsim.yaml)
#   OUTPUT_DIR        results root (default <repo>/outputs/hugsim)
#   SCENARIO_DIR      prepared scenarios (default close_loop/configs/benchmark/<dataset>)
#   HUGSIM_NUM_GPUS   GPU count (default: detected, falling back to 8)
#   HUGSIM_PER_GPU    scenarios per GPU (default 4)
#   JOBS              total concurrency (default HUGSIM_NUM_GPUS * HUGSIM_PER_GPU)
#   HUGSIM_COMPAT     1 (default) to patch the two upstream defects that abort 16 of the 436
#                     scenarios; 0 reproduces a run made before those patches existed
#   HD_MAP_PATH / HD_MAP_VERSION  only needed by scenarios with load_HD_map: true
#   HUGSIM_ENV_SETUP  script to source for the simulator environment
#   WM_ENV_SETUP      script to source for the WA-JEPA environment
#   WM_RUN_TAG        substring asserted to be present in the output path
#   SKIP_DONE         1 (default) to skip scenarios that already finished
#   SAVE_VIZ          1 to archive per-step model inputs and plans (default 0)
#   DRY_RUN           1 to print the plan and exit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${1:-nuscenes}"
shift || true

HUGSIM_ROOT="${HUGSIM_ROOT:-}"
HUGSIM_DATA_ROOT="${HUGSIM_DATA_ROOT:-}"
SIM_PYTHON="${SIM_PYTHON:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
CHECKPOINT="${CHECKPOINT:-}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/wa_jepa_hugsim.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/hugsim}"
SCENARIO_DIR="${SCENARIO_DIR:-${REPO_ROOT}/close_loop/configs/benchmark/${DATASET}}"
SKIP_DONE="${SKIP_DONE:-1}"
SAVE_VIZ="${SAVE_VIZ:-0}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done
scenario_names=()
for arg in "$@"; do
  [[ "${arg}" == "--dry-run" ]] || scenario_names+=("${arg}")
done

[[ -n "${HUGSIM_ROOT}" ]] || { echo "[ERROR] Set HUGSIM_ROOT to the HUGSIM checkout." >&2; exit 1; }
[[ -n "${HUGSIM_DATA_ROOT}" ]] || { echo "[ERROR] Set HUGSIM_DATA_ROOT to the HUGSIM data package." >&2; exit 1; }
[[ -n "${SIM_PYTHON}" ]] || { echo "[ERROR] Set SIM_PYTHON to the simulator environment's python." >&2; exit 1; }
[[ -n "${PYTHON_BIN}" ]] || { echo "[ERROR] Set PYTHON_BIN to the WA-JEPA environment's python." >&2; exit 1; }
[[ -n "${CHECKPOINT}" ]] || { echo "[ERROR] Set CHECKPOINT to the checkpoint to evaluate." >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "[ERROR] Config does not exist: ${CONFIG}" >&2; exit 1; }
[[ -d "${SCENARIO_DIR}" ]] || {
  echo "[ERROR] Scenario directory ${SCENARIO_DIR} does not exist. Run:" >&2
  echo "  ${SIM_PYTHON} close_loop/prep_scenarios.py --data-root ${HUGSIM_DATA_ROOT}" >&2
  exit 1
}

CAMERA_YAML="${CAMERA_YAML:-${HUGSIM_ROOT}/configs/sim/${DATASET}_camera.yaml}"
KINEMATIC_YAML="${KINEMATIC_YAML:-${HUGSIM_ROOT}/configs/sim/kinematic.yaml}"
for f in "${CAMERA_YAML}" "${KINEMATIC_YAML}"; do
  [[ -f "${f}" ]] || { echo "[ERROR] Missing simulator config: ${f}" >&2; exit 1; }
done

# ---------------- GPU count and concurrency ----------------
# One scenario is two processes: 3DGS rendering (~3.5 GB) plus the planner (~4.9 GB), about
# 8.5 GB together. Four per GPU saturates a 46 GB card at ~25 GB used, so 4 is the compute
# limit rather than the memory limit. Raising it on a larger card mostly costs CPU, since
# every scenario needs two processes.
if [[ -z "${HUGSIM_NUM_GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    HUGSIM_NUM_GPUS="$("${SIM_PYTHON}" - <<'PY'
import os
print(len([item for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]))
PY
)"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    HUGSIM_NUM_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  fi
fi
HUGSIM_NUM_GPUS="${HUGSIM_NUM_GPUS:-8}"
[[ "${HUGSIM_NUM_GPUS}" -ge 1 ]] || HUGSIM_NUM_GPUS=1
HUGSIM_PER_GPU="${HUGSIM_PER_GPU:-4}"
JOBS="${JOBS:-$(( HUGSIM_NUM_GPUS * HUGSIM_PER_GPU ))}"

# The slot directory must be unique per run and empty at the start: a leftover directory
# counts as an occupied slot, which silently reduces concurrency without any error.
GPU_LOCK_DIR="${OUTPUT_DIR}/.gpuslots_${DATASET}_$$"

# closed_loop.py appends `--ad` to base.output_dir, so the trailing underscore below plus
# "ltf" is the directory the results actually land in. The base config is generated here
# rather than checked in, which removes the class of mismatch where the config and the
# script disagree about where results go.
OUT_PREFIX="${OUTPUT_DIR}/${DATASET}_"
OUT_ROOT="${OUT_PREFIX}ltf"
LOG_DIR="${OUTPUT_DIR}/${DATASET}_logs"
BASE_YAML="${OUTPUT_DIR}/${DATASET}_base.yaml"

# Upstream's traj2control derives the target heading with the arguments to arctan2 swapped,
# so a straight plan is handed a target heading of 90 degrees and a nearly straight one swings
# the target across most of its range every step and saturates the steering rate. The score it
# costs scales with each plan's lateral jitter, so it is not an offset that cancels between
# methods, and a ranking measured through it is partly a ranking of jitter. Every run here
# corrects it; the reasoning and the equivalence check against upstream PR #57 are in that
# file's docstring.
HUGSIM_ENTRY="${REPO_ROOT}/close_loop/run_fixed_controller.py"

# Two upstream defects abort 16 of the 436 scenarios before they can be scored: the data
# package passes a `max_t` that upstream's ConstantPlanner never implemented, and IDM indexes
# an empty tensor when a background vehicle has one waypoint left. Both are patched in-process
# by run_upstream_compat.py, which then runs whatever entry was selected above -- so the
# heading fix composes with it instead of duplicating it. Each patch is a no-op
# wherever the unpatched code does not crash; the reasoning is in that file's docstring.
# HUGSIM_COMPAT=0 skips them, which is what reproduces a run made before they existed (that
# run scores PandaSet over 112 of its 127 scenarios).
HUGSIM_COMPAT="${HUGSIM_COMPAT:-1}"
HUGSIM_COMPAT_INNER=""
if [[ "${HUGSIM_COMPAT}" == "1" ]]; then
  HUGSIM_COMPAT_INNER="${HUGSIM_ENTRY}"
  HUGSIM_ENTRY="${REPO_ROOT}/close_loop/run_upstream_compat.py"
fi

# ---------------- scenario list ----------------
scenarios=()
if [[ ${#scenario_names[@]} -gt 0 ]]; then
  for n in "${scenario_names[@]}"; do
    [[ -f "${SCENARIO_DIR}/${n}.yaml" ]] || {
      echo "[ERROR] ${SCENARIO_DIR}/${n}.yaml not found" >&2; exit 1; }
    scenarios+=("${SCENARIO_DIR}/${n}.yaml")
  done
elif [[ -f "${SCENARIO_DIR}/runnable.txt" ]]; then
  # prep_scenarios.py already dropped the scenes whose 3DGS assets are missing.
  while IFS= read -r line; do
    [[ -n "${line}" ]] && scenarios+=("${line}")
  done < "${SCENARIO_DIR}/runnable.txt"
else
  for f in "${SCENARIO_DIR}"/*.yaml; do
    scenarios+=("${f}")
  done
fi

if [[ ${#scenarios[@]} -eq 0 ]]; then
  echo "[ERROR] no scenarios selected under ${SCENARIO_DIR}" >&2
  exit 1
fi

# `grep -l` exits 1 when nothing matches, which under `set -o pipefail` would abort the run
# here rather than report "no HD map scenarios".
n_hd_map="$( { grep -l '^load_HD_map:[[:space:]]*[Tt]rue' "${SCENARIO_DIR}"/*.yaml 2>/dev/null || true; } | wc -l)"
if [[ "${n_hd_map}" -gt 0 && -z "${HD_MAP_PATH:-}" ]]; then
  echo "[WARN] ${n_hd_map} scenarios in ${DATASET} set load_HD_map: true, but HD_MAP_PATH is" >&2
  echo "       unset. Those scenarios will fail when the simulator reads base.HD_map." >&2
fi

echo "=== HUGSIM ${DATASET}: ${#scenarios[@]} scenarios ==="
echo "entry:      ${HUGSIM_ENTRY:-closed_loop.py}"
echo "compat:     HUGSIM_COMPAT=${HUGSIM_COMPAT}  inner=${HUGSIM_COMPAT_INNER:-closed_loop.py}"
echo "gpus:       ${HUGSIM_NUM_GPUS} x ${HUGSIM_PER_GPU} scenarios/gpu => concurrency ${JOBS}"
echo "config:     ${CONFIG}"
echo "checkpoint: ${CHECKPOINT}"
echo "scenarios:  ${SCENARIO_DIR}"
echo "results:    ${OUT_ROOT}"
echo "logs:       ${LOG_DIR}"
echo "resume:     SKIP_DONE=${SKIP_DONE}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "(dry run, nothing executed)"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${OUT_ROOT}"
rm -rf "${GPU_LOCK_DIR}"
mkdir -p "${GPU_LOCK_DIR}"
trap 'rm -rf "${GPU_LOCK_DIR}"' EXIT

cat > "${BASE_YAML}" <<EOF
# Generated by scripts/evaluation/run_hugsim_benchmark.sh -- do not edit by hand.
realcar_path: ${HUGSIM_DATA_ROOT}/3DRealCar
model_base: ${HUGSIM_DATA_ROOT}/scenes/${DATASET}
# closed_loop.py accepts --ad uniad|vad|ltf only; the ltf slot carries our driving stack.
uniad_path: ${REPO_ROOT}/scripts/evaluation/run_hugsim_ad.sh
vad_path: ${REPO_ROOT}/scripts/evaluation/run_hugsim_ad.sh
ltf_path: ${REPO_ROOT}/scripts/evaluation/run_hugsim_ad.sh
output_dir: ${OUT_PREFIX}
HD_map:
  path: ${HD_MAP_PATH:-}
  version: ${HD_MAP_VERSION:-}
EOF

# Capture a clean LD_LIBRARY_PATH before the simulator's environment script runs. On many
# cluster images that value is the only source of the driver's libcuda.so.1 (it is not in
# ldconfig), and the driving stack must not lose it while discarding the simulator's
# CUDA-version-specific paths.
export AD_BASE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/nvidia/lib64}"

if [[ -n "${HUGSIM_ENV_SETUP:-}" ]]; then
  set +u
  # shellcheck disable=SC1090
  . "${HUGSIM_ENV_SETUP}"
  set -u
fi

export HUGSIM_ROOT SIM_PYTHON BASE_YAML CAMERA_YAML KINEMATIC_YAML OUT_ROOT LOG_DIR
export HUGSIM_NGPU="${HUGSIM_NUM_GPUS}"
export HUGSIM_PER_GPU GPU_LOCK_DIR SKIP_DONE
[[ -n "${HUGSIM_ENTRY}" ]] && export HUGSIM_ENTRY
[[ -n "${HUGSIM_COMPAT_INNER}" ]] && export HUGSIM_COMPAT_INNER
# Read by scripts/evaluation/run_hugsim_ad.sh, which the simulator spawns.
export WM_PYTHON="${PYTHON_BIN}"
export WM_REPO="${REPO_ROOT}"
export WM_CONFIG="${CONFIG}"
export WM_CHECKPOINT="${CHECKPOINT}"
export WM_SAVE_VIZ="${SAVE_VIZ}"
[[ -n "${WM_RUN_TAG:-}" ]] && export WM_RUN_TAG
[[ -n "${WM_ENV_SETUP:-}" ]] && export WM_ENV_SETUP
[[ -n "${WM_EXPECT_STATE_DICT_BYTES:-}" ]] && export WM_EXPECT_STATE_DICT_BYTES
[[ -n "${WM_EXPECT_STEP:-}" ]] && export WM_EXPECT_STEP

date
# xargs as a work pool: scenario runtimes differ by more than 10x (an early collision ends
# in ~80 s, a full rollout in ~19 min), so manual sharding leaves the fast shard idle.
set +e
printf '%s\n' "${scenarios[@]}" \
  | xargs -P "${JOBS}" -I{} bash "${REPO_ROOT}/scripts/evaluation/run_hugsim_one_scenario.sh" {}
xargs_rc=$?
set -e

echo ""
echo "=== finished ==="
date
# This counts every eval.json under OUT_ROOT, including scenarios completed by earlier runs,
# so it does not equal the number requested when running a subset.
n_eval="$( { ls "${OUT_ROOT}"/*/eval.json 2>/dev/null || true; } | wc -l)"
echo "requested ${#scenarios[@]} scenarios; ${OUT_ROOT} now holds ${n_eval} eval.json files"
echo "next: ${PYTHON_BIN} close_loop/aggregate_results.py --dataset ${DATASET} \\"
echo "        --results-root ${OUT_ROOT} --log-dir ${LOG_DIR} --scenario-dir ${SCENARIO_DIR}"
exit "${xargs_rc}"
