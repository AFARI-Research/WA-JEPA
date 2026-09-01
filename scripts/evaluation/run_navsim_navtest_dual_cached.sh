#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAVSIM_V11_ROOT="${NAVSIM_V11_DEVKIT_ROOT:-${NAVSIM_V1_DEVKIT_ROOT:-/path/to/navsim-v1.1}}"
NAVSIM_V2_ROOT="${NAVSIM_V2_DEVKIT_ROOT:-/path/to/navsim-v2.2}"
OPENSCENE_ROOT="${OPENSCENE_ROOT:-${OPENSCENE_DATA_ROOT:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/navsim_navtest_dual_cached}"
CONFIG="${CONFIG:-configs/causal_future_jepa_mv_navsim.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NAVSIM_NUM_GPUS="${NAVSIM_NUM_GPUS:-}"
TRAJECTORY_CACHE_PATH="${TRAJECTORY_CACHE_PATH:-${OUTPUT_DIR}/trajectory_cache/navtest_trajectories.pkl}"
V1_CACHE_PATH="${V1_CACHE_PATH:-${NAVSIM_V11_METRIC_CACHE:-${OUTPUT_DIR}/metric_cache_v1_1}}"
V2_CACHE_PATH="${V2_CACHE_PATH:-${NAVSIM_V2_NAVTEST_METRIC_CACHE:-${NAVSIM_V2_METRIC_CACHE:-${OUTPUT_DIR}/metric_cache_v2}}}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_V1="${SKIP_V1:-0}"
SKIP_V2="${SKIP_V2:-0}"
V1_SENSOR_PATH_KEY="${V1_SENSOR_PATH_KEY:-sensor_blobs_path}"
EXPORT_OVERRIDES="${EXPORT_OVERRIDES:-}"
export PYTHONPATH="${REPO_ROOT}:${NAVSIM_V2_ROOT}:${PYTHONPATH:-}"

if [[ -z "${OPENSCENE_ROOT}" ]]; then
  echo "[ERROR] Set OPENSCENE_ROOT or OPENSCENE_DATA_ROOT." >&2
  exit 1
fi
if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] Set CHECKPOINT." >&2
  exit 1
fi

if [[ -z "${NAVSIM_NUM_GPUS}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    NAVSIM_NUM_GPUS="$("${PYTHON_BIN}" - <<'PY'
import os
print(len([x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]))
PY
)"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NAVSIM_NUM_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  fi
fi
NAVSIM_NUM_GPUS="${NAVSIM_NUM_GPUS:-1}"

mkdir -p "${OUTPUT_DIR}"
dry_run_args=()
if [[ "${DRY_RUN}" == "1" ]]; then
  dry_run_args+=(--dry-run)
fi
export_override_args=()
if [[ -n "${EXPORT_OVERRIDES}" ]]; then
  export_override_args+=(--override "${EXPORT_OVERRIDES}")
fi

if [[ "${FORCE_EXPORT}" == "1" || ! -f "${TRAJECTORY_CACHE_PATH}" ]]; then
  "${PYTHON_BIN}" -m eval.navsim_export_trajectory_cache \
    --navsim-root "${NAVSIM_V2_ROOT}" \
    --openscene-root "${OPENSCENE_ROOT}" \
    --output-path "${TRAJECTORY_CACHE_PATH}" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --num-gpus "${NAVSIM_NUM_GPUS}" \
    --python-executable "${PYTHON_BIN}" \
    "${export_override_args[@]}" \
    "${dry_run_args[@]}"
else
  echo "[dual-navsim] reuse trajectory cache: ${TRAJECTORY_CACHE_PATH}"
fi

if [[ "${SKIP_V1}" != "1" ]]; then
  "${PYTHON_BIN}" -m eval.navsim_score_trajectory_cache \
    --navsim-version v1 \
    --navsim-root "${NAVSIM_V11_ROOT}" \
    --openscene-root "${OPENSCENE_ROOT}" \
    --metric-cache-path "${V1_CACHE_PATH}" \
    --trajectory-cache-path "${TRAJECTORY_CACHE_PATH}" \
    --output-dir "${OUTPUT_DIR}/v1_1" \
    --experiment-name worldmodel_navtest_v1_1_cached \
    --python-executable "${PYTHON_BIN}" \
    --v1-sensor-path-key "${V1_SENSOR_PATH_KEY}" \
    "${dry_run_args[@]}"
fi

if [[ "${SKIP_V2}" != "1" ]]; then
  "${PYTHON_BIN}" -m eval.navsim_score_trajectory_cache \
    --navsim-version v2 \
    --navsim-root "${NAVSIM_V2_ROOT}" \
    --openscene-root "${OPENSCENE_ROOT}" \
    --metric-cache-path "${V2_CACHE_PATH}" \
    --trajectory-cache-path "${TRAJECTORY_CACHE_PATH}" \
    --output-dir "${OUTPUT_DIR}/v2" \
    --experiment-name worldmodel_navtest_v2_cached \
    --python-executable "${PYTHON_BIN}" \
    "${dry_run_args[@]}"
fi
