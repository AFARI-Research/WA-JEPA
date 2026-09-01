#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAVSIM_ROOT="${NAVSIM_ROOT:-${NAVSIM_V2_DEVKIT_ROOT:-${NAVSIM_DEVKIT_ROOT:-/path/to/navsim-v2.2}}}"
OPENSCENE_ROOT="${OPENSCENE_ROOT:-${OPENSCENE_DATA_ROOT:-}}"
CACHE_PATH="${CACHE_PATH:-${METRIC_CACHE_PATH:-${NAVSIM_V2_NAVTEST_METRIC_CACHE:-${NAVSIM_V2_METRIC_CACHE:-${NAVSIM_EXP_ROOT:+${NAVSIM_EXP_ROOT}/metric_cache}}}}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/navsim_navtest_epdms}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/wa_jepa_navsim_epdms.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-worldmodel_navtest_epdms}"
NAVSIM_WORKER="${NAVSIM_WORKER:-single_machine_thread_pool}"
NAVSIM_MAX_WORKERS="${NAVSIM_MAX_WORKERS:-1}"
TRAFFIC_AGENTS_POLICY="${TRAFFIC_AGENTS_POLICY:-navsim.traffic_agents_policies.log_replay_traffic_agents.LogReplayTrafficAgents}"
CACHE_ONLY="${CACHE_ONLY:-0}"
SKIP_CACHE_CHECK="${SKIP_CACHE_CHECK:-0}"
NAVSIM_NUM_GPUS="${NAVSIM_NUM_GPUS:-}"
NAVSIM_RANKS="${NAVSIM_RANKS:-}"
NAVSIM_MERGE_ONLY="${NAVSIM_MERGE_ONLY:-0}"
export PYTHONPATH="${REPO_ROOT}:${NAVSIM_ROOT}:${PYTHONPATH:-}"

if [[ -z "${OPENSCENE_ROOT}" ]]; then
  echo "[ERROR] Set OPENSCENE_ROOT or OPENSCENE_DATA_ROOT." >&2
  exit 1
fi
if [[ -z "${CACHE_PATH}" ]]; then
  echo "[ERROR] Set METRIC_CACHE_PATH, CACHE_PATH, NAVSIM_V2_NAVTEST_METRIC_CACHE, NAVSIM_V2_METRIC_CACHE, or NAVSIM_EXP_ROOT." >&2
  exit 1
fi

if [[ "${CACHE_ONLY}" == "1" ]]; then
  "${PYTHON_BIN}" -m eval.navsim_eval_launcher \
    --navsim-version v2 \
    --navsim-v2-root "${NAVSIM_ROOT}" \
    --openscene-data-root "${OPENSCENE_ROOT}" \
    --metric-cache-path "${CACHE_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --config "${CONFIG}" \
    --train-test-split navtest \
    --data-split-name test \
    --python-executable "${PYTHON_BIN}" \
    --cache-only \
    "$@"
  exit $?
fi

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] Set CHECKPOINT for NAVSIM evaluation." >&2
  exit 1
fi

if [[ "${SKIP_CACHE_CHECK}" != "1" ]]; then
  CACHE_PATH="${CACHE_PATH}" "${PYTHON_BIN}" - <<'PY'
import csv
import lzma
import os
import pickle
import sys
from pathlib import Path

cache_path = Path(os.environ["CACHE_PATH"])
metadata_dir = cache_path / "metadata"
metadata_files = sorted(metadata_dir.glob("*.csv"))
if not metadata_files:
    raise SystemExit(f"[ERROR] No metric cache metadata csv found under {metadata_dir}")
with metadata_files[0].open() as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"[ERROR] Empty metric cache metadata: {metadata_files[0]}")
cache_file = Path(rows[0].get("file_path") or rows[0].get("path") or next(iter(rows[0].values())))
if not cache_file.is_absolute():
    cache_file = cache_path / cache_file
with lzma.open(cache_file, "rb") as f:
    metric_cache = pickle.load(f)
required = ["map_parameters", "past_human_trajectory", "human_trajectory"]
missing = [name for name in required if not hasattr(metric_cache, name)]
try:
    metric_cache.observation.detections_tracks
except (AttributeError, TypeError):
    missing.append("observation.detections_tracks")
if missing:
    raise SystemExit(
        "[ERROR] Metric cache is not compatible with NAVSIM v2 EPDMS "
        f"(missing {missing}). Rebuild it with:\n"
        "  CACHE_ONLY=1 bash scripts/evaluation/run_navsim_navtest_epdms.sh\n"
        f"Current cache: {cache_path}"
    )
PY
fi

if [[ -z "${NAVSIM_NUM_GPUS}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    NAVSIM_NUM_GPUS="$("${PYTHON_BIN}" - <<'PY'
import os
print(len([item for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]))
PY
)"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NAVSIM_NUM_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  fi
fi
NAVSIM_NUM_GPUS="${NAVSIM_NUM_GPUS:-1}"
echo "[navtest] using ${NAVSIM_NUM_GPUS} GPU evaluator process(es)"

if [[ "${NAVSIM_NUM_GPUS}" -gt 1 ]]; then
  multigpu_args=(
    -m eval.navsim_navtest_multigpu
    --navsim-root "${NAVSIM_ROOT}"
    --openscene-root "${OPENSCENE_ROOT}"
    --metric-cache-path "${CACHE_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --num-gpus "${NAVSIM_NUM_GPUS}"
    --python-executable "${PYTHON_BIN}"
    --experiment-name "${EXPERIMENT_NAME}"
    --traffic-policy "${TRAFFIC_AGENTS_POLICY}"
  )
  if [[ -n "${NAVSIM_RANKS}" ]]; then
    multigpu_args+=(--ranks "${NAVSIM_RANKS}")
  fi
  if [[ "${NAVSIM_MERGE_ONLY}" == "1" ]]; then
    multigpu_args+=(--merge-only)
  fi
  exec "${PYTHON_BIN}" "${multigpu_args[@]}" "$@"
fi

args=(
  --navsim-version v2
  --navsim-v2-root "${NAVSIM_ROOT}"
  --openscene-data-root "${OPENSCENE_ROOT}"
  --metric-cache-path "${CACHE_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --config "${CONFIG}"
  --train-test-split navtest
  --data-split-name test
  --experiment-name "${EXPERIMENT_NAME}"
  --python-executable "${PYTHON_BIN}"
  --device "${DEVICE}"
  --one-stage
  --override "worker=${NAVSIM_WORKER} worker.max_workers=${NAVSIM_MAX_WORKERS}"
  --override "traffic_agents=non_reactive traffic_agents_policy.non_reactive._target_=${TRAFFIC_AGENTS_POLICY}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi

exec "${PYTHON_BIN}" -m eval.navsim_eval_launcher "${args[@]}" "$@"
