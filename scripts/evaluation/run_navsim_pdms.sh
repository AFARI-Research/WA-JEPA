#!/usr/bin/env bash
set -euo pipefail

# Official NAVSIM v1.1 PDM/PDMS evaluation for the inference-only wa-jepa
# branch.  The checkpoint is loaded through eval.navsim_agent.WorldModelNavsimAgent.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAVSIM_ROOT="${NAVSIM_ROOT:-${NAVSIM_V1_DEVKIT_ROOT:-/path/to/navsim-v1.1}}"
OPENSCENE_ROOT="${OPENSCENE_ROOT:-${OPENSCENE_DATA_ROOT:-}}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_V1_METRIC_CACHE:-${NAVSIM_EXP_ROOT:+${NAVSIM_EXP_ROOT}/metric_cache}}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/navsim_pdms}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/wa_jepa_navsim_pdms.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-wa_jepa_navtest_pdms}"
NAVSIM_WORKER="${NAVSIM_WORKER:-single_machine_thread_pool}"
NAVSIM_MAX_WORKERS="${NAVSIM_MAX_WORKERS:-1}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${REPO_ROOT}:${NAVSIM_ROOT}:${PYTHONPATH:-}"

extra_args=()
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    extra_args+=("${arg}")
  fi
done

[[ -n "${OPENSCENE_ROOT}" ]] || {
  echo "[ERROR] Set OPENSCENE_ROOT or OPENSCENE_DATA_ROOT." >&2
  exit 1
}
[[ -n "${METRIC_CACHE_PATH}" ]] || {
  echo "[ERROR] Set METRIC_CACHE_PATH, NAVSIM_V1_METRIC_CACHE, or NAVSIM_EXP_ROOT." >&2
  exit 1
}
[[ -n "${CHECKPOINT}" ]] || {
  echo "[ERROR] Set CHECKPOINT." >&2
  exit 1
}
[[ -f "${CONFIG}" ]] || {
  echo "[ERROR] Config does not exist: ${CONFIG}" >&2
  exit 1
}
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -e "${CHECKPOINT}" ]] || {
    echo "[ERROR] Checkpoint does not exist: ${CHECKPOINT}" >&2
    exit 1
  }
  [[ -n "${VJEPA2_CKPT:-}" ]] || {
    echo "[ERROR] Set VJEPA2_CKPT to the V-JEPA 2.1 pretrained checkpoint." >&2
    exit 1
  }
fi

args=(
  train_test_split=navtest
  experiment_name="${EXPERIMENT_NAME}"
  output_dir="${OUTPUT_DIR}"
  navsim_log_path="${OPENSCENE_ROOT}/navsim_logs/test"
  sensor_blobs_path="${OPENSCENE_ROOT}/sensor_blobs/test"
  metric_cache_path="${METRIC_CACHE_PATH}"
  agent._target_=eval.navsim_agent.WorldModelNavsimAgent
  +agent.config_path="${CONFIG}"
  +agent.checkpoint_path="${CHECKPOINT}"
  worker="${NAVSIM_WORKER}"
  worker.max_workers="${NAVSIM_MAX_WORKERS}"
)

command=(
  "${PYTHON_BIN}" "${NAVSIM_ROOT}/navsim/planning/script/run_pdm_score.py"
  "${args[@]}" "${extra_args[@]}"
)
printf '[wa-jepa] %q ' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi
exec "${command[@]}"
