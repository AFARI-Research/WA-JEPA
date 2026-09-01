#!/usr/bin/env bash
set -euo pipefail

# Official NAVSIM v2 one-stage EPDMS evaluation for wa-jepa.
#
# NAVSIM v2's public navtest/EPDMS setup uses the navtest scene filter and
# non-reactive log-replay traffic.  It is deliberately separate from the v1.1
# PDMS wrapper because the devkits and metric caches are not interchangeable.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAVSIM_ROOT="${NAVSIM_ROOT:-${NAVSIM_V2_DEVKIT_ROOT:-/path/to/navsim-v2.2}}"
OPENSCENE_ROOT="${OPENSCENE_ROOT:-${OPENSCENE_DATA_ROOT:-}}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_V2_METRIC_CACHE:-${NAVSIM_EXP_ROOT:+${NAVSIM_EXP_ROOT}/metric_cache}}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/navsim_epdms}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/wa_jepa_navsim_epdms.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-wa_jepa_navtest_epdms}"
NAVSIM_WORKER="${NAVSIM_WORKER:-single_machine_thread_pool}"
NAVSIM_MAX_WORKERS="${NAVSIM_MAX_WORKERS:-1}"
TRAFFIC_POLICY="${TRAFFIC_POLICY:-navsim.traffic_agents_policies.log_replay_traffic_agents.LogReplayTrafficAgents}"
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
  echo "[ERROR] Set METRIC_CACHE_PATH, NAVSIM_V2_METRIC_CACHE, or NAVSIM_EXP_ROOT." >&2
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
  original_sensor_path="${OPENSCENE_ROOT}/sensor_blobs/test"
  metric_cache_path="${METRIC_CACHE_PATH}"
  agent._target_=eval.navsim_agent.WorldModelNavsimAgent
  +agent.config_path="${CONFIG}"
  +agent.checkpoint_path="${CHECKPOINT}"
  agent.trajectory_sampling._target_=nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling
  agent.trajectory_sampling.time_horizon=4
  agent.trajectory_sampling.interval_length=0.5
  worker=single_machine_thread_pool
  worker.max_workers="${NAVSIM_MAX_WORKERS}"
  traffic_agents=non_reactive
  traffic_agents_policy.non_reactive._target_="${TRAFFIC_POLICY}"
)

command=(
  "${PYTHON_BIN}" "${NAVSIM_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py"
  "${args[@]}" "${extra_args[@]}"
)
printf '[wa-jepa] %q ' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi
exec "${command[@]}"
