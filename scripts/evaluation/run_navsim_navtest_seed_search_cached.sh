#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DUAL_SCRIPT="${DUAL_SCRIPT:-scripts/evaluation/run_navsim_navtest_dual_cached.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/navsim_navtest_seed_search_cached}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/seed_search_summary.csv}"
SEED_LIST="${SEED_LIST:-}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-}"
SEED_STRIDE="${SEED_STRIDE:-1}"
MAX_RUNS="${MAX_RUNS:-0}"
SKIP_COMPLETED_SEEDS="${SKIP_COMPLETED_SEEDS:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
FLOW_INFERENCE_SEED_MODE="${FLOW_INFERENCE_SEED_MODE:-same_as_seed}"
FIXED_FLOW_INFERENCE_SEED="${FIXED_FLOW_INFERENCE_SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  log "ERROR: Set CHECKPOINT."
  exit 1
fi
if [[ -z "${OPENSCENE_ROOT:-${OPENSCENE_DATA_ROOT:-}}" ]]; then
  log "ERROR: Set OPENSCENE_ROOT or OPENSCENE_DATA_ROOT."
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  printf 'seed,flow_inference_seed,status,v1_metric,v1_score,v2_metric,v2_score,trajectory_cache,output_dir,started_at,finished_at\n' > "${SUMMARY_CSV}"
fi

seed_done() {
  local seed="$1"
  [[ "${SKIP_COMPLETED_SEEDS}" == "1" || "${SKIP_COMPLETED_SEEDS,,}" == "true" ]] || return 1
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${seed}" <<'PY'
import csv
import sys

summary_csv, seed = sys.argv[1:3]
try:
    with open(summary_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("seed") == seed and row.get("status") == "success":
                raise SystemExit(0)
except FileNotFoundError:
    pass
raise SystemExit(1)
PY
}

extract_score() {
  local output_dir="$1"
  "${PYTHON_BIN}" - "${output_dir}" <<'PY'
import math
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
csv_files = sorted(root.glob("*.csv"), key=lambda p: p.stat().st_mtime)
if not csv_files:
    print("")
    print("")
    raise SystemExit(0)

df = pd.read_csv(csv_files[-1])
if "token" in df.columns and (df["token"] == "average_all_frames").any():
    row = df.loc[df["token"] == "average_all_frames"].iloc[-1]
else:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    row = df[numeric_cols].mean(numeric_only=True)

for key in ("epdms", "score", "pdms", "pdm_score"):
    if key in row.index:
        value = row[key]
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            print(key)
            print(f"{value:.10f}")
            raise SystemExit(0)
print("")
print("")
PY
}

append_summary() {
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "$@" <<'PY'
import csv
import sys

summary_csv = sys.argv[1]
row = sys.argv[2:]
with open(summary_csv, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY
}

print_best() {
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" <<'PY'
import csv
import math
import sys

summary_csv = sys.argv[1]
best = None
with open(summary_csv, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("status") != "success":
            continue
        score_text = row.get("v2_score") or row.get("v1_score") or ""
        try:
            score = float(score_text)
        except ValueError:
            continue
        if math.isnan(score):
            continue
        if best is None or score > best[0]:
            best = (score, row)
if best is None:
    print("当前还没有成功结果")
else:
    score, row = best
    print(
        f"当前最佳 seed={row.get('seed')} flow_seed={row.get('flow_inference_seed')} "
        f"score={score:.8f} output={row.get('output_dir')}"
    )
PY
}

run_one_seed() {
  local seed="$1"
  local flow_seed="$seed"
  local seed_output_dir="${OUTPUT_ROOT%/}/seed_${seed}"
  local trajectory_cache="${seed_output_dir}/trajectory_cache/navtest_trajectories.pkl"
  local started_at
  local finished_at
  local status="failed"
  local rc=0
  local v1_metric=""
  local v1_score=""
  local v2_metric=""
  local v2_score=""

  if [[ "${FLOW_INFERENCE_SEED_MODE}" == "fixed" ]]; then
    flow_seed="${FIXED_FLOW_INFERENCE_SEED}"
  fi
  if seed_done "${seed}"; then
    log "跳过已成功 seed=${seed}"
    return 0
  fi

  mkdir -p "${seed_output_dir}"
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  log "开始 seed=${seed} model.flow_inference_seed=${flow_seed}"

  set +e
  OUTPUT_DIR="${seed_output_dir}" \
  TRAJECTORY_CACHE_PATH="${trajectory_cache}" \
  FORCE_EXPORT=1 \
  EXPORT_OVERRIDES="model.flow_inference_seed=${flow_seed}" \
  bash "${DUAL_SCRIPT}" 2>&1 | tee "${seed_output_dir}/run.log"
  rc=${PIPESTATUS[0]}
  set -e

  finished_at="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ "${rc}" -eq 0 ]]; then
    status="success"
    if [[ -d "${seed_output_dir}/v1_1" ]]; then
      mapfile -t parsed_v1 < <(extract_score "${seed_output_dir}/v1_1")
      v1_metric="${parsed_v1[0]:-}"
      v1_score="${parsed_v1[1]:-}"
    fi
    if [[ -d "${seed_output_dir}/v2" ]]; then
      mapfile -t parsed_v2 < <(extract_score "${seed_output_dir}/v2")
      v2_metric="${parsed_v2[0]:-}"
      v2_score="${parsed_v2[1]:-}"
    fi
    log "完成 seed=${seed} v1=${v1_metric}:${v1_score} v2=${v2_metric}:${v2_score}"
  else
    log "失败 seed=${seed} exit_code=${rc}"
  fi

  append_summary "${seed}" "${flow_seed}" "${status}" "${v1_metric}" "${v1_score}" "${v2_metric}" "${v2_score}" \
    "${trajectory_cache}" "${seed_output_dir}" "${started_at}" "${finished_at}"
  print_best

  if [[ "${status}" == "failed" && ("${STOP_ON_FAILURE}" == "1" || "${STOP_ON_FAILURE,,}" == "true") ]]; then
    return "${rc:-1}"
  fi
  return 0
}

run_count=0
if [[ -n "${SEED_LIST}" ]]; then
  for seed in ${SEED_LIST}; do
    run_one_seed "${seed}"
    run_count=$((run_count + 1))
    if [[ "${MAX_RUNS}" =~ ^[0-9]+$ ]] && (( MAX_RUNS > 0 && run_count >= MAX_RUNS )); then break; fi
  done
else
  if [[ -z "${SEED_END}" ]]; then
    log "ERROR: Set SEED_LIST or SEED_END."
    exit 1
  fi
  seed="${SEED_START}"
  while (( seed <= SEED_END )); do
    run_one_seed "${seed}"
    run_count=$((run_count + 1))
    if [[ "${MAX_RUNS}" =~ ^[0-9]+$ ]] && (( MAX_RUNS > 0 && run_count >= MAX_RUNS )); then break; fi
    seed=$((seed + SEED_STRIDE))
  done
fi

log "seed 搜索结束: ${SUMMARY_CSV}"
