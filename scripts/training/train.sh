#!/usr/bin/env bash
set -eu -o pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[ERROR] This script requires bash." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHONPATH_PREFIX="${REPO_ROOT}"
if [[ -n "${NAVSIM_DEVKIT_ROOT:-}" ]]; then
  PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX}:${NAVSIM_DEVKIT_ROOT}"
fi
export PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-63669}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER:-worldmodel}}"
mkdir -p "${TRITON_CACHE_DIR}"

if command -v nvidia-smi &>/dev/null; then
  CUDA_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
else
  CUDA_DEVICES=""
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ -n "${CUDA_DEVICES}" && "${NPROC_PER_NODE}" -eq 1 ]]; then
  gpu_count="$(echo "${CUDA_DEVICES}" | tr "," "\n" | grep -c . || true)"
  [[ "${gpu_count}" -ge 1 ]] && NPROC_PER_NODE="${gpu_count}"
fi

NUM_NODES="${NUM_NODES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
PYTHON="${PYTHON:-$(command -v python3)}"
WM_CONFIG="${WM_CONFIG:-configs/basic/base_training.yaml}"

while [[ $# -gt 0 && "$1" == "--config" ]]; do
  if [[ -z "${2:-}" ]]; then
    echo "[ERROR] --config requires a YAML path." >&2
    exit 1
  fi
  WM_CONFIG="$2"
  shift 2
done

export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/ds_zero2_bf16.json}"

echo "====== train.sh ======"
echo "  config:    ${WM_CONFIG}"
echo "  nodes:     ${NUM_NODES} rank=${MACHINE_RANK} gpus/node=${NPROC_PER_NODE}"
echo "  master:    ${MASTER_ADDR}:${MASTER_PORT}"
echo "======================"

"${PYTHON}" -m torch.distributed.run \
  --nnodes="${NUM_NODES}" \
  --node_rank="${MACHINE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  training/train_world_model.py \
  --config "${WM_CONFIG}" \
  "$@"
