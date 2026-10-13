#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")}"
CONFIG="${CONFIG:-configs/sd35m_wb.yaml}"
MASTER_PORT="${MASTER_PORT:-29531}"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[launch] gpus=$CUDA_VISIBLE_DEVICES nproc=$NPROC config=$CONFIG port=$MASTER_PORT"
echo "[launch] override=${OVERRIDE:-<none>}"

ARGS=(--config "$CONFIG")
if [ -n "${OVERRIDE:-}" ]; then
    ARGS+=(--override "$OVERRIDE")
fi
if [ -n "${RESUME_FROM:-}" ]; then
    ARGS+=(--resume-from "$RESUME_FROM")
fi

exec torchrun \
    --nproc_per_node="$NPROC" \
    --nnodes=1 \
    --master_port="$MASTER_PORT" \
    scripts/train_sd35_wb.py "${ARGS[@]}"
