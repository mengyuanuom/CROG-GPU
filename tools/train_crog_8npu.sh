#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/OCID-VLG}"
CLIP_WEIGHT="${CLIP_WEIGHT:-${REPO_ROOT}/pretrain/RN50.pt}"
CONFIG="${CONFIG:-config/OCID-VLG/crog_multiple_r50.yaml}"

[[ -d "${DATA_ROOT}" ]] || {
  echo "OCID-VLG dataset directory not found: ${DATA_ROOT}" >&2
  exit 2
}
[[ -f "${DATA_ROOT}/refer/multiple/train_expressions.json" ]] || {
  echo "OCID-VLG multiple-split training expressions not found under: ${DATA_ROOT}" >&2
  exit 2
}
# Download the official OpenAI checkpoint when absent and verify its SHA-256
# before every run. A partial/invalid file never reaches model construction.
python3 tools/download_clip_rn50.py --output "${CLIP_WEIGHT}"

echo "[launch] visible NPUs: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "[launch] torchrun processes on this node: ${NPROC_PER_NODE}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_crog.py \
  --config "${CONFIG}" \
  --opts \
  DATA.root_path "${DATA_ROOT}" \
  TRAIN.clip_pretrain "${CLIP_WEIGHT}"
