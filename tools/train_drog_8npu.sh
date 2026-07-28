#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG="${CONFIG:-config/OCID-VLG/drog.yaml}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/OCID-VLG}"
CLIP_WEIGHT="${CLIP_WEIGHT:-${REPO_ROOT}/pretrain/ViT-B-16.pt}"
DINO_WEIGHT="${DINO_WEIGHT:-${REPO_ROOT}/pretrain/dinov2_vitb14_reg4_pretrain.pth}"

for required in "${DATA_ROOT}" "${CLIP_WEIGHT}" "${DINO_WEIGHT}" "${CONFIG}"; do
  [[ -e "${required}" ]] || {
    echo "Required DROG artifact not found: ${required}" >&2
    exit 2
  }
done

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_crog.py \
  --config "${CONFIG}" \
  --opts \
  DATA.root_path "${DATA_ROOT}" \
  TRAIN.clip_pretrain "${CLIP_WEIGHT}" \
  TRAIN.dino_pretrain "${DINO_WEIGHT}"
