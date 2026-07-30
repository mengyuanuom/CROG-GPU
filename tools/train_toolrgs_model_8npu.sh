#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${1:-drog}"
CONFIG="config/OCID-VLG/${MODEL}.yaml"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/OCID-VLG}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EXP_NAME="${EXP_NAME:-${MODEL}_crog_protocol_8npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

case "${MODEL}" in
  crogoff|ggcnnclip|grconvnetclip|lgd|maplegrasp)
    WEIGHTS=(clip-rn50)
    ;;
  drog|drogoff)
    WEIGHTS=(clip-vit-b16 dinov2-vitb14-reg4)
    ;;
  graspmamba)
    WEIGHTS=(clip-rn50 mambavision-t)
    ;;
  *)
    echo "Unsupported model: ${MODEL}" >&2
    echo "Choose: crogoff drog drogoff ggcnnclip grconvnetclip graspmamba lgd maplegrasp" >&2
    exit 2
    ;;
esac

[[ -f "${CONFIG}" ]] || {
  echo "Model config not found: ${CONFIG}" >&2
  exit 2
}
[[ -d "${DATA_ROOT}" ]] || {
  echo "OCID-VLG dataset directory not found: ${DATA_ROOT}" >&2
  exit 2
}
[[ -f "${DATA_ROOT}/refer/multiple/train_expressions.json" ]] || {
  echo "OCID-VLG training expressions not found under: ${DATA_ROOT}" >&2
  exit 2
}

python3 tools/download_pretrained.py "${WEIGHTS[@]}"

echo "[launch] model: ${MODEL}"
echo "[launch] config: ${CONFIG}"
echo "[launch] protocol: CROG legacy"
echo "[launch] global batch size comes from YAML; processes: ${NPROC_PER_NODE}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_crog.py \
  --config "${CONFIG}" \
  --opts \
  DATA.root_path "${DATA_ROOT}" \
  TRAIN.exp_name "${EXP_NAME}"
