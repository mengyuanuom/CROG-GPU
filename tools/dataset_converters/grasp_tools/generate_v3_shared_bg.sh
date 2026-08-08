#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Extra arguments may override these defaults.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" tools/dataset_converters/grasp_tools/augment.py \
  --src-dir assets/grasp_tools/graspall \
  --background-dir assets/grasp_tools/backgrounds \
  --out-dir datasets/grasp-tools/aug_graspall_v3_shared_bg \
  --background-policy shared \
  --train-scenes 24000 \
  --val-scenes 3000 \
  --test-scenes 3000 \
  --objects-min 2 \
  --objects-max 3 \
  --train-queries-per-scene 1 \
  --eval-queries-per-scene 1 \
  --max-query-difficulty 1 \
  --language-templates shared \
  --category-vocabulary expanded \
  --scales 0.5,0.7,0.9,1.1,1.3,1.5,1.8 \
  --angle-bins 36 \
  --same-category-probability 0.0 \
  --hard-negative-probability 0.0 \
  --placement-attempts 300 \
  --scene-attempts 40 \
  --grasp-height 20 \
  --brightness-jitter 0.05 \
  --contrast-jitter 0.05 \
  --saturation-jitter 0.05 \
  --feather-radius 0.8 \
  --seed 2025 \
  --image-ext jpg \
  --jpeg-quality 95 \
  --preview-count 30 \
  "$@"
