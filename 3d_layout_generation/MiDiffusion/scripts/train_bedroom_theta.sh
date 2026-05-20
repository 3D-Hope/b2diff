#!/usr/bin/env bash
# Train bedroom scalar-theta model (bedrooms_mixed_theta.yaml, sample_num_points=12).
#
# Local:
#   ./scripts/train_bedroom_theta.sh
#
# Resume (same experiment tag, weights must match config N=12):
#   WEIGHT_FILE=output/log/pretrain_bedroom_theta_n12/model_10000 \
#   CONTINUE_FROM_EPOCH=10000 \
#   ./scripts/train_bedroom_theta.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE="${CONFIG_FILE:-${MD}/config/bedrooms_mixed_theta.yaml}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-pretrain_bedroom_theta_n12}"
GPU="${GPU:-0}"
WITH_WANDB="${WITH_WANDB:-1}"

WEIGHT_FILE="${WEIGHT_FILE:-}"
CONTINUE_FROM_EPOCH="${CONTINUE_FROM_EPOCH:-0}"

cd "${MD}"
export PYTHONPATH=.

ARGS=(
  "${CONFIG_FILE}"
  --experiment_tag "${EXPERIMENT_TAG}"
  --gpu "${GPU}"
  --continue_from_epoch "${CONTINUE_FROM_EPOCH}"
)

if [[ "${WITH_WANDB}" == "1" ]]; then
  ARGS+=(--with_wandb_logger)
fi

if [[ -n "${WEIGHT_FILE}" ]]; then
  if [[ "${WEIGHT_FILE}" != /* ]]; then
    WEIGHT_FILE="${MD}/${WEIGHT_FILE}"
  fi
  ARGS+=(--weight_file "${WEIGHT_FILE}")
fi

echo "=== Train bedroom theta ==="
echo "CONFIG:          ${CONFIG_FILE}"
echo "EXPERIMENT_TAG:  ${EXPERIMENT_TAG}"
echo "GPU:             ${GPU}"
echo "CONTINUE_EPOCH:  ${CONTINUE_FROM_EPOCH}"
echo "WEIGHT_FILE:     ${WEIGHT_FILE:-<none>}"
echo

python scripts/ashok_train.py "${ARGS[@]}"
