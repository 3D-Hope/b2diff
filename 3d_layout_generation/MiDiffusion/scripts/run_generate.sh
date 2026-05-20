#!/usr/bin/env bash
# Generate layouts with ashok_generate_results.py
#
# Usage:
#   # Custom output tag, weights from another training run, latest checkpoint:
#   RESULT_TAG=bedroom_theta_check \
#   EXPERIMENT_TAG=pretrain_bedroom_theta \
#   CHECKPOINT=last \
#   ./scripts/run_generate.sh
#
#   # Explicit checkpoint:
#   RESULT_TAG=bedroom_theta_check \
#   WEIGHT_FILE=output/log/pretrain_bedroom_theta/model_10600 \
#   ./scripts/run_generate.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULT_TAG="${RESULT_TAG:-${1:-bedroom_cos_sin_pretrained}}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-${RESULT_TAG}}"
CHECKPOINT="${CHECKPOINT:-best_model.pt}"

# Positional arg 2: experiment tag name, or path to checkpoint
if [[ -n "${2:-}" ]] && [[ -z "${WEIGHT_FILE:-}" ]]; then
  if [[ -f "${2}" ]] || [[ "${2}" == */* ]]; then
    WEIGHT_FILE="${2}"
    if [[ "${WEIGHT_FILE}" != /* ]]; then
      WEIGHT_FILE="${MD}/${WEIGHT_FILE}"
    fi
  else
    EXPERIMENT_TAG="${2}"
  fi
fi

if [[ -z "${WEIGHT_FILE:-}" ]]; then
  LOG_DIR="${MD}/output/log/${EXPERIMENT_TAG}"
  if [[ "${CHECKPOINT}" == "last" ]]; then
    WEIGHT_FILE="$(ls -1 "${LOG_DIR}"/model_* 2>/dev/null | sort -V | tail -1)"
    if [[ -z "${WEIGHT_FILE:-}" ]]; then
      echo "ERROR: no model_* checkpoints in ${LOG_DIR}" >&2
      exit 1
    fi
  else
    WEIGHT_FILE="${LOG_DIR}/${CHECKPOINT}"
  fi
fi

if [[ "${WEIGHT_FILE}" != /* ]]; then
  WEIGHT_FILE="${MD}/${WEIGHT_FILE}"
fi

CONFIG_FILE="${CONFIG_FILE:-}"
N_SYN_SCENES="${N_SYN_SCENES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GPU="${GPU:-0}"

RESULT_DIR="${MD}/output/predicted_results/${RESULT_TAG}"
PKL="${RESULT_DIR}/results.pkl"

if [[ ! -f "${WEIGHT_FILE}" ]]; then
  echo "ERROR: checkpoint not found: ${WEIGHT_FILE}" >&2
  echo "Set WEIGHT_FILE, or EXPERIMENT_TAG + CHECKPOINT (best_model.pt | last | model_10600)." >&2
  exit 1
fi

echo "=== Generation ==="
echo "RESULT_TAG:      ${RESULT_TAG}"
echo "EXPERIMENT_TAG:  ${EXPERIMENT_TAG}"
echo "WEIGHT_FILE:     ${WEIGHT_FILE}"
echo "CONFIG_FILE:     ${CONFIG_FILE:-<next to checkpoint>}"
echo "OUTPUT:          ${PKL}"
echo "N_SYN_SCENES:    ${N_SYN_SCENES}"
echo "BATCH_SIZE:      ${BATCH_SIZE}"
echo "GPU:             ${GPU}"
echo

cd "${MD}"
export PYTHONPATH=.

CONFIG_ARGS=()
if [[ -n "${CONFIG_FILE}" ]]; then
  if [[ "${CONFIG_FILE}" != /* ]]; then
    CONFIG_FILE="${MD}/${CONFIG_FILE}"
  fi
  CONFIG_ARGS=(--config_file "${CONFIG_FILE}")
fi

python scripts/ashok_generate_results.py \
  "${WEIGHT_FILE}" \
  "${CONFIG_ARGS[@]}" \
  --result_tag "${RESULT_TAG}" \
  --n_syn_scenes "${N_SYN_SCENES}" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${GPU}"

echo
echo "Done. Saved: ${PKL}"
