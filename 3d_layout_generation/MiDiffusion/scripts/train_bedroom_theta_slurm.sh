#!/bin/bash
#SBATCH --job-name=bedroom_theta_n12
#SBATCH --partition=batch
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --qos=neurips-2026
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# MiDiffusion — bedroom scalar-theta pretrain (sample_num_points=12).
#
# Submit from MiDiffusion/:
#   mkdir -p logs
#   sbatch scripts/train_bedroom_theta_slurm.sh
#
# Resume:
#   export WEIGHT_FILE=output/log/pretrain_bedroom_theta_n12/model_10000
#   export CONTINUE_FROM_EPOCH=10000
#   sbatch scripts/train_bedroom_theta_slurm.sh
#
# Override paths (cluster):
#   export CONDA_DIR=/scratch/$USER/tools/miniforge
#   export CONDA_ENV_NAME=midiffusion
#   export WANDB_ENTITY=your-entity

set -euo pipefail

# ------------------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------------------
trap 'ERR_CODE=$?; echo "Error on line ${LINENO}. Exit code: ${ERR_CODE}" >&2; exit ${ERR_CODE}' ERR
trap 'echo "Job interrupted"; exit 130' INT

# ------------------------------------------------------------------------------
# Paths (auto-detect repo; override via env on cluster)
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD="${MD:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${MD}/../.." && pwd)}"

CONFIG_FILE="${CONFIG_FILE:-${MD}/config/bedrooms_mixed_theta.yaml}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-pretrain_bedroom_theta_n12}"
GPU="${GPU:-0}"
WITH_WANDB="${WITH_WANDB:-1}"
WEIGHT_FILE="${WEIGHT_FILE:-}"
CONTINUE_FROM_EPOCH="${CONTINUE_FROM_EPOCH:-0}"

# Skip miniforge bootstrap if conda is already active (e.g. local dev)
SKIP_CONDA_SETUP="${SKIP_CONDA_SETUP:-0}"

CONDA_DIR="${CONDA_DIR:-/scratch/${USER}/tools/miniforge}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-midiffusion}"
DESIRED_PY="${DESIRED_PY:-3.10}"

export WANDB_ENTITY="${WANDB_ENTITY:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
mkdir -p "${MD}/logs"

echo "──────────────────────────────────────────────────────────────────────────────"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "MiDiffusion: ${MD}"
echo "Repo root:   ${REPO_ROOT}"
echo "Experiment:  ${EXPERIMENT_TAG}"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

# ------------------------------------------------------------------------------
# STAGE 1: Conda / Miniforge (cluster)
# ------------------------------------------------------------------------------
if [[ "${SKIP_CONDA_SETUP}" == "1" ]]; then
  echo "STAGE 1: SKIP_CONDA_SETUP=1 — using existing conda env"
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "Active env: ${CONDA_PREFIX}"
  fi
else
  echo "STAGE 1: Setting up Miniforge (if missing)..."

  if [[ ! -d "${CONDA_DIR}" ]]; then
    echo "Installing Miniforge to ${CONDA_DIR}..."
    mkdir -p "$(dirname "${CONDA_DIR}")"
    INSTALLER="Miniforge3-24.11.3-0-Linux-x86_64.sh"
    wget -q --show-progress \
      "https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/${INSTALLER}" \
      -O "/tmp/${INSTALLER}"
    bash "/tmp/${INSTALLER}" -b -p "${CONDA_DIR}"
    rm -f "/tmp/${INSTALLER}"
    echo "Miniforge installed at ${CONDA_DIR}"
  else
    echo "Miniforge already exists at ${CONDA_DIR}"
  fi

  if [[ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
  else
    echo "conda.sh not found under ${CONDA_DIR}"
    exit 1
  fi

  eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"

  echo "STAGE 2: Creating / activating conda env ${CONDA_ENV_NAME}"

  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    conda create -n "${CONDA_ENV_NAME}" python="${DESIRED_PY}" -y
  fi

  conda activate "${CONDA_ENV_NAME}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  hash -r
fi

echo "Environment verification:"
which python || true
python --version || true
which pip || true
echo ""

# ------------------------------------------------------------------------------
# STAGE 3: Dependencies
# ------------------------------------------------------------------------------
echo "STAGE 3: Installing Python dependencies"

if [[ -f "${REPO_ROOT}/requirements.txt" ]]; then
  pip install -q uv==0.9.26 || pip install uv==0.9.26
  uv pip install -r "${REPO_ROOT}/requirements.txt" || {
    echo "Dependency installation failed"
    exit 1
  }
else
  echo "No ${REPO_ROOT}/requirements.txt — assuming env already has deps"
fi

# MiDiffusion / eval stack pins (install if missing)
pip install -q diffusers==0.27.2 huggingface_hub==0.20.3 2>/dev/null || true

# ------------------------------------------------------------------------------
# STAGE 4: GPU check
# ------------------------------------------------------------------------------
echo "STAGE 4: GPU check"
nvidia-smi || echo "nvidia-smi unavailable"

# ------------------------------------------------------------------------------
# STAGE 5: Training
# ------------------------------------------------------------------------------
echo "STAGE 5: Starting MiDiffusion bedroom-theta training"

cd "${MD}"
export PYTHONPATH=.

START_TIME=$(date +%s)
START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
  NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
fi

echo "Training started at: ${START_TIME_READABLE}"
echo "GPUs detected: ${NUM_GPUS}"
echo "Config: ${CONFIG_FILE}"
echo "Tag:    ${EXPERIMENT_TAG}"
echo "Resume: epoch=${CONTINUE_FROM_EPOCH} weights=${WEIGHT_FILE:-none}"
echo ""

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

python scripts/ashok_train.py "${ARGS[@]}"

# ------------------------------------------------------------------------------
# Timing summary
# ------------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED_SECONDS=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(awk "BEGIN {printf \"%.4f\", ${ELAPSED_SECONDS}/3600}")
GPU_HOURS=$(awk "BEGIN {printf \"%.4f\", ${ELAPSED_HOURS}*${NUM_GPUS}}")

TIMING_LOG="${MD}/logs/${EXPERIMENT_TAG}_timing.txt"

{
  echo "Experiment: ${EXPERIMENT_TAG}"
  echo "Start time: ${START_TIME_READABLE}"
  echo "End time:   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "GPUs used:  ${NUM_GPUS}"
  echo "Wall time:  ${ELAPSED_SECONDS}s"
  echo "GPU hours:  ${GPU_HOURS}"
} > "${TIMING_LOG}"

echo "Training completed successfully"
echo "Timing log: ${TIMING_LOG}"
