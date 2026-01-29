#!/bin/bash
#SBATCH --job-name=incremental_no_branch
#SBATCH --partition=batch
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ------------------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------------------
trap 'ERR_CODE=$?; echo "❌ Error on line ${LINENO}. Exit code: ${ERR_CODE}" >&2; exit ${ERR_CODE}' ERR
trap 'echo "🛑 Job interrupted"; exit 130' INT

# ------------------------------------------------------------------------------
# Basic setup / logging
# ------------------------------------------------------------------------------
mkdir -p logs

echo "──────────────────────────────────────────────────────────────────────────────"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

export WANDB_ENTITY="pramish-paudel-insait"

# ------------------------------------------------------------------------------
# STAGE 3: Miniforge / Conda
# ------------------------------------------------------------------------------
echo "STAGE 3: Setting up Miniforge (if missing)..."

CONDA_DIR="/scratch/pramish_paudel/tools/miniforge"

if [[ ! -d "${CONDA_DIR}" ]]; then
    echo "Installing Miniforge to ${CONDA_DIR}..."
    mkdir -p /scratch/pramish_paudel/tools
    cd /scratch/pramish_paudel/tools

    INSTALLER="Miniforge3-Linux-x86_64.sh"
    wget -q --show-progress \
        "https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}" \
        -O "${INSTALLER}"

    bash "${INSTALLER}" -b -p "${CONDA_DIR}"
    rm -f "${INSTALLER}"

    echo "✅ Miniforge installed at ${CONDA_DIR}"
else
    echo "✅ Miniforge already exists at ${CONDA_DIR}"
fi

# Source conda
if [[ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
else
    echo "❌ conda.sh not found"
    exit 1
fi

eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"

# ------------------------------------------------------------------------------
# STAGE 4: Conda env
# ------------------------------------------------------------------------------
echo "STAGE 4: Creating / activating conda env"

CONDA_ENV_NAME="b2"
DESIRED_PY="3.10"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    conda create -n "${CONDA_ENV_NAME}" python="${DESIRED_PY}" -y
fi

conda activate "${CONDA_ENV_NAME}"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

echo "Environment verification:"
which python
python --version
which pip
echo ""

# ------------------------------------------------------------------------------
# STAGE 5: pip install
# ------------------------------------------------------------------------------
echo "STAGE 5: Installing Python dependencies"

cd /home/pramish_paudel/codes/b2diff

pip install uv
uv pip install -r requirements.txt || {
    echo "❌ Dependency installation failed"
    exit 1
}

# ------------------------------------------------------------------------------
# STAGE 9: GPU check
# ------------------------------------------------------------------------------
echo "STAGE 9: GPU check"
nvidia-smi || echo "⚠️ nvidia-smi unavailable"

# ------------------------------------------------------------------------------
# STAGE 10: Training
# ------------------------------------------------------------------------------
echo "STAGE 10: Starting training"

export PYTHONUNBUFFERED=1
export DISPLAY=:0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

START_TIME=$(date +%s)
START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l || echo 1)
fi

echo "Training started at: ${START_TIME_READABLE}"
echo "GPUs detected: ${NUM_GPUS}"

run_name="incremental_no_branch"
python3 ./scripts/training/train_pipeline.py \
    exp_name="${run_name}" \
    train.incremental_training=true \
    sample.no_branching=true
# ------------------------------------------------------------------------------
# Timing summary
# ------------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED_SECONDS=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(awk "BEGIN {printf \"%.4f\", ${ELAPSED_SECONDS}/3600}")
GPU_HOURS=$(awk "BEGIN {printf \"%.4f\", ${ELAPSED_HOURS}*${NUM_GPUS}}")

TIMING_LOG="logs/${ExpName}_timing.txt"

{
echo "Experiment: ${ExpName}"
echo "Start time: ${START_TIME_READABLE}"
echo "End time:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPUs used:  ${NUM_GPUS}"
echo "Wall time:  ${ELAPSED_SECONDS}s"
echo "GPU hours:  ${GPU_HOURS}"
} > "${TIMING_LOG}"

echo "✅ Training completed successfully"
