#!/bin/bash
#SBATCH --job-name=b2diff_test
#SBATCH --partition=batch
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Better error reporting including line number
trap 'ERR_CODE=$?; echo "❌ Error on line ${LINENO:-?}. Exit code: $ERR_CODE" >&2; exit $ERR_CODE' ERR
trap 'echo "🛑 Job interrupted"; exit 130' INT

# ============================================================================
# TEST CONFIGURATION - Minimal settings for quick testing
# ============================================================================

# Wandb configuration (optional for testing)
WANDB_ENTITY="${WANDB_ENTITY:-pramish-paudel-insait}"
WANDB_PROJECT="${WANDB_PROJECT:-b2diff-test}"
export WANDB_API_KEY="${WANDB_API_KEY}"  # Set this in your environment

# Project directory (adjust to your actual path)
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Conda installation directory (adjust to your scratch space)
USER="${USER:-pramish_paudel}"
CONDA_DIR="${CONDA_DIR:-/scratch/${USER}/tools/miniforge}"

# Conda environment name
CONDA_ENV_NAME="b2"

# Python version
DESIRED_PY="3.11"

# MINIMAL TEST CONFIGURATION - Quick run to verify system works
EXP_NAME="${EXP_NAME:-test_B2DiffuRL}"
SEED="${SEED:-42}"
STAGE_CNT="${STAGE_CNT:-3}"  # Only 3 stages for testing
SPLIT_STEP_LEFT="${SPLIT_STEP_LEFT:-14}"
SPLIT_STEP_RIGHT="${SPLIT_STEP_RIGHT:-20}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"  # Only 1 epoch per stage
BATCH_SIZE="${BATCH_SIZE:-4}"  # Smaller batch for faster testing
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"  # Smaller for faster testing
WANDB_ENABLED="${WANDB_ENABLED:-false}"  # Disable by default for testing

# ============================================================================
# Basic setup / logging
# ============================================================================
mkdir -p "${PROJECT_DIR}/logs"
cd "$PROJECT_DIR"

echo "══════════════════════════════════════════════════════════════════════════════"
echo "B²-DiffuRL Training Pipeline - TEST RUN"
echo "══════════════════════════════════════════════════════════════════════════════"
echo "⚠️  This is a MINIMAL TEST configuration - not for production!"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Project directory: $PROJECT_DIR"
echo "══════════════════════════════════════════════════════════════════════════════"
echo ""

echo "System information:"
echo "  OS: $(uname -s)"
echo "  Kernel: $(uname -r)"
echo "  Architecture: $(uname -m)"
echo "  Hostname: $(hostname)"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "  SLURM Job ID: $SLURM_JOB_ID"
    echo "  SLURM Job Name: ${SLURM_JOB_NAME:-N/A}"
    echo "  SLURM Node: ${SLURM_NODELIST:-N/A}"
fi
echo ""

# ============================================================================
# STAGE 1: Check for Conda installation
# ============================================================================
echo "──────────────────────────────────────────────────────────────────────────────"
echo "STAGE 1: Checking for Conda installation..."
echo "──────────────────────────────────────────────────────────────────────────────"

if [ ! -d "$CONDA_DIR" ]; then
    echo "❌ Conda directory not found at: $CONDA_DIR"
    echo "Please install Miniforge3 first or set CONDA_DIR to your conda installation."
    exit 1
fi

echo "✅ Conda directory found at: $CONDA_DIR"
echo ""

# ============================================================================
# STAGE 2: Initialize Conda
# ============================================================================
echo "──────────────────────────────────────────────────────────────────────────────"
echo "STAGE 2: Initializing Conda..."
echo "──────────────────────────────────────────────────────────────────────────────"

# Source conda.sh to make conda command available
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    echo "✅ Conda initialized from ${CONDA_DIR}/etc/profile.d/conda.sh"
elif [ -f "${CONDA_DIR}/condabin/conda" ]; then
    eval "$(${CONDA_DIR}/condabin/conda shell.bash hook)"
    echo "✅ Conda initialized using condabin/conda"
else
    echo "❌ Could not find conda initialization script"
    exit 1
fi

# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda command not found after initialization"
    exit 1
fi

CONDA_VERSION=$(conda --version 2>&1)
echo "  Conda version: $CONDA_VERSION"
echo ""

# ============================================================================
# STAGE 3: Check/Create environment
# ============================================================================
echo "──────────────────────────────────────────────────────────────────────────────"
echo "STAGE 3: Checking for Conda environment '$CONDA_ENV_NAME'..."
echo "──────────────────────────────────────────────────────────────────────────────"

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "✅ Environment '$CONDA_ENV_NAME' exists"
else
    echo "❌ Environment '$CONDA_ENV_NAME' not found"
    echo "Creating environment with Python $DESIRED_PY..."
    conda create -n "$CONDA_ENV_NAME" python="$DESIRED_PY" -y
    echo "✅ Environment created"
fi
echo ""

# ============================================================================
# STAGE 4: Activate environment
# ============================================================================
echo "──────────────────────────────────────────────────────────────────────────────"
echo "STAGE 4: Activating Conda environment '$CONDA_ENV_NAME'..."
echo "──────────────────────────────────────────────────────────────────────────────"

conda activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate environment '$CONDA_ENV_NAME'"
    exit 1
fi

ACTIVE_ENV=$(conda info --envs | grep '\*' | awk '{print $1}')
if [ "$ACTIVE_ENV" != "$CONDA_ENV_NAME" ]; then
    echo "❌ Expected environment '$CONDA_ENV_NAME' but got '$ACTIVE_ENV'"
    exit 1
fi

echo "✅ Environment activated: $CONDA_ENV_NAME"
PYTHON_VERSION=$(python --version 2>&1)
echo "  Python version: $PYTHON_VERSION"
echo "  Python path: $(which python)"
echo ""

# ============================================================================
# STAGE 5: Install dependencies
# ============================================================================
echo "──────────────────────────────────────────────────────────────────────────────"
echo "STAGE 5: Installing/Verifying dependencies..."
echo "──────────────────────────────────────────────────────────────────────────────"

if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -q -r "${PROJECT_DIR}/requirements.txt" --upgrade
    echo "✅ Dependencies installed"
else
    echo "⚠️  requirements.txt not found, skipping pip install"
fi
echo ""

# ============================================================================
# STAGE 6: Verify GPU availability
# ============================================================================
echo "──────────────────────────────────────────────────────────────────────────────"
echo "STAGE 6: Verifying GPU availability..."
echo "──────────────────────────────────────────────────────────────────────────────"

python -c "
import torch
import os

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CUDA device count: {torch.cuda.device_count()}')
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"Not set\")}')

if torch.cuda.is_available():
    print(f'\nDetected GPUs:')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1e9
        print(f'  GPU {i}: {props.name} ({mem_gb:.1f} GB)')
else:
    print('  ❌ No CUDA devices found!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GPU verification failed"
    exit 1
fi

echo "✅ GPU verification passed"
echo ""

# ============================================================================
# STAGE 7: Display test configuration
# ============================================================================
echo "══════════════════════════════════════════════════════════════════════════════"
echo "TEST CONFIGURATION - Minimal settings for quick verification"
echo "══════════════════════════════════════════════════════════════════════════════"
echo "⚠️  TEST MODE: Only ${STAGE_CNT} stages, ${TRAIN_EPOCHS} epoch per stage"
echo "Experiment name: $EXP_NAME"
echo "Random seed: $SEED"
echo "Total stages: $STAGE_CNT (vs 100 in production)"
echo "Split step range: [$SPLIT_STEP_LEFT, $SPLIT_STEP_RIGHT]"
echo "Training epochs per stage: $TRAIN_EPOCHS (vs 2 in production)"
echo "Batch size: $BATCH_SIZE (vs 8 in production)"
echo "Learning rate: $LEARNING_RATE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS (vs 8 in production)"
echo "Wandb enabled: $WANDB_ENABLED"
if [ "$WANDB_ENABLED" = "true" ]; then
    echo "  Wandb entity: $WANDB_ENTITY"
    echo "  Wandb project: $WANDB_PROJECT"
fi
echo ""
echo "Expected duration: ~10-20 minutes (vs ~3 days for full 100 stages)"
echo "══════════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# STAGE 8: Run the NEW training pipeline (TEST MODE)
# ============================================================================
echo "══════════════════════════════════════════════════════════════════════════════"
echo "STAGE 8: Starting B²-DiffuRL test run..."
echo "Test started at: $(date)"
echo "══════════════════════════════════════════════════════════════════════════════"
echo ""

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Navigate to scripts/training directory where train_pipeline.py is located
cd "${PROJECT_DIR}/scripts/training"

# Run the NEW Python-based training pipeline with MINIMAL TEST configuration
python train_pipeline.py \
    exp_name="${EXP_NAME}" \
    seed=${SEED} \
    dev_id=0 \
    pipeline.stage_cnt=${STAGE_CNT} \
    pipeline.continue_from_stage=0 \
    pipeline.split_step_left=${SPLIT_STEP_LEFT} \
    pipeline.split_step_right=${SPLIT_STEP_RIGHT} \
    train.num_epochs=${TRAIN_EPOCHS} \
    train.batch_size=${BATCH_SIZE} \
    train.learning_rate=${LEARNING_RATE} \
    train.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    wandb.entity="${WANDB_ENTITY}" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.enabled=${WANDB_ENABLED}

# Capture exit code
EXIT_CODE=$?

# ============================================================================
# Final status
# ============================================================================
cd "$PROJECT_DIR"
echo ""
echo "══════════════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test run completed successfully at: $(date)"
    echo ""
    echo "Generated outputs:"
    echo "  - Trained model checkpoints: model/lora/${EXP_NAME}/stage*/"
    echo "  - CLIP score plot: logs/${EXP_NAME}_clip_vs_queries.png"
    echo "  - Incremental plot: logs/${EXP_NAME}_clip_vs_queries_progress.png"
    echo "  - Metrics history: logs/${EXP_NAME}_metrics_history.json"
    echo "  - Timing info: logs/${EXP_NAME}_timing.txt"
    if [ "$WANDB_ENABLED" = "true" ]; then
        echo "  - Wandb dashboard: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
    fi
    echo ""
    echo "✅ System verification PASSED!"
    echo ""
    echo "Next steps:"
    echo "  1. Check the generated plots to verify CLIP score progression"
    echo "  2. Review logs to ensure no errors occurred"
    echo "  3. If test looks good, run full training with: sbatch scripts/run_b2diff_slurm.sh"
    echo ""
    echo "To run full production training (100 stages, 2 epochs each):"
    echo "  $ sbatch scripts/run_b2diff_slurm.sh"
    exit 0
else
    echo "❌ Test run failed with exit code $EXIT_CODE at: $(date)"
    echo ""
    echo "Check logs for details:"
    echo "  - SLURM output: logs/b2diff_test-${SLURM_JOB_ID}.out"
    echo "  - SLURM errors: logs/b2diff_test-${SLURM_JOB_ID}.err"
    echo ""
    echo "Common issues:"
    echo "  - GPU not available"
    echo "  - Missing dependencies (check requirements.txt)"
    echo "  - Incorrect paths (check PROJECT_DIR, CONDA_DIR)"
    echo "  - WANDB_API_KEY not set (if wandb.enabled=true)"
    exit $EXIT_CODE
fi
echo "══════════════════════════════════════════════════════════════════════════════"
