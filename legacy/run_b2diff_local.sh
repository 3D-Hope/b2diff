#!/bin/bash

set -euo pipefail

# Better error reporting including line number
trap 'ERR_CODE=$?; echo "❌ Error on line ${LINENO:-?}. Exit code: $ERR_CODE" >&2; exit $ERR_CODE' ERR
trap 'echo "🛑 Script interrupted"; exit 130' INT

# ============================================================================
# LOCAL TEST CONFIGURATION - Minimal settings for local machine testing
# ============================================================================

# Wandb configuration (optional for local testing)
WANDB_ENTITY="${WANDB_ENTITY:-pramish-paudel-insait}"
WANDB_PROJECT="${WANDB_PROJECT:-b2diff-local-test}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"  # Optional: set in environment if using wandb

# Project directory (auto-detect from script location)
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Conda installation directory (auto-detect common locations)
if [ -d "$HOME/miniforge3" ]; then
    CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_DIR="${CONDA_DIR:-$HOME/anaconda3}"
elif [ -d "/opt/conda" ]; then
    CONDA_DIR="${CONDA_DIR:-/opt/conda}"
else
    CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
fi

# Conda environment name (must already exist)
CONDA_ENV_NAME="${CONDA_ENV_NAME:-b2}"

# VERY MINIMAL LOCAL TEST CONFIGURATION - Quick run on local machine
EXP_NAME="${EXP_NAME:-local_test_B2DiffuRL}"
SEED="${SEED:-42}"
STAGE_CNT="${STAGE_CNT:-1}"  # Only 1 stage for ultra-quick local testing
SPLIT_STEP_LEFT="${SPLIT_STEP_LEFT:-14}"
SPLIT_STEP_RIGHT="${SPLIT_STEP_RIGHT:-20}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"  # Only 1 epoch per stage
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"  # Training batch size 1 for minimal memory
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"  # Sampling batch size 1 (RTX 2080 Ti = 11GB)
NUM_BATCHES="${NUM_BATCHES:-2}"  # Only 2 batches for ultra-fast testing
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"  # Minimal for faster local testing
WANDB_ENABLED="${WANDB_ENABLED:-false}"  # Disabled by default for local testing

# ============================================================================
# Basic setup / logging
# ============================================================================
mkdir -p "${PROJECT_DIR}/logs"
cd "$PROJECT_DIR"

echo "══════════════════════════════════════════════════════════════════════════════"
echo "B²-DiffuRL Training Pipeline - LOCAL TEST RUN"
echo "══════════════════════════════════════════════════════════════════════════════"
echo "⚠️  This is a MINIMAL LOCAL TEST - not for production!"
echo "Script started at: $(date)"
echo "Running on machine: $(hostname)"
echo "Working directory: $(pwd)"
echo "Project directory: $PROJECT_DIR"
echo "══════════════════════════════════════════════════════════════════════════════"
echo ""

echo "System information:"
echo "  OS: $(uname -s)"
echo "  Kernel: $(uname -r)"
echo "  Architecture: $(uname -m)"
echo "  Hostname: $(hostname)"
echo "  User: $(whoami)"
echo ""

# ============================================================================
# STAGE 1: Check for Conda installation
# ============================================================================
echo "──────────────────────────────────────────────────────────────────────────────"
echo "STAGE 1: Checking for Conda installation..."
echo "──────────────────────────────────────────────────────────────────────────────"

if [ ! -d "$CONDA_DIR" ]; then
    echo "❌ Conda directory not found at: $CONDA_DIR"
    echo ""
    echo "Common conda installation paths:"
    echo "  - $HOME/miniforge3"
    echo "  - $HOME/miniconda3"
    echo "  - $HOME/anaconda3"
    echo "  - /opt/conda"
    echo ""
    echo "Please install Miniforge3 or set CONDA_DIR to your conda installation:"
    echo "  $ export CONDA_DIR=/path/to/your/conda"
    echo "  $ bash scripts/run_b2diff_local.sh"
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
# STAGE 3: Activate existing environment
# ============================================================================
echo "──────────────────────────────────────────────────────────────────────────────"
echo "STAGE 3: Activating Conda environment '$CONDA_ENV_NAME'..."
echo "──────────────────────────────────────────────────────────────────────────────"

conda activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate environment '$CONDA_ENV_NAME'"
    echo ""
    echo "Please ensure the conda environment exists:"
    echo "  $ conda env list | grep $CONDA_ENV_NAME"
    echo ""
    echo "If not found, create it first or set CONDA_ENV_NAME to an existing environment."
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
# STAGE 4: Install/verify dependencies
# ============================================================================
echo "──────────────────────────────────────────────────────────────────────────────"
echo "STAGE 4: Installing/Verifying dependencies..."
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
# STAGE 5: Verify GPU availability (optional for local)
# ============================================================================
echo "──────────────────────────────────────────────────────────────────────────────"
echo "STAGE 5: Verifying GPU availability..."
echo "──────────────────────────────────────────────────────────────────────────────"

python -c "
import torch
import os

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"Not set\")}')
    print(f'\nDetected GPUs:')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1e9
        print(f'  GPU {i}: {props.name} ({mem_gb:.1f} GB)')
else:
    print('⚠️  No CUDA devices found - will run on CPU (VERY SLOW!)')
    print('For GPU acceleration, ensure:')
    print('  1. NVIDIA GPU is installed')
    print('  2. NVIDIA drivers are installed')
    print('  3. PyTorch with CUDA support is installed')
"

echo "✅ GPU check completed"
echo ""

# ============================================================================
# STAGE 6: Display local test configuration
# ============================================================================
echo "══════════════════════════════════════════════════════════════════════════════"
echo "LOCAL TEST CONFIGURATION - Minimal settings for quick verification"
echo "══════════════════════════════════════════════════════════════════════════════"
echo "⚠️  LOCAL TEST MODE: Only ${STAGE_CNT} stage, ${TRAIN_EPOCHS} epoch, ${NUM_BATCHES} batches"
echo "⚠️  RTX 2080 Ti (11GB) requires batch_size=1 due to Stable Diffusion memory needs"
echo "Experiment name: $EXP_NAME"
echo "Random seed: $SEED"
echo "Total stages: $STAGE_CNT (vs 100 in production)"
echo "Split step range: [$SPLIT_STEP_LEFT, $SPLIT_STEP_RIGHT]"
echo "Training epochs per stage: $TRAIN_EPOCHS (vs 2 in production)"
echo "Training batch size: $TRAIN_BATCH_SIZE (vs 8 in production)"
echo "Sampling batch size: $SAMPLE_BATCH_SIZE (vs 16 in production) - MINIMAL for RTX 2080 Ti"
echo "Num sampling batches: $NUM_BATCHES (vs 16 in production)"
echo "Learning rate: $LEARNING_RATE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS (vs 8 in production)"
echo "Wandb enabled: $WANDB_ENABLED"
if [ "$WANDB_ENABLED" = "true" ]; then
    echo "  Wandb entity: $WANDB_ENTITY"
    echo "  Wandb project: $WANDB_PROJECT"
fi
echo ""
echo "Expected duration: ~5-15 minutes (depending on GPU)"
echo "══════════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# STAGE 7: Run the NEW training pipeline (LOCAL TEST MODE)
# ============================================================================
echo "══════════════════════════════════════════════════════════════════════════════"
echo "STAGE 7: Starting B²-DiffuRL local test run..."
echo "Test started at: $(date)"
echo "══════════════════════════════════════════════════════════════════════════════"
echo ""

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Navigate to scripts/training directory where train_pipeline.py is located
cd "${PROJECT_DIR}/scripts/training"

# Run the NEW Python-based training pipeline with MINIMAL LOCAL TEST configuration
python train_pipeline.py \
    exp_name="${EXP_NAME}" \
    seed=${SEED} \
    dev_id=0 \
    pipeline.stage_cnt=${STAGE_CNT} \
    pipeline.continue_from_stage=0 \
    pipeline.split_step_left=${SPLIT_STEP_LEFT} \
    pipeline.split_step_right=${SPLIT_STEP_RIGHT} \
    train.num_epochs=${TRAIN_EPOCHS} \
    train.batch_size=${TRAIN_BATCH_SIZE} \
    train.learning_rate=${LEARNING_RATE} \
    train.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    sample.batch_size=${SAMPLE_BATCH_SIZE} \
    sample.num_batches_per_epoch=${NUM_BATCHES} \
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
    echo "✅ Local test run completed successfully at: $(date)"
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
    echo "✅ System verification PASSED on local machine!"
    echo ""
    echo "Next steps:"
    echo "  1. Check the generated plots: open logs/${EXP_NAME}_clip_vs_queries.png"
    echo "  2. Review logs to ensure no errors occurred"
    echo "  3. If test looks good, run cluster test: sbatch scripts/run_b2diff_test.sh"
    echo "  4. If cluster test passes, run full training: sbatch scripts/run_b2diff_slurm.sh"
    echo ""
    echo "Quick commands to view outputs:"
    echo "  $ ls -lh model/lora/${EXP_NAME}/  # View trained checkpoints"
    echo "  $ cat logs/${EXP_NAME}_timing.txt  # View timing information"
    echo "  $ python -m json.tool logs/${EXP_NAME}_metrics_history.json  # View metrics"
    exit 0
else
    echo "❌ Local test run failed with exit code $EXIT_CODE at: $(date)"
    echo ""
    echo "Common issues for local testing:"
    echo "  - GPU not available or out of memory (try reducing batch_size further)"
    echo "  - Missing dependencies: pip install -r requirements.txt"
    echo "  - Incorrect conda environment: conda activate ${CONDA_ENV_NAME}"
    echo "  - Path issues: check PROJECT_DIR and CONDA_DIR settings"
    echo "  - WANDB_API_KEY not set (if wandb.enabled=true)"
    echo ""
    echo "Debugging tips:"
    echo "  - Run with verbose output: bash -x scripts/run_b2diff_local.sh"
    echo "  - Check Python imports: python -c 'import torch; import diffusers; import open_clip'"
    echo "  - Check GPU: python -c 'import torch; print(torch.cuda.is_available())'"
    exit $EXIT_CODE
fi
echo "══════════════════════════════════════════════════════════════════════════════"
