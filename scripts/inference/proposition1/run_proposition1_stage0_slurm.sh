#!/bin/bash
#SBATCH --job-name=prop1_stage0_1080
#SBATCH --partition=batch
#SBATCH --constraint=zone-msp3
#SBATCH --qos=neurips-2026
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ============================================================================
# Error handling
# ============================================================================
trap 'ERR_CODE=$?; echo "❌ Error on line ${LINENO}. Exit code: ${ERR_CODE}" >&2; exit ${ERR_CODE}' ERR
trap 'echo "🛑 Job interrupted"; exit 130' INT

# ============================================================================
# Basic setup / logging
# ============================================================================
mkdir -p logs

echo "──────────────────────────────────────────────────────────────────────────────"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Task: Proposition 1 Validation (Stage 0, 1080 samples, Both Methods)"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

export WANDB_ENTITY="pramish-paudel-insait"

# ============================================================================
# STAGE 1: Miniforge Setup
# ============================================================================
echo "STAGE 1: Setting up Miniforge (if missing)..."

CONDA_DIR="/scratch/pramish_paudel/tools/miniforge"

if [[ ! -d "${CONDA_DIR}" ]]; then
    echo "Installing Miniforge to ${CONDA_DIR}..."
    mkdir -p /scratch/pramish_paudel/tools
    cd /scratch/pramish_paudel/tools

    INSTALLER="Miniforge3-25.11.0-1-Linux-x86_64.sh"
    wget -q --show-progress \
        "https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/${INSTALLER}" \
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

# ============================================================================
# STAGE 2: Conda Environment
# ============================================================================
echo "STAGE 2: Creating / activating conda env"

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

# ============================================================================
# STAGE 3: Dependencies
# ============================================================================
echo "STAGE 3: Installing Python dependencies"

cd /home/pramish_paudel/codes/b2diff

pip install uv
uv pip install -r requirements.txt || {
    echo "❌ Dependency installation failed"
    exit 1
}
uv pip install scipy
pip uninstall setuptools -y
pip install setuptools==80.9.0

# ============================================================================
# STAGE 4: GPU Check
# ============================================================================
echo "STAGE 4: GPU check"
nvidia-smi || echo "⚠️ nvidia-smi unavailable"

# ============================================================================
# STAGE 5: Proposition 1 Experiment Pipeline
# ============================================================================
echo "STAGE 5: Starting Proposition 1 Pipeline (Stage 0, 1080 samples)"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START_TIME=$(date +%s)
START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l || echo 1)
fi

echo "Experiment started at: ${START_TIME_READABLE}"
echo "GPUs detected: ${NUM_GPUS}"
echo ""

# ============================================================================
# STEP 1: Generate 1080-sample pretrained baseline
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Generate 1080-sample Pretrained Baseline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BASELINE_DIR="outputs/baseline_sd14_samples_1080_10steps"
BASELINE_FILE="${BASELINE_DIR}/pretrained_latents.pt"

if [[ ! -f "${BASELINE_FILE}" ]]; then
    echo "Generating baseline with 1080 samples..."
    python3 ./scripts/inference/generate_pretrained_baseline_1080.py || {
        echo "❌ Baseline generation failed"
        exit 1
    }
    echo "✅ Baseline generated"
else
    echo "✅ Baseline already exists at ${BASELINE_FILE}, skipping generation"
fi

echo ""

# ============================================================================
# STEP 2: Run inference for uniform_10/stage0 using the 1080-sample baseline
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Inference - uniform_10/stage0 (1080 samples)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

export EXPERIMENT_NAME="uniform_10_stage0_1080"
export LORA_CHECKPOINT_PATH="./model/lora/uniform_10/stage0/checkpoints/checkpoint_1"
export OUTPUT_DIR="outputs/proposition1_stage0_1080/uniform_10"
export PRETRAINED_LATENTS_PATH="${BASELINE_FILE}"

mkdir -p "${OUTPUT_DIR}"

echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "LORA_CHECKPOINT_PATH: ${LORA_CHECKPOINT_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "PRETRAINED_LATENTS_PATH: ${PRETRAINED_LATENTS_PATH}"
echo ""

python3 ./scripts/inference/inference_lora_for_proposal.py || {
    echo "❌ uniform_10 inference failed"
    exit 1
}

echo "✅ uniform_10/stage0 inference complete"
echo ""

# ============================================================================
# STEP 3: Run inference for last_10/stage0 using the same 1080-sample baseline
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Inference - last_10/stage0 (1080 samples, same baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

export EXPERIMENT_NAME="last_10_stage0_1080"
export LORA_CHECKPOINT_PATH="./model/lora/last_10/stage0/checkpoints/checkpoint_1"
export OUTPUT_DIR="outputs/proposition1_stage0_1080/last_10"
export PRETRAINED_LATENTS_PATH="${BASELINE_FILE}"

mkdir -p "${OUTPUT_DIR}"

echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "LORA_CHECKPOINT_PATH: ${LORA_CHECKPOINT_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "PRETRAINED_LATENTS_PATH: ${PRETRAINED_LATENTS_PATH}"
echo ""

python3 ./scripts/inference/inference_lora_for_proposal.py || {
    echo "❌ last_10 inference failed"
    exit 1
}

echo "✅ last_10/stage0 inference complete"
echo ""

# ============================================================================
# STEP 4: Perturbation Geometry Analysis
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Perturbation Geometry Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Computing:"
echo "  - Per-dimension variance"
echo "  - Effective dimensionality"
echo "  - Anisotropy ratio"
echo "  - Per-channel analysis"
echo "  - PCA cumulative variance"
echo "  - Participation ratio"
echo ""

python3 ./scripts/inference/analyze_perturbation_geometry.py || {
    echo "⚠️ Analysis completed with warnings (non-blocking)"
}

echo "✅ Geometry analysis complete"
echo ""

# ============================================================================
# STEP 5: Timing Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED_SECONDS=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(awk "BEGIN {printf \"%.4f\", ${ELAPSED_SECONDS}/3600}")
GPU_HOURS=$(awk "BEGIN {printf \"%.4f\", ${ELAPSED_HOURS}*${NUM_GPUS}}")

TIMING_LOG="logs/prop1_stage0_1080_timing.txt"

{
    echo "Experiment: Proposition 1 Stage 0 (1080 samples)"
    echo "Start time: ${START_TIME_READABLE}"
    echo "End time:   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "GPUs used:  ${NUM_GPUS}"
    echo "Wall time:  ${ELAPSED_SECONDS}s (${ELAPSED_HOURS}h)"
    echo "GPU hours:  ${GPU_HOURS}"
    echo ""
    echo "Output locations:"
    echo "  Baseline:  ${BASELINE_FILE}"
    echo "  uniform_10: outputs/proposition1_stage0_1080/uniform_10/lora_latents.pt"
    echo "  last_10:    outputs/proposition1_stage0_1080/last_10/lora_latents.pt"
} > "${TIMING_LOG}"

echo "⏱️  Timing Summary"
echo "  Wall time:  ${ELAPSED_SECONDS}s (${ELAPSED_HOURS}h)"
echo "  GPU hours:  ${GPU_HOURS}"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ PROPOSITION 1 STAGE 0 PIPELINE COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Output Locations:"
echo "  Baseline:             outputs/baseline_sd14_samples_1080_10steps/pretrained_latents.pt"
echo "  uniform_10 latents:   outputs/proposition1_stage0_1080/uniform_10/lora_latents.pt"
echo "  last_10 latents:      outputs/proposition1_stage0_1080/last_10/lora_latents.pt"
echo ""
echo "🔬 Geometry Analysis:"
echo "  Summary metrics:      outputs/proposition1_geometry_analysis/geometry_summary.csv"
echo "  PCA variance:         outputs/proposition1_geometry_analysis/pca_variance.csv"
echo "  Plots (5 PNG):        outputs/proposition1_geometry_analysis/[01-05]_*.png"
echo ""
echo "📈 Next Steps:"
echo "  1. Review geometry analysis CSV files"
echo "  2. Check generated plots: dimension variance, PCA, channel analysis, metrics"
echo "  3. Run full proposal1.py for L2 distance comparison:"
echo "     python3 scripts/inference/proposal1.py"
echo "  4. If stage0 validates, scale to all stages (0-5)"
echo "════════════════════════════════════════════════════════════════════════════════"
