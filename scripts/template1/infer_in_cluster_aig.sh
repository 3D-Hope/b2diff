#!/bin/bash
#SBATCH --job-name=infer_template1_aig
#SBATCH --partition=batch
#SBATCH --qos=neurips-2026
#SBATCH --constraint=zone-msp3
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

trap 'ERR_CODE=$?; echo "Error on line ${LINENO}. Exit code: ${ERR_CODE}" >&2; exit ${ERR_CODE}' ERR
trap 'echo "Job interrupted"; exit 130' INT

mkdir -p logs

echo "────────────────────────────────────────────────────────"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "────────────────────────────────────────────────────────"

# ── AIG configuration ─────────────────────────────────────────────────────────
#
# AIG is a single inference-time operating point applied to the best iADD
# checkpoint (stage97 of incremental_branch_lambda_2_fk_4particles).
# AIG_BETA controls annealing sharpness (1=diversity-biased, 4=reward-biased).
# Start with 2.0 as recommended by Jena et al. WACV 2025.
#
AIG_BETA=2.0
CHECKPOINT="outputs/incremental_branch_lambda_2_fk_4particles/stage97"
OUTPUT_DIR="outputs/template1_aig/stage0"
PROMPT_FILE="configs/prompt/template1_train.json"
NUM_IMAGES=1080
BATCH_SIZE=4   # reduce to 2 if OOM

# ── Conda setup ───────────────────────────────────────────────────────────────
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
    echo "Miniforge installed at ${CONDA_DIR}"
else
    echo "Miniforge already exists at ${CONDA_DIR}"
fi

if [[ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
else
    echo "conda.sh not found"; exit 1
fi

eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"

# ── Conda env ─────────────────────────────────────────────────────────────────
CONDA_ENV_NAME="b2"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    conda create -n "${CONDA_ENV_NAME}" python="3.10" -y
fi

conda activate "${CONDA_ENV_NAME}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

echo "Python: $(python --version)"
echo "pip:    $(pip --version)"

# ── Dependencies ──────────────────────────────────────────────────────────────
REPO_DIR="/home/pramish_paudel/codes/b2diff"
cd "${REPO_DIR}"

pip install uv
uv pip install -r requirements.txt || { echo "Dependency install failed"; exit 1; }
uv pip install scipy
pip uninstall setuptools -y
pip install setuptools==80.9.0
pip install opencv-python scikit-learn

# ── GPU check ─────────────────────────────────────────────────────────────────
echo "GPU info:"
nvidia-smi || echo "nvidia-smi unavailable"

# ── Skip if already done ──────────────────────────────────────────────────────
if [[ -f "${OUTPUT_DIR}/clip_rewards.json" ]]; then
    echo "clip_rewards.json already exists at ${OUTPUT_DIR} — skipping generation."
else
    echo "────────────────────────────────────────────────────────"
    echo "Running AIG inference  beta=${AIG_BETA}"
    echo "  Checkpoint : ${CHECKPOINT}"
    echo "  Output     : ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────────"

    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    python3 scripts/inference/inference_lora_aig.py \
        --checkpoint_path "${CHECKPOINT}" \
        --output_dir      "${OUTPUT_DIR}" \
        --prompt_file     "${PROMPT_FILE}" \
        --num_images      "${NUM_IMAGES}" \
        --batch_size      "${BATCH_SIZE}" \
        --num_inference_steps 20 \
        --guidance_scale  5.0 \
        --seed            42 \
        --aig_beta        "${AIG_BETA}"
fi

echo "Done at: $(date)"
