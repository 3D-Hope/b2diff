#!/bin/bash
#SBATCH --job-name=volprop2
#SBATCH --partition=batch
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --qos=neurips-2026
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

trap 'ERR=$?; echo "❌ Error on line ${LINENO}. Exit ${ERR}" >&2; exit ${ERR}' ERR

mkdir -p logs
echo "──────────────────────────────────────────────"
echo "Job started:  $(date)"
echo "Node:         $(hostname)"
echo "WD:           $(pwd)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "──────────────────────────────────────────────"

# Miniforge bootstrap (mirrors scripts/template2/iadd_grpo.sh) ----------------
CONDA_DIR="/scratch/pramish_paudel/tools/miniforge"
if [[ ! -d "${CONDA_DIR}" ]]; then
    echo "Installing Miniforge to ${CONDA_DIR}..."
    mkdir -p /scratch/pramish_paudel/tools
    cd /scratch/pramish_paudel/tools
    INSTALLER="Miniforge3-24.11.3-0-Linux-x86_64.sh"
    wget -q --show-progress \
        "https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/${INSTALLER}" \
        -O "${INSTALLER}"
    bash "${INSTALLER}" -b -p "${CONDA_DIR}"
    rm -f "${INSTALLER}"
fi
source "${CONDA_DIR}/etc/profile.d/conda.sh"
eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"

CONDA_ENV_NAME="b2"
DESIRED_PY="3.10"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    conda create -n "${CONDA_ENV_NAME}" python="${DESIRED_PY}" -y
fi
conda activate "${CONDA_ENV_NAME}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

cd /home/pramish_paudel/codes/b2diff
python --version
which pip

# Install deps if first time (idempotent; pip is fast when already satisfied)
pip install -q uv==0.9.26 || true
uv pip install -q -r requirements.txt || { echo "❌ deps"; exit 1; }
pip uninstall -y setuptools >/dev/null 2>&1 || true
pip install -q setuptools==80.9.0 opencv-python scikit-learn

echo "STAGE GPU"
nvidia-smi || true

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Run ----
ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=(--models ref,iadd,full
          --num_prompts 6 --seeds_per_prompt 4
          --num_probes 4 --num_terms 3
          --num_inference_steps 20 --guidance_scale 5.0
          --iadd_ckpt model/lora/only_10_ddpo/stage27/checkpoints/checkpoint_1
          --full_ckpt model/lora/vanilla_ddpo/stage27/checkpoints/checkpoint_1
          --output_dir rebuttal/artifacts/volume_preservation/run2_stage27)
fi

python scripts/diagnostics/volume_preservation.py "${ARGS[@]}"
echo "done=$(date)"
