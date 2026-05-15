#!/bin/bash
#SBATCH --job-name=infer_all_stages_all_norm_inc
#SBATCH --partition=batch
#SBATCH --constraint=zone-sof1
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

echo "=============================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=============================================================="

CONDA_DIR="/scratch/pramish_paudel/tools/miniforge"
if [[ ! -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]]; then
    echo "conda.sh not found at ${CONDA_DIR}. Please set up Miniforge first."
    exit 1
fi
source "${CONDA_DIR}/etc/profile.d/conda.sh"
eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"
conda activate b2

cd /home/pramish_paudel/codes/b2diff

export PYTHONUNBUFFERED=1
export DISPLAY=:0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_name="all_norm_inc"
model_root="model/lora/${run_name}"
output_root="outputs/${run_name}"

if [[ ! -d "${model_root}" ]]; then
    echo "Run folder not found: ${model_root}"
    exit 1
fi

if [[ -n "${STAGES_OVERRIDE:-}" ]]; then
    stages="${STAGES_OVERRIDE}"
    echo "Using STAGES_OVERRIDE: ${stages}"
else
    stages=$(ls "${model_root}" | sed -n 's/^stage\([0-9]\+\)$/\1/p' | sort -n)
fi

if [[ -z "${stages}" ]]; then
    echo "No stage folders found in ${model_root}"
    exit 1
fi

for stage_number in ${stages}; do
    checkpoint_dir="${model_root}/stage${stage_number}/checkpoints/checkpoint_1"
    stage_out="${output_root}/stage${stage_number}"
    score_file="${stage_out}/clip_rewards.json"

    if [[ ! -d "${checkpoint_dir}" ]]; then
        echo "Skipping stage${stage_number}: checkpoint not found"
        continue
    fi

    if [[ -f "${score_file}" ]]; then
        echo "Skipping stage${stage_number}: ${score_file} already exists"
        continue
    fi

    echo "--------------------------------------------------------------"
    echo "Running inference for ${run_name} stage${stage_number}"
    echo "--------------------------------------------------------------"
    python3 ./scripts/inference/inference_lora_clip_reward.py \
        --checkpoint_path "${checkpoint_dir}/" \
        --output_dir "./${stage_out}" \
        --num_images 1080 \
        --batch_size 32
done

echo "All available stages completed for ${run_name}"
