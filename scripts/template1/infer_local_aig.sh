#!/bin/bash
set -euo pipefail

trap 'echo "Error on line ${LINENO}. Exit code: $?" >&2; exit 1' ERR

cd "$(dirname "$0")/../.."

echo "Working directory: $(pwd)"
echo "Started at: $(date)"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# AIG runs on a single fixed checkpoint (best locally-available iADD stage).
# stage74 is the highest checkpoint on this machine (stage97 only on cluster).
# stage74 is also the paper's CLIP operating-point stage for iADD — a good choice.
CHECKPOINT="model/lora/incremental_branch_lambda_2_fk_4particles/stage74/checkpoints/checkpoint_1"
OUTPUT_DIR="outputs/template1_aig/stage0"

# AIG beta: annealing exponent (1=diversity-biased, 4=reward-biased; paper default=2)
AIG_BETA=2.0

# Reduce to 2 if OOM (AIG holds two UNets in VRAM simultaneously)
BATCH_SIZE=4

if [ -f "${OUTPUT_DIR}/clip_rewards.json" ]; then
    echo "Already done — ${OUTPUT_DIR}/clip_rewards.json exists, skipping."
    exit 0
fi

echo "================================================================"
echo "Running AIG inference  beta=${AIG_BETA}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Output     : ${OUTPUT_DIR}"
echo "================================================================"

python3 ./scripts/inference/inference_lora_aig.py \
    --checkpoint_path "${CHECKPOINT}" \
    --output_dir      "${OUTPUT_DIR}" \
    --prompt_file     "configs/prompt/template1_train.json" \
    --num_images      1080 \
    --batch_size      "${BATCH_SIZE}" \
    --num_inference_steps 20 \
    --guidance_scale  5.0 \
    --seed            42 \
    --aig_beta        "${AIG_BETA}"

echo ""
echo "Done at: $(date)"
