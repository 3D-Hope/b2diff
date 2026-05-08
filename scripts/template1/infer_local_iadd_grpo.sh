#!/bin/bash
set -euo pipefail

trap 'echo "❌ Error on line ${LINENO}. Exit code: $?" >&2; exit 1' ERR

cd "$(dirname "$0")/../.."

echo "Working directory: $(pwd)"
echo "Started at: $(date)"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_name="template1_iadd_grpo"

# All stages (0-29) — comment out and use the line below to select specific ones
# STAGES=($(seq 0 29))
STAGES=(38)

echo "Evaluating stages: ${STAGES[*]}"

for stage_number in "${STAGES[@]}"; do
    checkpoint_dir="model/lora/${run_name}/stage${stage_number}/checkpoints/checkpoint_1/"

    if [ ! -d "$checkpoint_dir" ]; then
        echo "⚠️  Skipping stage${stage_number}: $checkpoint_dir not found."
        continue
    fi

    output_dir="./outputs/${run_name}/stage${stage_number}"

    if [ -f "${output_dir}/clip_rewards.json" ]; then
        echo "✅ stage${stage_number}: already done, skipping."
        continue
    fi

    echo "================================================================"
    echo "🚀 Running inference for stage ${stage_number}"
    echo "================================================================"

    python3 ./scripts/inference/inference_lora_clip_reward.py \
        --checkpoint_path "$checkpoint_dir" \
        --output_dir      "$output_dir" \
        --num_images      1080 \
        --batch_size      4
done

echo ""
echo "✅ Done at: $(date)"
