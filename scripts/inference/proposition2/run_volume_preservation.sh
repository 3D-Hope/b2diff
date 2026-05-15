#!/bin/bash
#SBATCH --job-name=volprop2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

echo "node=$(hostname)  job=${SLURM_JOB_ID:-N/A}  start=$(date)"

# Use the /work-shared b2 env (visible on hala + sof1 + this compute node)
PY=/work/pramish/tools/miniconda3/envs/b2/bin/python

cd /home/pramish_paudel/codes/b2diff
$PY --version
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"

# ---- Run ----
# Pass through any args, with sensible defaults if none supplied.
ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=(--models ref,iadd,full
          --num_prompts 32
          --seeds_per_prompt 2
          --num_probes 8
          --num_inference_steps 20
          --guidance_scale 5.0
          --output_dir rebuttal/artifacts/volume_preservation)
fi

$PY scripts/diagnostics/volume_preservation.py "${ARGS[@]}"

echo "done=$(date)"
