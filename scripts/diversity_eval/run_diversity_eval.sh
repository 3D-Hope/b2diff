#!/bin/bash
#SBATCH --job-name=diversity_eval
#SBATCH --partition=batch
#SBATCH --constraint=zone-sof1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ------------------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------------------
trap 'ERR_CODE=$?; echo "Error on line ${LINENO}. Exit code: ${ERR_CODE}" >&2; exit ${ERR_CODE}' ERR
trap 'echo "Job interrupted"; exit 130' INT

# ------------------------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------------------------
mkdir -p logs

echo "────────────────────────────────────────────────────────────────"
echo "Job started at:    $(date)"
echo "Running on node:   $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID:            ${SLURM_JOB_ID:-N/A}"
echo "────────────────────────────────────────────────────────────────"
echo ""

# ------------------------------------------------------------------------------
# STAGE 1: Miniforge / Conda
# ------------------------------------------------------------------------------
echo "STAGE 1: Setting up Miniforge..."

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

# ------------------------------------------------------------------------------
# STAGE 2: Conda env
# ------------------------------------------------------------------------------
echo "STAGE 2: Activating conda env"

CONDA_ENV_NAME="b2"
DESIRED_PY="3.10"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    conda create -n "${CONDA_ENV_NAME}" python="${DESIRED_PY}" -y
fi

conda activate "${CONDA_ENV_NAME}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

echo "Python: $(which python) — $(python --version)"

# ------------------------------------------------------------------------------
# STAGE 3: pip install
# ------------------------------------------------------------------------------
echo "STAGE 3: Installing Python dependencies"

cd /home/pramish_paudel/codes/b2diff

pip install uv
uv pip install -r requirements.txt || { echo "Dependency installation failed"; exit 1; }
uv pip install scipy lpips open_clip_torch
pip uninstall setuptools -y
pip install setuptools==80.9.0

# ------------------------------------------------------------------------------
# STAGE 4: GPU check
# ------------------------------------------------------------------------------
echo "STAGE 4: GPU check"
nvidia-smi || echo "nvidia-smi unavailable"

# ------------------------------------------------------------------------------
# STAGE 5: Diversity evaluation across methods + stages
# ------------------------------------------------------------------------------
echo "STAGE 5: Starting diversity evaluation"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START_TIME=$(date +%s)
START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')

# ── Configuration ─────────────────────────────────────────────────────────────
PROMPT_FILE="configs/prompt/template1_train.json"
OUTPUT_DIR="outputs/diversity_eval"
NUM_PER_PROMPT=24          # 24 images × 45 prompts = 1080 images per run
BASE_SEED=0                # MUST stay the same for all methods
NUM_STEPS=20
GUIDANCE_SCALE=5.0

# Map: "label" -> "run_name_in_model_lora_dir"
# Add / remove entries here as more experiments finish.
declare -A METHOD_RUN_MAP
METHOD_RUN_MAP["ddpo"]="vanilla_ddpo"
METHOD_RUN_MAP["b2diffurl"]="b2diffu_try2"
METHOD_RUN_MAP["iadd_full"]="incremental_branch_lambda_2_fk_4particles_new"
METHOD_RUN_MAP["dance_grpo"]="template1_dance_grpo"
# Uncomment once iADD+GRPO run is done:
# METHOD_RUN_MAP["iadd_plus_grpo"]="iadd_plus_grpo"

# Stages to evaluate (every 5th, up to 40)
EVAL_STAGES="0 5 10 15 20 25 30 35 40"

# ── Loop ──────────────────────────────────────────────────────────────────────
for method_label in "${!METHOD_RUN_MAP[@]}"; do
    run_name="${METHOD_RUN_MAP[$method_label]}"
    echo ""
    echo "================================================================"
    echo "  Method : ${method_label}  (run dir: model/lora/${run_name})"
    echo "================================================================"

    for stage_number in ${EVAL_STAGES}; do
        checkpoint_dir="model/lora/${run_name}/stage${stage_number}/checkpoints/checkpoint_1"

        if [[ ! -d "${checkpoint_dir}" ]]; then
            echo "  Skipping stage${stage_number}: ${checkpoint_dir} not found."
            continue
        fi

        result_file="${OUTPUT_DIR}/${method_label}/stage${stage_number}/diversity_results.json"
        if [[ -f "${result_file}" ]]; then
            echo "  stage${stage_number}: results already exist, skipping."
            continue
        fi

        echo "  Running stage${stage_number} ..."
        python3 scripts/diversity_eval/measure_diversity_spread.py \
            --checkpoint_path "${checkpoint_dir}" \
            --method_name     "${method_label}" \
            --stage           "${stage_number}" \
            --output_dir      "${OUTPUT_DIR}" \
            --prompt_file     "${PROMPT_FILE}" \
            --num_images_per_prompt "${NUM_PER_PROMPT}" \
            --base_seed       "${BASE_SEED}" \
            --num_steps       "${NUM_STEPS}" \
            --guidance_scale  "${GUIDANCE_SCALE}"

        echo "  stage${stage_number}: done."
    done
done

# ── Aggregate all results into a single JSON summary ─────────────────────────
echo ""
echo "Aggregating results..."
python3 - <<'PYEOF'
import os, json, glob

output_dir = "outputs/diversity_eval"
summary = []

for result_file in sorted(glob.glob(f"{output_dir}/*/*/diversity_results.json")):
    with open(result_file) as f:
        data = json.load(f)
    summary.append({
        "method":               data["method"],
        "stage":                data["stage"],
        "clip_score_mean":      data.get("clip_score_mean"),
        "clip_score_std":       data.get("clip_score_std"),
        "clip_image_diversity_mean": data.get("clip_image_diversity_mean"),
        "clip_image_diversity_std":  data.get("clip_image_diversity_std"),
        "intra_lpips_mean":     data.get("intra_lpips_mean"),
        "intra_lpips_std":      data.get("intra_lpips_std"),
    })

summary.sort(key=lambda r: (r["method"], r["stage"]))
out_path = os.path.join(output_dir, "summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

# Pretty-print table
print(f"\n{'Method':<20} {'Stage':>5} {'CLIP':>8} {'±':>6} {'CLIPdiv':>8} {'LPIPS':>8}")
print("-" * 65)
for r in summary:
    lpips_str = f"{r['intra_lpips_mean']:.4f}" if r["intra_lpips_mean"] is not None else "   N/A"
    print(f"{r['method']:<20} {r['stage']:>5} "
          f"{r['clip_score_mean']:>8.4f} {r['clip_score_std']:>6.4f} "
          f"{r['clip_image_diversity_mean']:>8.4f} "
          f"{lpips_str:>8}")

print(f"\nSummary saved to: {out_path}")
PYEOF

# ── Timing ────────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED_SECONDS=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(awk "BEGIN {printf \"%.4f\", ${ELAPSED_SECONDS}/3600}")

TIMING_LOG="logs/diversity_eval_timing.txt"
{
echo "Start time:  ${START_TIME_READABLE}"
echo "End time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "Wall time:   ${ELAPSED_SECONDS}s (${ELAPSED_HOURS}h)"
} > "${TIMING_LOG}"

echo ""
echo "Diversity evaluation completed successfully."
