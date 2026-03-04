#!/bin/bash
# eval_all_stages.sh
#
# Runs the full pipeline (inference → render → eval → parse) for every
# training stage of a given run and writes a metrics summary CSV.
#
# Usage:
#   bash eval_all_stages.sh <run_name> [start_stage=0] [end_stage=99]
#
# Example:
#   ! ddpo 0 49
#   bash eval_all_stages.sh 8_particles_incremental_branch_fk
#
# Prerequisites (paths can be overridden via env variables):
#   - conda env "b2" for inference and rendering
#   - steerable-scene-generation venv for evaluation (falls back to b2 if absent)
# --------------------------------------------------------------------------

set -euo pipefail

RUN_NAME="${1:?ERROR: provide run_name as first argument. Usage: $0 <run_name> [start=0] [end=99]}"
START_STAGE="${2:-0}"
END_STAGE="${3:-99}"

# ─── Fixed paths (override with env vars if needed) ─────────────────────────
B2DIFF_DIR="${B2DIFF_DIR:-/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff}"
MIDIFFUSION_DIR="${MIDIFFUSION_DIR:-${B2DIFF_DIR}/3d_layout_generation/MiDiffusion}"
THREEDFRONT_DIR="${THREEDFRONT_DIR:-/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront}"
MODEL_WEIGHT="${MODEL_WEIGHT:-${MIDIFFUSION_DIR}/output/log/pretrained_3d_layout_custom_attn/model_06000}"
LORA_BASE="${LORA_BASE:-${B2DIFF_DIR}/model/lora}"
PKL_3D_FUTURE="${PKL_3D_FUTURE:-${THREEDFRONT_DIR}/output/threed_future_model_bedroom.pkl}"
DATASET_DIR="${DATASET_DIR:-/mnt/sv-share/MiDiffusion/gravee/bedroom}"
PARSE_SCRIPT="${B2DIFF_DIR}/parse_eval_metrics.py"
CONDA_ENV="${CONDA_ENV:-b2}"

OUTPUT_BASE="${MIDIFFUSION_DIR}/output/full_predicted_results/${RUN_NAME}"
RESULTS_CSV="${OUTPUT_BASE}/metrics_table.csv"

# ─── Resolve Python interpreters ─────────────────────────────────────────────
EVAL_VENV_PYTHON="/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.venv/bin/python"



# ─── Setup ───────────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_BASE}"

# Write CSV header only when starting a fresh table
if [ ! -f "${RESULTS_CSV}" ]; then
    echo "stage,col_obj,col_scene,avg_num_obj,kl_div,mean_tv_bed_reward,scenes_with_multiple_tv_stands,scenes_with_multiple_beds,out_of_bound_rate" \
        > "${RESULTS_CSV}"
fi

echo ""
echo "========================================================"
echo " eval_all_stages.sh"
echo " run_name  : ${RUN_NAME}"
echo " stages    : ${START_STAGE} – ${END_STAGE}"
echo " output    : ${OUTPUT_BASE}"
echo " CSV       : ${RESULTS_CSV}"
echo "========================================================"

# ─── Main loop ───────────────────────────────────────────────────────────────
for i in $(seq "${START_STAGE}" "${END_STAGE}"); do

    LORA_WEIGHTS="${LORA_BASE}/${RUN_NAME}/stage${i}/checkpoints/checkpoint_1/lora_weights.pt"
    STAGE_DIR="${OUTPUT_BASE}/stage${i}"
    PKL_FILE="${STAGE_DIR}/results.pkl"
    EVAL_LOG="${STAGE_DIR}/eval.log"

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  Stage ${i}"
    echo "──────────────────────────────────────────────────────"

    # ── Skip if LoRA weights are missing ──────────────────────────────────
    if [ ! -f "${LORA_WEIGHTS}" ]; then
        echo "  [SKIP] LoRA weights not found: ${LORA_WEIGHTS}"
        continue
    fi

    # ── Skip if already fully evaluated (idempotency) ─────────────────────
    if [ -f "${EVAL_LOG}" ] && grep -q "Average accuracy" "${EVAL_LOG}" 2>/dev/null; then
        echo "  [SKIP] Already evaluated. Remove ${EVAL_LOG} to force re-run."
        if ! grep -q "^${i}," "${RESULTS_CSV}" 2>/dev/null; then
            METRICS=$(python3 "${PARSE_SCRIPT}" "${EVAL_LOG}")
            echo "${i},${METRICS}" >> "${RESULTS_CSV}"
            echo "  Re-added to CSV: ${METRICS}"
        fi
        continue
    fi


    # ── Step 1: Inference ───────────────────────────────────────────────
    echo ""
    echo "  [1/3] Inference..."

    # Delete the stage dir completely so ashok_generate_results.py sees an
    # absent (not just empty) directory and never hits the input() prompt.
    # Do NOT mkdir here — let the Python script create it.
    rm -rf "${STAGE_DIR}"

    # Tee inference output to a sibling temp log (parent dir already exists),
    # then move it into the stage dir once Python has created it.
    INFERENCE_TMP="${OUTPUT_BASE}/stage${i}_inference.tmp.log"

    conda run -n "${CONDA_ENV}" bash -c "
        set -e
        cd '${MIDIFFUSION_DIR}'
        PYTHONPATH=. python scripts/ashok_generate_results.py '${MODEL_WEIGHT}' \
            --output_directory '${MIDIFFUSION_DIR}/output/full_predicted_results/${RUN_NAME}' \
            --result_tag 'stage${i}' \
            --n_syn_scenes 1080 \
            --batch_size 512 \
            --lora '${LORA_WEIGHTS}'
    " 2>&1 | tee "${INFERENCE_TMP}"

    # Move log into the stage dir (which Python just created)
    [ -d "${STAGE_DIR}" ] && mv "${INFERENCE_TMP}" "${STAGE_DIR}/inference.log" \
                            || mv "${INFERENCE_TMP}" "${STAGE_DIR}.inference.log"

    if [ ! -f "${PKL_FILE}" ]; then
        echo "  [ERROR] Inference did not produce ${PKL_FILE}. Skipping stage."
        continue
    fi


    # ── Step 3: Evaluation ────────────────────────────────────────────────
    echo ""
    echo "  [3/3] Evaluation..."

    > "${EVAL_LOG}"  # create / clear log

    "${EVAL_VENV_PYTHON}" "${THREEDFRONT_DIR}/scripts/evaluate_kl_divergence_object_category.py" \
        "${PKL_FILE}" \
        2>&1 | tee -a "${EVAL_LOG}"

    echo "--- Bbox Analysis ---" | tee -a "${EVAL_LOG}"
    "${EVAL_VENV_PYTHON}" "${THREEDFRONT_DIR}/scripts/bbox_analysis.py" "${PKL_FILE}" \
        2>&1 | tee -a "${EVAL_LOG}"


    echo "--- Object Count ---" | tee -a "${EVAL_LOG}"
    "${EVAL_VENV_PYTHON}" "${THREEDFRONT_DIR}/scripts/calculate_num_obj.py" "${PKL_FILE}" \
        2>&1 | tee -a "${EVAL_LOG}"

    echo "--- Reward Evaluation ---" | tee -a "${EVAL_LOG}"
    "${EVAL_VENV_PYTHON}" "${THREEDFRONT_DIR}/scripts/evaluate_tv_bed_reward.py" "${PKL_FILE}" \
        2>&1 | tee -a "${EVAL_LOG}"



    # ── Step 4: Parse & record ─────────────────────────────────────────────
    echo ""
    echo "  Parsing metrics from ${EVAL_LOG}..."
    METRICS=$(python3 "${PARSE_SCRIPT}" "${EVAL_LOG}")
    echo "${i},${METRICS}" >> "${RESULTS_CSV}"
    echo "  stage=${i}: "
    echo "            ${METRICS}"

done

# ─── Final summary ────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Done!  Results: ${RESULTS_CSV}"
echo "========================================================"
echo ""
# Pretty-print if 'column' is available
if command -v column &>/dev/null; then
    column -t -s ',' "${RESULTS_CSV}"
else
    cat "${RESULTS_CSV}"
fi


BASE=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results
for run in ddpo 3d_b2_lora_16_test inc_b2 inc_b2_8_particles 8_particles_incremental_branch_fk 8_particles_incremental_fk 4_particles_incremental_branch_fk 4_particles_incremental_fk; do
    log="${BASE}/${run}/pipeline.log"
    if [ ! -f "$log" ]; then
        echo "[$run] NO LOG FILE"
    else
        last=$(tail -3 "$log")
        errors=$(grep -c "ERROR\|Traceback\|Error\|FAILED" "$log" 2>/dev/null || true)
        echo "=== $run (errors: $errors) ==="
        tail -3 "$log"
        echo ""
    fi
done