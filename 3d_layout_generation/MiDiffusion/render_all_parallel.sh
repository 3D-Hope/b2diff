#!/bin/bash
set -e

RENDER_SCRIPT="../ThreedFront/scripts/render_results.py"
FUTURE_MODEL="/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl"
LOG_DIR="render_logs"

mkdir -p "${LOG_DIR}"

pkl_files=(
"/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/150_inf/stage0/sample_00000/result.pkl"
"/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/150_inf/stage0/sample_00001/result.pkl"
"/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/150_inf/stage0/sample_00002/result.pkl"
"/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/150_inf/stage0/sample_00003/result.pkl"
)

pids=()
names=()
for pkl in "${pkl_files[@]}"; do
    name=$(basename "${pkl}" .pkl)
    names+=("${name}")
    echo "Launching render for: ${name}"
    python "${RENDER_SCRIPT}" \
        "${pkl}" \
        --no_texture \
        --path_to_pickled_3d_future_model "${FUTURE_MODEL}" \
        > "${LOG_DIR}/${name}.log" 2>&1 &
    pids+=($!)
done

echo "All ${#pids[@]} jobs launched. PIDs: ${pids[*]}"
echo "Waiting for all to finish..."

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[DONE]   ${names[$i]}"
    else
        echo "[FAILED] ${names[$i]} (exit code $?)"
        failed=$((failed + 1))
    fi
done

if [[ ${failed} -eq 0 ]]; then
    echo "All renders completed successfully."
else
    echo "${failed} render(s) failed. Check logs in ${LOG_DIR}/."
    exit 1
fi



