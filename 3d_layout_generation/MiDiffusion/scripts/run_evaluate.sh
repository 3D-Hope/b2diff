#!/usr/bin/env bash
# Run KL, bbox, FID, KID (optional), and SCA metrics; log everything to evaluation.txt
#
# Usage:
#   ./scripts/run_evaluate.sh
#   RESULT_TAG=bedroom_cos_sin_pretrained ROOM_TYPE=bedroom ./scripts/run_evaluate.sh
#   COMPUTE_KID=1 ./scripts/run_evaluate.sh bedroom_cos_sin_pretrained bedroom
#
# Prerequisites:
#   - results.pkl (run_generate.sh)
#   - renders/*.png (run_render.sh)
#   - MiData/<room>/.../rendered_scene_notexture_256.png for real images (preprocess_data --no_texture)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD="$(cd "${SCRIPT_DIR}/.." && pwd)"
TF="${TF:-${MD}/../ThreedFront}"

MIDATA_ROOT="${MIDATA_ROOT:-/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData}"

# --- defaults ---
RESULT_TAG="${RESULT_TAG:-${1:-bedroom_cos_sin_pretrained}}"
ROOM_TYPE="${ROOM_TYPE:-${2:-bedroom}}"
COMPUTE_KID="${COMPUTE_KID:-0}"   # set to 1 to also compute KID

RESULT_DIR="${MD}/output/predicted_results/${RESULT_TAG}"
PKL="${RESULT_DIR}/results.pkl"
RENDER_DIR="${RESULT_DIR}/renders"
EVAL_LOG="${RESULT_DIR}/evaluation.txt"
KL_DIR="${RESULT_DIR}/kl_stats"
FID_DIR="${RESULT_DIR}/fid_tmps"
CLASSIFIER_DIR="${RESULT_DIR}/classifier_tmps"

case "${ROOM_TYPE}" in
  bedroom)   DATA_DIR="${MIDATA_ROOT}/bedroom" ;;
  livingroom) DATA_DIR="${MIDATA_ROOT}/livingroom" ;;
  diningroom) DATA_DIR="${MIDATA_ROOT}/diningroom" ;;
  library)   DATA_DIR="${MIDATA_ROOT}/library" ;;
  *)
    echo "ERROR: ROOM_TYPE must be bedroom|livingroom|diningroom|library (got: ${ROOM_TYPE})" >&2
    exit 1
    ;;
esac

if [[ ! -f "${PKL}" ]]; then
  echo "ERROR: results.pkl not found: ${PKL}" >&2
  exit 1
fi
if [[ ! -d "${RENDER_DIR}" ]] || [[ -z "$(find "${RENDER_DIR}" -maxdepth 1 -name '*.png' 2>/dev/null | head -1)" ]]; then
  echo "ERROR: no PNG renders in ${RENDER_DIR}" >&2
  echo "Run scripts/run_render.sh first." >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: MiData directory not found: ${DATA_DIR}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}" "${KL_DIR}" "${FID_DIR}" "${CLASSIFIER_DIR}"

log_section() {
  {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo "Started: $(date -Iseconds)"
    echo ""
  } | tee -a "${EVAL_LOG}"
}

run_cmd() {
  "$@" 2>&1 | tee -a "${EVAL_LOG}"
}

{
  echo "MiDiffusion evaluation log"
  echo "RESULT_TAG:  ${RESULT_TAG}"
  echo "ROOM_TYPE:   ${ROOM_TYPE}"
  echo "PKL:         ${PKL}"
  echo "RENDER_DIR:  ${RENDER_DIR}"
  echo "DATA_DIR:    ${DATA_DIR}"
  echo "Started:     $(date -Iseconds)"
} > "${EVAL_LOG}"

cd "${TF}"

log_section "KL divergence (object categories)"
run_cmd python scripts/evaluate_kl_divergence_object_category.py \
  "${PKL}" \
  --output_directory "${KL_DIR}"

log_section "BBox analysis (OOB / IoU)"
run_cmd python scripts/bbox_analysis.py "${PKL}"

log_section "FID"
run_cmd python scripts/compute_fid_scores.py \
  "${PKL}" \
  --output_directory "${FID_DIR}" \
  --no_texture \
  --dataset_directory "${DATA_DIR}" \
  --synthesized_directory "${RENDER_DIR}"

if [[ "${COMPUTE_KID}" == "1" ]]; then
  log_section "KID"
  run_cmd python scripts/compute_fid_scores.py \
    "${PKL}" \
    --compute_kid \
    --output_directory "${FID_DIR}" \
    --no_texture \
    --dataset_directory "${DATA_DIR}" \
    --synthesized_directory "${RENDER_DIR}"
fi

log_section "SCA (synthetic vs real classifier accuracy)"
run_cmd python scripts/synthetic_vs_real_classifier.py \
  "${PKL}" \
  --output_directory "${CLASSIFIER_DIR}" \
  --no_texture \
  --dataset_directory "${DATA_DIR}" \
  --synthesized_directory "${RENDER_DIR}"

{
  echo ""
  echo "Finished: $(date -Iseconds)"
  echo "Full log: ${EVAL_LOG}"
} | tee -a "${EVAL_LOG}"

echo "Done. Metrics written to: ${EVAL_LOG}"
