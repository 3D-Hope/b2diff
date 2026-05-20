#!/usr/bin/env bash
# Render top-down PNGs from results.pkl (no_texture, matches FID/SCA eval)
#
# Usage:
#   ./scripts/run_render.sh
#   RESULT_TAG=bedroom_cos_sin_pretrained ROOM_TYPE=bedroom ./scripts/run_render.sh
#   ./scripts/run_render.sh bedroom_cos_sin_pretrained bedroom
#
# Prerequisite: threed_future_model_<room>.pkl paths must point at local 3D-FUTURE meshes.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD="$(cd "${SCRIPT_DIR}/.." && pwd)"
TF="${TF:-${MD}/../ThreedFront}"

# --- defaults ---
RESULT_TAG="${RESULT_TAG:-${1:-bedroom_cos_sin_pretrained}}"
ROOM_TYPE="${ROOM_TYPE:-${2:-bedroom}}"   # bedroom | livingroom | diningroom | library

RESULT_DIR="${MD}/output/predicted_results/${RESULT_TAG}"
PKL="${RESULT_DIR}/results.pkl"
RENDER_DIR="${RESULT_DIR}/renders"
FUTURE_PKL="${TF}/output/threed_future_model_${ROOM_TYPE}.pkl"

if [[ ! -f "${PKL}" ]]; then
  echo "ERROR: results.pkl not found: ${PKL}" >&2
  echo "Run scripts/run_generate.sh first." >&2
  exit 1
fi
if [[ ! -f "${FUTURE_PKL}" ]]; then
  echo "ERROR: 3D-FUTURE pickle not found: ${FUTURE_PKL}" >&2
  exit 1
fi

mkdir -p "${RENDER_DIR}"

echo "=== Rendering ==="
echo "RESULT_TAG:  ${RESULT_TAG}"
echo "ROOM_TYPE:   ${ROOM_TYPE}"
echo "PKL:         ${PKL}"
echo "RENDER_DIR:  ${RENDER_DIR}"
echo "FUTURE_PKL:  ${FUTURE_PKL}"
echo

cd "${TF}"
python scripts/render_results.py \
  "${PKL}" \
  --path_to_pickled_3d_future_model "${FUTURE_PKL}" \
  --no_texture \
  --output_directory "${RENDER_DIR}"

echo
echo "Done. PNGs in: ${RENDER_DIR}"
