"""
Flask API server for image viewer.
Serves run listings and images from named run directories.
Run: python server.py
"""
import os
from flask import Flask, jsonify, send_file, abort
from flask_cors import CORS

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MOUT = os.path.join(_BASE, "3d_layout_generation", "MiDiffusion", "output")

# Named runs: display_name -> absolute directory path
NAMED_RUNS = {
    "pretrained": os.path.join(_MOUT, "predicted_results", "test_pretrained_6k"),
    "ddpo": os.path.join(_MOUT, "full_predicted_results", "ddpo_tv_bed", "stage92"),
    "b2":   os.path.join(_MOUT, "full_predicted_results", "b2_tv_bed", "stage76"),
    "ours": os.path.join(_MOUT, "full_predicted_results", "4_particles_incremental_branch_fk_tv_bed", "stage50"),
    "ours_collision": os.path.join(_MOUT, "predicted_results", "4_particles_incremental_branch_fk"),
}

app = Flask(__name__)
CORS(app)


@app.route("/api/runs")
def list_runs():
    return jsonify(list(NAMED_RUNS.keys()))


@app.route("/api/images/<run_name>")
def list_images(run_name):
    run_dir = NAMED_RUNS.get(run_name)
    if run_dir is None or not os.path.isdir(run_dir):
        abort(404)
    images = sorted([
        f for f in os.listdir(run_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ])
    return jsonify(images)


@app.route("/api/image/<run_name>/<filename>")
def serve_image(run_name, filename):
    run_dir = NAMED_RUNS.get(run_name)
    if run_dir is None:
        abort(404)
    # Prevent path traversal
    img_path = os.path.realpath(os.path.join(run_dir, filename))
    if not img_path.startswith(os.path.realpath(run_dir)):
        abort(403)
    if not os.path.isfile(img_path):
        abort(404)
    return send_file(img_path)


if __name__ == "__main__":
    for name, path in NAMED_RUNS.items():
        print(f"  {name:6s} -> {path}")
    app.run(host="0.0.0.0", port=5001, debug=False)
