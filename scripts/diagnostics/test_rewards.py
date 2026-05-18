"""Test + visualise the new physics rewards.

Runs a handful of bedroom scenes through the new
``compute_non_penetration_reward`` (OBB + tuck-under) and
``compute_oob_reward_sdf`` (SDF-based smooth OOB), prints scores, and writes
top-down 2D layout images with the reward values printed on top.

Usage:
    python scripts/diagnostics/test_rewards.py \
        --data_root /home/pramish_paudel/codes/3dhope_data \
        --out_dir   scripts/diagnostics/out

Each test case is one of:
  * "real_scene"           - a real bedroom scene from 3D-FRONT
                              (expected: high reward on both)
  * "chair_under_desk"     - synthetic chair tucked under a desk
                              (expected: tuck-under handled → near 1.0)
  * "chair_on_desk"        - synthetic chair placed on top of a desk
                              (expected: penalty)
  * "beds_overlapping"     - two double beds overlapping
                              (expected: heavy penalty)
  * "bed_outside_floor"    - bed pushed half outside the room polygon
                              (expected: heavy OOB penalty)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

# Make the repo importable when run from scripts/diagnostics.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.universal_rewards.penetration_reward import (  # noqa: E402
    compute_non_penetration_reward,
    BEDROOM_LABELS,
)
from core.universal_rewards.not_out_of_bound_reward import (  # noqa: E402
    compute_oob_reward_sdf,
    SDFCache,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NUM_CLASSES_BEDROOM = 22
LABEL_TO_IDX = {v: k for k, v in BEDROOM_LABELS.items()}


def make_object(label: str, pos_xyz, half_size_xyz, angle_rad: float = 0.0):
    """Build a (1, 30)-vector entry: [pos(3), half_size(3), cos, sin, one_hot(22)]."""
    one_hot = np.zeros(NUM_CLASSES_BEDROOM, dtype=np.float32)
    one_hot[LABEL_TO_IDX[label]] = 1.0
    return np.concatenate([
        np.asarray(pos_xyz, dtype=np.float32),
        np.asarray(half_size_xyz, dtype=np.float32),
        np.asarray([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32),
        one_hot,
    ])


def pad_scene(objects: List[np.ndarray], max_n: int = 22) -> np.ndarray:
    """Pad with empty rows so every scene has the same N."""
    out = np.zeros((max_n, 8 + NUM_CLASSES_BEDROOM), dtype=np.float32)
    empty_oh = np.zeros(NUM_CLASSES_BEDROOM, dtype=np.float32)
    empty_oh[LABEL_TO_IDX["empty"]] = 1.0
    for i in range(max_n):
        if i < len(objects):
            out[i] = objects[i]
        else:
            out[i, 6] = 1.0  # cos
            out[i, 7] = 0.0  # sin
            out[i, 8:] = empty_oh
    return out


def parse_local(scenes: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Local parse that bypasses descaling (data is already in world coords)."""
    positions = scenes[:, :, 0:3]
    sizes = scenes[:, :, 3:6]
    orientations = scenes[:, :, 6:8]
    one_hot = scenes[:, :, 8:8 + NUM_CLASSES_BEDROOM]
    obj_idx = torch.argmax(one_hot, dim=-1)
    empty_idx = NUM_CLASSES_BEDROOM - 1
    is_empty = obj_idx == empty_idx
    return {
        "one_hot": one_hot,
        "positions": positions,
        "sizes": sizes,
        "orientations": orientations,
        "object_indices": obj_idx,
        "is_empty": is_empty,
        "device": scenes.device,
    }


# --------------------------------------------------------------------------- #
# Real-scene loader
# --------------------------------------------------------------------------- #


def load_real_bedroom(data_root: Path, idx_hint: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    bedroom_root = data_root / "bedroom"
    scene_dirs = sorted(p for p in bedroom_root.iterdir() if p.is_dir())
    if not scene_dirs:
        raise FileNotFoundError(f"No bedroom scenes under {bedroom_root}")
    npz = np.load(scene_dirs[idx_hint] / "boxes.npz", allow_pickle=True)
    cls_idx = npz["class_labels"].argmax(-1)
    trans = npz["translations"]
    sizes = npz["sizes"]
    angles = npz["angles"].squeeze(-1)
    floor = npz["floor_plan_ordered_corners"].astype(np.float64)

    objs = []
    for i in range(len(cls_idx)):
        label = BEDROOM_LABELS.get(int(cls_idx[i]), "empty")
        if label == "empty":
            continue
        objs.append(make_object(label,
                                trans[i].tolist(),
                                sizes[i].tolist(),
                                float(angles[i])))
    return pad_scene(objs), floor


# --------------------------------------------------------------------------- #
# Synthetic scenes
# --------------------------------------------------------------------------- #


def synth_rect_floor(half_x: float = 2.0, half_z: float = 2.0) -> np.ndarray:
    return np.array(
        [[-half_x, -half_z], [-half_x, half_z], [half_x, half_z], [half_x, -half_z]],
        dtype=np.float64,
    )


def scene_chair_under_desk() -> Tuple[np.ndarray, np.ndarray]:
    floor = synth_rect_floor()
    # Desk: 1.2 m wide, 0.6 m deep, 0.75 m tall, centered at origin.
    # Center y = 0.375 (half-height). half size = (0.6, 0.375, 0.3).
    desk = make_object("desk", (0.0, 0.375, 0.0), (0.6, 0.375, 0.3))
    # Chair: 0.5 m wide/deep, 0.9 m tall, center y = 0.45.
    # Chair top = 0.9, desk top = 0.75 — but with tuck-under slab the desk
    # only collides on [0.65, 0.75], so chair top is OK.
    # Wait: chair_top=0.9 > 0.65 → there would be y-overlap.
    # Use a typical low chair under desk: total height 0.65 (chair top below desk surface).
    chair = make_object("chair", (0.0, 0.30, 0.0), (0.22, 0.30, 0.22))
    bed = make_object("double_bed", (-1.2, 0.3, -1.0), (0.5, 0.3, 0.9))
    return pad_scene([desk, chair, bed]), floor


def scene_chair_on_desk() -> Tuple[np.ndarray, np.ndarray]:
    """Chair perched on top of desk (touching, not interpenetrating).

    Expected: r_pen ≈ 1.0. This validates that the tuck-under exception does
    NOT spuriously fail when the chair sits on top with no overlap.
    """
    floor = synth_rect_floor()
    desk = make_object("desk", (0.0, 0.375, 0.0), (0.6, 0.375, 0.3))
    chair = make_object("chair", (0.0, 1.05, 0.0), (0.22, 0.30, 0.22))
    return pad_scene([desk, chair]), floor


def scene_chair_clipping_desk() -> Tuple[np.ndarray, np.ndarray]:
    """Chair sunken into the desk top slab.

    Expected: r_pen < 1.0 — the chair's vertical extent overlaps the desk's
    top slab so the tuck-under exception does not save it.
    """
    floor = synth_rect_floor()
    desk = make_object("desk", (0.0, 0.375, 0.0), (0.6, 0.375, 0.3))
    # Chair top at 0.85, desk top at 0.75, top-slab [0.65, 0.75].
    # Chair bottom 0.25 is below the slab, top 0.85 above — overlap of 0.10 m.
    chair = make_object("chair", (0.0, 0.55, 0.0), (0.22, 0.30, 0.22))
    return pad_scene([desk, chair]), floor


def scene_beds_overlapping() -> Tuple[np.ndarray, np.ndarray]:
    floor = synth_rect_floor(2.5, 2.5)
    bed1 = make_object("double_bed", (0.0, 0.3, 0.0), (1.0, 0.3, 1.0))
    bed2 = make_object("double_bed", (0.6, 0.3, 0.3), (1.0, 0.3, 1.0))
    return pad_scene([bed1, bed2]), floor


def scene_bed_outside_floor() -> Tuple[np.ndarray, np.ndarray]:
    floor = synth_rect_floor()
    # Push the bed so half of it lies outside the +x floor edge.
    bed = make_object("double_bed", (2.0, 0.3, 0.0), (1.0, 0.3, 0.9))
    wardrobe = make_object("wardrobe", (-1.4, 1.0, -1.3), (0.5, 1.0, 0.3))
    return pad_scene([bed, wardrobe]), floor


def scene_rotated_corner_clip() -> Tuple[np.ndarray, np.ndarray]:
    """Rotated wardrobe wedged into a corner; corner just clips out."""
    floor = synth_rect_floor()
    # 45-deg rotated wardrobe at the corner.
    wardrobe = make_object("wardrobe",
                           (-1.7, 1.0, -1.7),
                           (0.4, 1.0, 0.6),
                           angle_rad=np.pi / 4)
    return pad_scene([wardrobe]), floor


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #


def _rotated_corners(cx, cz, hx, hz, cos_t, sin_t):
    local = np.array([[-hx, -hz], [-hx, hz], [hx, hz], [hx, -hz]], dtype=np.float64)
    rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cz], dtype=np.float64)


CLASS_COLORS = {
    "double_bed": "#4878d0",
    "single_bed": "#4878d0",
    "kids_bed": "#4878d0",
    "wardrobe": "#937860",
    "cabinet": "#937860",
    "children_cabinet": "#937860",
    "nightstand": "#bababa",
    "desk": "#d65f5f",
    "table": "#d65f5f",
    "coffee_table": "#d65f5f",
    "dressing_table": "#d65f5f",
    "tv_stand": "#8c8c8c",
    "bookshelf": "#a07a3a",
    "shelf": "#a07a3a",
    "chair": "#82c341",
    "dressing_chair": "#82c341",
    "stool": "#82c341",
    "armchair": "#82c341",
    "sofa": "#82c341",
    "ceiling_lamp": "#dec841",
    "pendant_lamp": "#dec841",
}


def visualise_scene(scene_vec: np.ndarray, floor: np.ndarray, title: str,
                    rewards: Dict[str, float],
                    diag_pen: Optional[Dict] = None,
                    diag_oob: Optional[Dict] = None,
                    out_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    floor_poly = MplPoly(floor, closed=True, fill=False,
                         edgecolor="black", linewidth=2.5, zorder=1)
    ax.add_patch(floor_poly)

    # Draw each object footprint.
    for n in range(scene_vec.shape[0]):
        one_hot = scene_vec[n, 8:8 + NUM_CLASSES_BEDROOM]
        cls = int(np.argmax(one_hot))
        if cls == LABEL_TO_IDX["empty"]:
            continue
        label = BEDROOM_LABELS[cls]
        cx, _, cz = scene_vec[n, 0:3]
        hx, _, hz = scene_vec[n, 3:6]
        cos_t, sin_t = scene_vec[n, 6:8]
        corners = _rotated_corners(cx, cz, hx, hz, cos_t, sin_t)
        is_ceiling = label in ("ceiling_lamp", "pendant_lamp")
        ax.add_patch(MplPoly(
            corners, closed=True,
            facecolor=CLASS_COLORS.get(label, "#aaaaaa"),
            edgecolor="black", linewidth=1.0,
            alpha=0.35 if is_ceiling else 0.7,
            zorder=2 if is_ceiling else 3,
        ))
        ax.text(cx, cz, label, fontsize=7, ha="center", va="center",
                color="black", zorder=4)

    # Mark colliding pairs.
    if diag_pen is not None and diag_pen.get("pairs"):
        for rec in diag_pen["pairs"][0]:
            i, j = rec["i"], rec["j"]
            ci = scene_vec[i, [0, 2]]
            cj = scene_vec[j, [0, 2]]
            ax.plot([ci[0], cj[0]], [ci[1], cj[1]],
                    color="red", linewidth=2.0, zorder=5,
                    label="_collision")

    # Padding around bounds.
    all_x = np.concatenate([floor[:, 0], scene_vec[:, 0]])
    all_z = np.concatenate([floor[:, 1], scene_vec[:, 2]])
    pad = 0.5
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_z.min() - pad, all_z.max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.grid(True, alpha=0.3)

    parts = [f"r_pen = {rewards['penetration']:.3f}",
             f"r_oob = {rewards['boundary']:.3f}"]
    if diag_pen is not None:
        parts.append(
            f"pen_vol = {diag_pen['total_volume'][0]:.4f} m^3, "
            f"#pairs = {diag_pen['num_colliding_pairs'][0]}"
        )
        for rec in diag_pen["pairs"][0][:3]:
            parts.append(f"{rec['label_i']}↔{rec['label_j']} v={rec['volume']:.3f}")
    if diag_oob is not None:
        parts.append(f"oob_area = {diag_oob[0]['total_outside_area']:.4f} m^2")
    ax.set_title(title + "\n" + "   |   ".join(parts), fontsize=10)

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_all(data_root: Path, out_dir: Path, sdf_cache_dir: Optional[Path]):
    out_dir.mkdir(parents=True, exist_ok=True)

    real_scene, real_floor = load_real_bedroom(data_root, idx_hint=0)

    cases = [
        ("real_scene",          real_scene,                              real_floor),
        ("chair_under_desk",    *scene_chair_under_desk()),
        ("chair_on_desk",       *scene_chair_on_desk()),
        ("chair_clipping_desk", *scene_chair_clipping_desk()),
        ("beds_overlapping",    *scene_beds_overlapping()),
        ("bed_outside_floor",   *scene_bed_outside_floor()),
        ("rotated_corner_clip", *scene_rotated_corner_clip()),
    ]

    sdf_cache = None
    if sdf_cache_dir is not None and sdf_cache_dir.exists():
        # Cache exists on disk → use it for real scenes, ignore for synthetic.
        # Synthetic floors won't be in the cache so on-the-fly path runs.
        sdf_cache = SDFCache(str(sdf_cache_dir), split="test", lazy=True)
        print(f"[sdf] using cache dir: {sdf_cache_dir} (lazy)")
    else:
        print(f"[sdf] no cache at {sdf_cache_dir} - SDFs will be computed on the fly")

    summary = []
    for name, scene_vec, floor in cases:
        scenes = torch.from_numpy(scene_vec[None]).float()
        parsed = parse_local(scenes)

        r_pen, diag_pen = compute_non_penetration_reward(
            parsed,
            room_type="bedroom",
            sigma_volume=0.05,
            return_diagnostics=True,
        )
        r_oob, diag_oob = compute_oob_reward_sdf(
            parsed,
            floor_polygons=[floor],
            indices=None,
            sdf_cache=sdf_cache,
            sigma_area=0.05,
            return_diagnostics=True,
        )

        rewards = {"penetration": float(r_pen[0]), "boundary": float(r_oob[0])}
        summary.append({
            "case": name,
            **rewards,
            "pen_vol": diag_pen["total_volume"][0],
            "num_pairs": diag_pen["num_colliding_pairs"][0],
            "oob_area": diag_oob[0]["total_outside_area"],
        })

        out_path = out_dir / f"{name}.png"
        visualise_scene(scene_vec, floor, name, rewards, diag_pen, diag_oob,
                        out_path=str(out_path))
        print(f"  - wrote {out_path}")

    print("\n=== summary ===")
    print(f"{'case':22} {'r_pen':>8} {'r_oob':>8} {'pen_vol':>10} "
          f"{'#pairs':>7} {'oob_area':>10}")
    for row in summary:
        print(f"{row['case']:22} {row['penetration']:>8.3f} {row['boundary']:>8.3f} "
              f"{row['pen_vol']:>10.4f} {row['num_pairs']:>7d} {row['oob_area']:>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path,
                    default=Path("/home/pramish_paudel/codes/3dhope_data"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--sdf_cache", type=Path,
                    default=Path("/home/pramish_paudel/codes/3dhope_data/sdf_cache"))
    args = ap.parse_args()
    run_all(args.data_root, args.out_dir, args.sdf_cache)


if __name__ == "__main__":
    main()
