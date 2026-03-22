#!/usr/bin/env python3
"""
Extract floor data for the test split in dataset order.

Output is a JSON list where index i corresponds directly to test scene index i.
This is the simplest sequential export when condition order already matches
test split order.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Ensure ThreedFront package is importable when running from repo root.
ROOT_DIR = Path(__file__).resolve().parents[1]
THREED_FRONT_DIR = ROOT_DIR / "3d_layout_generation" / "ThreedFront"
if str(THREED_FRONT_DIR) not in sys.path:
    sys.path.append(str(THREED_FRONT_DIR))

from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_file",
        required=True,
        help="Pickled ThreedFrontResults file from eval pipeline.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path: list where index=test scene index and value=floor polygon corners.",
    )
    parser.add_argument(
        "--out_floor_geometry",
        default=None,
        help=(
            "Optional JSON path: list where index=test scene index and value is a dict "
            "with floor_plan_vertices, floor_plan_faces, and floor_plan_centroid."
        ),
    )
    args = parser.parse_args()

    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    if not isinstance(threed_front_results, ThreedFrontResults):
        raise TypeError("result_file must contain a ThreedFrontResults object")

    config = threed_front_results.config
    print(f"{config=}")
    raw_dataset = get_raw_dataset(
        config["data"],
        split=config["validation"].get("splits", ["test"]),
        include_room_mask=True,
    )

    num_scenes = len(raw_dataset)
    floor_polygons = []
    floor_geometry = []

    for scene_idx in range(num_scenes):
        room_npz_path = raw_dataset._path_to_room(scene_idx)
        room_npz = np.load(room_npz_path)

        if "floor_plan_ordered_corners" in room_npz:
            corners = room_npz["floor_plan_ordered_corners"].astype(np.float64)
        else:
            raise ValueError(f"Ordered floor corners not found for scene {scene_idx} at {room_npz_path}")
        floor_polygons.append(corners.tolist())

        required_keys = (
            "floor_plan_vertices",
            "floor_plan_faces",
            "floor_plan_centroid",
        )
        for key in required_keys:
            if key not in room_npz:
                raise ValueError(
                    f"{key} not found for scene {scene_idx} at {room_npz_path}"
                )

        floor_geometry.append(
            {
                "floor_plan_vertices": room_npz["floor_plan_vertices"].astype(np.float64).tolist(),
                "floor_plan_faces": room_npz["floor_plan_faces"].astype(np.int64).tolist(),
                "floor_plan_centroid": room_npz["floor_plan_centroid"].astype(np.float64).tolist(),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(floor_polygons, f)

    if args.out_floor_geometry:
        geometry_path = Path(args.out_floor_geometry)
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(geometry_path, "w") as f:
            json.dump(floor_geometry, f)

    print(f"Extracted test scenes: {num_scenes}")
    print(f"Saved floor polygons: {out_path}")
    if args.out_floor_geometry:
        print(f"Saved floor geometry: {geometry_path}")


if __name__ == "__main__":
    main()
