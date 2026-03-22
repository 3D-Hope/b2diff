import numpy as np
import torch
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union


_IDX_TO_LABEL_BEDROOM = {
    0: "armchair",
    1: "bookshelf",
    2: "cabinet",
    3: "ceiling_lamp",
    4: "chair",
    5: "children_cabinet",
    6: "coffee_table",
    7: "desk",
    8: "double_bed",
    9: "dressing_chair",
    10: "dressing_table",
    11: "kids_bed",
    12: "nightstand",
    13: "pendant_lamp",
    14: "shelf",
    15: "single_bed",
    16: "sofa",
    17: "stool",
    18: "table",
    19: "tv_stand",
    20: "wardrobe",
}


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_idx_to_labels(idx_to_labels):
    if idx_to_labels is None:
        return _IDX_TO_LABEL_BEDROOM
    if len(idx_to_labels) == 0:
        return _IDX_TO_LABEL_BEDROOM
    first_key = next(iter(idx_to_labels.keys()))
    if isinstance(first_key, str):
        return {int(k): v for k, v in idx_to_labels.items()}
    return idx_to_labels


def _build_floor_geometry(vertices, faces, centroid):
    verts = _to_numpy(vertices)
    fcs = _to_numpy(faces)
    ctr = _to_numpy(centroid)

    verts = verts - ctr
    verts_xz = verts[:, [0, 2]]

    polys = []
    for face in fcs:
        face_indices = np.asarray(face, dtype=np.int64)
        if face_indices.size < 3:
            continue
        pts = verts_xz[face_indices]
        poly = Polygon(pts.tolist())
        if poly.is_valid and poly.area > 1e-12:
            polys.append(poly)

    if len(polys) == 0:
        return None

    merged = unary_union(polys)
    if merged.is_empty:
        return None
    if isinstance(merged, (Polygon, MultiPolygon)):
        return merged

    # Fallback for GeometryCollection-like outputs.
    valid_polys = [g for g in getattr(merged, "geoms", []) if isinstance(g, Polygon)]
    if len(valid_polys) == 0:
        return None
    return unary_union(valid_polys)


def _signed_distance_to_wall(point_xz, floor_geom):
    pt = Point(float(point_xz[0]), float(point_xz[1]))
    wall_dist = float(pt.distance(floor_geom.boundary))
    inside_or_on = floor_geom.covers(pt)
    return wall_dist if inside_or_on else -wall_dist


def compute_bed_headboard_wall_reward(
    parsed_scene,
    floor_plan_vertices,
    floor_plan_faces,
    floor_plan_centroid,
    head_target_dist=0.12,
    near_sigma=0.12,
    gap_margin=0.20,
    gap_scale=0.20,
    near_weight=1.0,
    gap_weight=1.0,
    no_bed_reward=0.0,
    **kwargs,
):
    """
    Reward beds where the +axis endpoint (assumed headboard) is closer to wall.

    Terms per bed:
    1) headboard-near-wall term (inside room only)
    2) headboard-vs-foot wall-gap term
    """
    positions = parsed_scene["positions"]
    sizes = parsed_scene["sizes"]
    orientations = parsed_scene["orientations"]
    object_indices = parsed_scene["object_indices"]
    is_empty = parsed_scene["is_empty"]

    device = positions.device
    dtype = positions.dtype
    batch_size = positions.shape[0]

    idx_to_labels = _normalize_idx_to_labels(kwargs.get("idx_to_labels", None))
    bed_indices = {
        idx for idx, label in idx_to_labels.items() if "bed" in str(label).lower()
    }

    rewards = torch.zeros(batch_size, dtype=dtype, device=device)

    for b in range(batch_size):
        floor_geom = _build_floor_geometry(
            floor_plan_vertices[b], floor_plan_faces[b], floor_plan_centroid[b]
        )
        if floor_geom is None:
            continue

        bed_scores = []

        for n in range(positions.shape[1]):
            if bool(is_empty[b, n].item()):
                continue

            obj_idx = int(object_indices[b, n].item())
            if obj_idx not in bed_indices:
                continue

            x = float(positions[b, n, 0].item())
            z = float(positions[b, n, 2].item())
            hx = float(sizes[b, n, 0].item())
            hz = float(sizes[b, n, 2].item())
            cos_t = float(orientations[b, n, 0].item())
            sin_t = float(orientations[b, n, 1].item())

            # Local axes in world XZ plane.
            u = np.array([cos_t, sin_t], dtype=np.float64)
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-8:
                continue
            u = u / norm_u
            v = np.array([-u[1], u[0]], dtype=np.float64)

            # Bed length axis: choose the larger half-extent.
            if hx >= hz:
                length_half = hx
                axis = u
            else:
                length_half = hz
                axis = v

            center = np.array([x, z], dtype=np.float64)

            # User-specified convention: +direction is headboard.
            head_pt = center + length_half * axis
            foot_pt = center - length_half * axis

            d_head = _signed_distance_to_wall(head_pt, floor_geom)
            d_foot = _signed_distance_to_wall(foot_pt, floor_geom)

            if d_head <= 0.0:
                near_term = 0.0
            else:
                near_term = np.exp(
                    -((d_head - float(head_target_dist)) ** 2)
                    / (2.0 * float(near_sigma) ** 2 + 1e-12)
                )

            delta = d_foot - d_head
            gap_term = np.tanh((delta - float(gap_margin)) / (float(gap_scale) + 1e-12))

            bed_score = float(near_weight) * float(near_term) + float(gap_weight) * float(gap_term)
            bed_scores.append(bed_score)

        if len(bed_scores) == 0:
            rewards[b] = float(no_bed_reward)
        else:
            rewards[b] = float(np.mean(bed_scores))

    return rewards


def compute_reward(parsed_scene, **kwargs):
    """Generic custom-reward entrypoint used by dynamic loader."""
    passthrough_kwargs = dict(kwargs)
    floor_plan_vertices = passthrough_kwargs.pop("floor_plan_vertices", None)
    floor_plan_faces = passthrough_kwargs.pop("floor_plan_faces", None)
    floor_plan_centroid = passthrough_kwargs.pop("floor_plan_centroid", None)

    if floor_plan_vertices is None or floor_plan_faces is None or floor_plan_centroid is None:
        raise ValueError(
            "bed_headboard_wall requires floor_plan_vertices, floor_plan_faces, "
            "and floor_plan_centroid. Set midiffusion.floor_geometry_path and pass "
            "geometry via selection.py."
        )

    with torch.no_grad():
        return compute_bed_headboard_wall_reward(
            parsed_scene,
            floor_plan_vertices=floor_plan_vertices,
            floor_plan_faces=floor_plan_faces,
            floor_plan_centroid=floor_plan_centroid,
            **passthrough_kwargs,
        )
