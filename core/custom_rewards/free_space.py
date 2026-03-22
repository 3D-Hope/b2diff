import numpy as np
import torch
from shapely.geometry import Polygon


def _bbox_xz_corners_oriented(x, z, hx, hz, cos_t, sin_t, erosion=0.0):
    hx = max(float(hx) - float(erosion), 0.0)
    hz = max(float(hz) - float(erosion), 0.0)
    local = np.array(
        [
            [-hx, -hz],
            [-hx, hz],
            [hx, hz],
            [hx, -hz],
        ],
        dtype=np.float64,
    )

    rot = np.array(
        [[float(cos_t), float(sin_t)], [-float(sin_t), float(cos_t)]],
        dtype=np.float64,
    )
    return local @ rot.T + np.array([float(x), float(z)], dtype=np.float64)


def compute_free_space_reward(
    parsed_scene,
    floor_polygons,
    free_space_threshold=0.60,
    reward_if_threshold_met=1.0,
    erosion=0.0,
    empty_scene_penalty=-1.0,
    invalid_floor_penalty=-1.0,
    penalize_shortfall=True,
    **kwargs,
):
    """
    Reward scenes with larger free space inside the floor polygon.

    Per scene:
      free_ratio = (floor_area - occupied_inside_floor_area) / floor_area

    Reward:
      - if free_ratio >= free_space_threshold: reward_if_threshold_met
      - else: -(free_space_threshold - free_ratio) when penalize_shortfall=True
              -free_ratio when penalize_shortfall=False
    """
    if floor_polygons is None:
        raise ValueError("free_space reward requires floor_polygons.")

    positions = parsed_scene["positions"]
    sizes = parsed_scene["sizes"]
    orientations = parsed_scene["orientations"]
    is_empty = parsed_scene["is_empty"]
    device = parsed_scene["device"]

    B, N = positions.shape[:2]
    rewards = torch.zeros(B, device=device, dtype=torch.float32)

    threshold = float(free_space_threshold)
    threshold_reward = float(reward_if_threshold_met)

    def _safe_geometry(g):
        """Return a valid, non-empty shapely geometry or None."""
        if g is None:
            return None
        try:
            if g.is_empty:
                return None
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty:
                return None
            if float(getattr(g, "area", 0.0)) <= 1e-12:
                return None
            return g
        except Exception:
            return None

    def _safe_union_area(parts):
        """Robust union area with fallbacks across shapely versions/types."""
        cleaned = []
        for g in parts:
            gg = _safe_geometry(g)
            if gg is not None:
                cleaned.append(gg)

        if len(cleaned) == 0:
            return 0.0

        # Fast path for a single shape.
        if len(cleaned) == 1:
            return float(cleaned[0].area)

        # Iterative union is more robust than unary_union in mixed-runtime setups.
        cur = cleaned[0]
        for g in cleaned[1:]:
            try:
                cur = cur.union(g)
                if not cur.is_valid:
                    cur = cur.buffer(0)
            except Exception:
                # Skip problematic geometry and continue.
                continue

        cur = _safe_geometry(cur)
        if cur is None:
            # Conservative fallback: cap summed area at zero/later floor area clip handles upper bound.
            return float(sum(float(gg.area) for gg in cleaned))
        return float(cur.area)

    for b in range(B):
        floor_poly_np = np.asarray(floor_polygons[b], dtype=np.float64)
        if floor_poly_np.ndim != 2 or floor_poly_np.shape[0] < 3:
            rewards[b] = float(invalid_floor_penalty)
            continue

        floor_poly = Polygon(floor_poly_np.tolist())
        if (not floor_poly.is_valid) or floor_poly.area <= 1e-12:
            rewards[b] = float(invalid_floor_penalty)
            continue

        occupied_parts = []
        valid_object_count = 0

        for n in range(N):
            if bool(is_empty[b, n].item()):
                continue

            x = float(positions[b, n, 0].item())
            z = float(positions[b, n, 2].item())
            hx = float(sizes[b, n, 0].item())
            hz = float(sizes[b, n, 2].item())
            cos_t = float(orientations[b, n, 0].item())
            sin_t = float(orientations[b, n, 1].item())

            corners = _bbox_xz_corners_oriented(x, z, hx, hz, cos_t, sin_t, erosion=erosion)
            obj_poly = Polygon(corners.tolist())
            if (not obj_poly.is_valid) or obj_poly.area <= 1e-12:
                continue

            inter = floor_poly.intersection(obj_poly)
            if inter.is_empty or inter.area <= 1e-12:
                continue

            valid_object_count += 1
            occupied_parts.append(inter)

        if valid_object_count == 0:
            rewards[b] = float(empty_scene_penalty)
            continue

        occupied_area = float(max(_safe_union_area(occupied_parts), 0.0))
        floor_area = float(max(floor_poly.area, 1e-12))
        occupied_area = min(occupied_area, floor_area)
        free_ratio = float(np.clip((floor_area - occupied_area) / floor_area, 0.0, 1.0))

        if free_ratio >= threshold:
            rewards[b] = threshold_reward
        else:
            if penalize_shortfall:
                rewards[b] = -(threshold - free_ratio)
            else:
                rewards[b] = -free_ratio

    return rewards


def compute_reward(parsed_scene, **kwargs):
    """Generic custom-reward entrypoint used by dynamic loader."""
    floor_polygons = kwargs.pop("floor_polygons", None)
    with torch.no_grad():
        return compute_free_space_reward(parsed_scene, floor_polygons=floor_polygons, **kwargs)
