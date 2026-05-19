"""Non-penetration reward.

Upgrades over the previous AABB version:
  * Uses oriented bounding boxes (OBB) in the xz plane with the per-object
    rotation angle that the model actually predicts. Axis-aligned was treating
    rotated furniture as if it were not rotated, which produced large false
    positives.
  * Reports a 3D intersection volume = (OBB area in xz) * (1D y-overlap),
    closer to actual physical interpenetration.
  * Adds a "top-slab" tuck-under exception so that a chair tucked under a
    desk/table does not count as a collision. The table is treated as if its
    only collidable y-range with seating objects were the top thickness slab
    of the bbox; if the chair sits below the slab, y-overlap is zero so the
    pair contributes nothing.
  * Smooth bounded reward in (0, 1] using r = exp(-violation / sigma) so RL
    receives a well-conditioned, monotone signal everywhere.

Designed to be used as a (non-differentiable) reward signal; computation is
vectorised where it matters and falls back to numpy for the OBB area test.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

BEDROOM_LABELS = {
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
    21: "empty",
}

LIVINGROOM_LABELS = {
    0: "armchair",
    1: "bookshelf",
    2: "cabinet",
    3: "ceiling_lamp",
    4: "chaise_longue_sofa",
    5: "chinese_chair",
    6: "coffee_table",
    7: "console_table",
    8: "corner_side_table",
    9: "desk",
    10: "dining_chair",
    11: "dining_table",
    12: "l_shaped_sofa",
    13: "lazy_sofa",
    14: "lounge_chair",
    15: "loveseat_sofa",
    16: "multi_seat_sofa",
    17: "pendant_lamp",
    18: "round_end_table",
    19: "shelf",
    20: "stool",
    21: "tv_stand",
    22: "wardrobe",
    23: "wine_cabinet",
    24: "empty",
}

# Pendant / ceiling-mounted classes — exempt from ground penetration.
CEILING_CLASSES = {"ceiling_lamp", "pendant_lamp"}

# Class groups for tuck-under handling.
SEATING_CLASSES = {
    "chair",
    "dressing_chair",
    "stool",
    "dining_chair",
    "chinese_chair",
    "lounge_chair",
}

# Tables with usable underspace. Restricted to the ones that actually have free
# room beneath (excludes nightstand, cabinet, tv_stand, wardrobe).
TABLES_WITH_UNDERSPACE = {
    "coffee_table",
    "desk",
    "dressing_table",
    "table",
    "dining_table",
    "console_table",
}

# Approximate tabletop thickness (m). Anything below `T_top - TABLE_TOP_SLAB`
# is treated as free space when checking against seating objects.
TABLE_TOP_SLAB_M = 0.10


def _labels_for_room(room_type: str) -> Dict[int, str]:
    if room_type and "living" in room_type.lower():
        return LIVINGROOM_LABELS
    return BEDROOM_LABELS


# ---------------------------------------------------------------------------
# OBB intersection in xz (numpy / shapely)
# ---------------------------------------------------------------------------


def _obb_corners_xz(cx: float, cz: float, hx: float, hz: float,
                    cos_t: float, sin_t: float) -> np.ndarray:
    """Return (4, 2) corners of the oriented xz footprint.

    Rotation convention matches ThreedFront eval (see boundary reward).
    """
    local = np.array(
        [[-hx, -hz], [-hx, hz], [hx, hz], [hx, -hz]],
        dtype=np.float64,
    )
    rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cz], dtype=np.float64)


def _obb_intersection_area(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    pa = Polygon(corners_a.tolist())
    pb = Polygon(corners_b.tolist())
    if (not pa.is_valid) or (not pb.is_valid):
        return 0.0
    if pa.area <= 1e-12 or pb.area <= 1e-12:
        return 0.0
    return float(pa.intersection(pb).area)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def compute_aabb_penetration_depth(centers1, sizes1, centers2, sizes2):
    """Backward-compatible AABB penetration depth (kept for callers/tests).

    Returns (B, N1, N2) penetration depth = min overlap across the 3 axes.
    """
    c1 = centers1.unsqueeze(2)
    s1 = sizes1.unsqueeze(2)
    c2 = centers2.unsqueeze(1)
    s2 = sizes2.unsqueeze(1)
    overlap = (s1 + s2) - torch.abs(c1 - c2)
    is_overlap = (overlap > 0).all(dim=3)
    min_overlap = overlap.min(dim=3)[0]
    return torch.where(is_overlap, min_overlap, torch.zeros_like(min_overlap)).clamp_(min=0.0)


def _classify_object(label: str) -> str:
    if label in CEILING_CLASSES:
        return "ceiling"
    if label in SEATING_CLASSES:
        return "seating"
    if label in TABLES_WITH_UNDERSPACE:
        return "table"
    return "other"


def _y_overlap(top_a: float, bot_a: float, top_b: float, bot_b: float) -> float:
    return max(0.0, min(top_a, top_b) - max(bot_a, bot_b))


def compute_non_penetration_reward(
    parsed_scenes: Dict[str, torch.Tensor],
    sigma_volume: float = 0.05,
    table_top_slab_m: float = TABLE_TOP_SLAB_M,
    return_diagnostics: bool = False,
    **kwargs,
):
    """OBB-aware non-penetration reward with table tuck-under exception.

    Args:
        parsed_scenes: dict from ``parse_and_descale_scenes`` with keys
            positions (B,N,3), sizes (B,N,3) half-extents, orientations
            (B,N,2) = [cos, sin], object_indices (B,N), is_empty (B,N),
            device.
        sigma_volume: scale (m^3) for the bounded reward
            ``exp(-violation / sigma)``. A scene whose normalised penetration
            volume equals sigma scores ~0.37.
        table_top_slab_m: thickness (m) of the table-top slab used in the
            tuck-under exception. Tables vs. seating are only compared on
            the top slab; everywhere else they are free space.
        return_diagnostics: if True, also return per-scene diagnostics.

    Returns:
        rewards: (B,) tensor in (0, 1].
        diagnostics (optional): dict with per-scene totals + per-pair info.
    """
    room_type = kwargs.get("room_type", "bedroom")
    labels_map = _labels_for_room(room_type)

    positions = parsed_scenes["positions"]      # (B, N, 3)
    sizes = parsed_scenes["sizes"]              # (B, N, 3) half-extents
    orientations = parsed_scenes["orientations"]  # (B, N, 2)
    object_indices = parsed_scenes["object_indices"]
    is_empty = parsed_scenes["is_empty"]
    device = parsed_scenes["device"]

    B, N = positions.shape[0], positions.shape[1]
    rewards = torch.zeros(B, device=device)
    diag = {"total_volume": [], "num_colliding_pairs": [], "pairs": []}

    pos_np = positions.detach().cpu().numpy()
    siz_np = sizes.detach().cpu().numpy()
    ori_np = orientations.detach().cpu().numpy()
    cls_np = object_indices.detach().cpu().numpy()
    emp_np = is_empty.detach().cpu().numpy()

    for b in range(B):
        kinds: List[str] = []
        for n in range(N):
            if emp_np[b, n]:
                kinds.append("empty")
            else:
                kinds.append(_classify_object(labels_map.get(int(cls_np[b, n]), "")))

        # Pre-compute corners and y extents for active objects.
        active = [n for n in range(N) if kinds[n] not in ("empty", "ceiling")]
        corners = {}
        for n in active:
            corners[n] = _obb_corners_xz(
                pos_np[b, n, 0], pos_np[b, n, 2],
                siz_np[b, n, 0], siz_np[b, n, 2],
                ori_np[b, n, 0], ori_np[b, n, 1],
            )

        total_volume = 0.0
        num_pairs = 0
        pair_records = []

        for i_idx, i in enumerate(active):
            for j in active[i_idx + 1:]:
                area = _obb_intersection_area(corners[i], corners[j])
                if area <= 1e-9:
                    continue

                top_i = pos_np[b, i, 1] + siz_np[b, i, 1]
                bot_i = pos_np[b, i, 1] - siz_np[b, i, 1]
                top_j = pos_np[b, j, 1] + siz_np[b, j, 1]
                bot_j = pos_np[b, j, 1] - siz_np[b, j, 1]

                # Tuck-under exception: when one is seating and the other is a
                # table with underspace, treat the table as just its top slab.
                ki, kj = kinds[i], kinds[j]
                if {ki, kj} == {"seating", "table"}:
                    if ki == "table":
                        bot_i = max(bot_i, top_i - table_top_slab_m)
                    else:
                        bot_j = max(bot_j, top_j - table_top_slab_m)

                yov = _y_overlap(top_i, bot_i, top_j, bot_j)
                vol = area * yov
                if vol <= 1e-9:
                    continue

                total_volume += vol
                num_pairs += 1
                if return_diagnostics:
                    pair_records.append({
                        "i": int(i),
                        "j": int(j),
                        "label_i": labels_map.get(int(cls_np[b, i]), "?"),
                        "label_j": labels_map.get(int(cls_np[b, j]), "?"),
                        "xz_area": float(area),
                        "y_overlap": float(yov),
                        "volume": float(vol),
                    })

        num_active = max(1, len(active))
        violation = total_volume / num_active
        rewards[b] = float(np.exp(-violation / max(sigma_volume, 1e-6)))

        diag["total_volume"].append(total_volume)
        diag["num_colliding_pairs"].append(num_pairs)
        diag["pairs"].append(pair_records)

    if return_diagnostics:
        return rewards, diag
    return rewards
