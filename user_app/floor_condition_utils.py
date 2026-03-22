from typing import Dict, Iterable, Tuple

import numpy as np
from shapely.geometry import MultiPoint, Polygon


def validate_fpbpn(fpbpn: np.ndarray) -> np.ndarray:
    """Validate and return fpbpn as float32 array of shape (num_points, 4)."""
    arr = np.asarray(fpbpn, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"fpbpn must have shape (num_points, 4), got {arr.shape}")
    if arr.shape[0] < 8:
        raise ValueError("fpbpn must contain at least 8 boundary points")
    return arr


def scale_fpbpn_world_to_model(fpbpn_world: np.ndarray, fpbpn_bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Scale fpbpn from world space to model space [-1, 1] using dataset bounds."""
    fpbpn_world = validate_fpbpn(fpbpn_world)
    minimum, maximum = fpbpn_bounds
    minimum = np.asarray(minimum, dtype=np.float32)
    maximum = np.asarray(maximum, dtype=np.float32)
    clipped = np.clip(fpbpn_world, minimum, maximum)
    x = (clipped - minimum) / (maximum - minimum)
    x = 2.0 * x - 1.0
    return x.astype(np.float32)


def descale_fpbpn_model_to_world(fpbpn_scaled: np.ndarray, fpbpn_bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Descale fpbpn from model space [-1, 1] to world space."""
    fpbpn_scaled = validate_fpbpn(fpbpn_scaled)
    minimum, maximum = fpbpn_bounds
    minimum = np.asarray(minimum, dtype=np.float32)
    maximum = np.asarray(maximum, dtype=np.float32)
    x = (fpbpn_scaled + 1.0) / 2.0
    x = x * (maximum - minimum) + minimum
    return x.astype(np.float32)


def normalize_user_fpbpn(
    fpbpn: np.ndarray,
    input_space: str,
    fpbpn_bounds: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Return both scaled and world fpbpn arrays.

    input_space:
      - "scaled": user input already in model [-1, 1] space
      - "world": user input in world coordinates
    """
    input_space = input_space.lower().strip()
    if input_space not in {"scaled", "world"}:
        raise ValueError(f"input_space must be 'scaled' or 'world', got {input_space}")

    fpbpn = validate_fpbpn(fpbpn)
    if input_space == "scaled":
        fpbpn_scaled = fpbpn
        fpbpn_world = descale_fpbpn_model_to_world(fpbpn, fpbpn_bounds)
    else:
        fpbpn_world = fpbpn
        fpbpn_scaled = scale_fpbpn_world_to_model(fpbpn, fpbpn_bounds)

    return {
        "fpbpn_scaled": fpbpn_scaled.astype(np.float32),
        "fpbpn_world": fpbpn_world.astype(np.float32),
    }


def reconstruct_floor_polygon_xz_from_fpbpn(fpbpn_world: np.ndarray) -> np.ndarray:
    """Build a clean XZ polygon from fpbpn boundary points.

    This intentionally stays simple: it assumes fpbpn comes from dataset pipeline
    and only repairs invalid geometry with a convex-hull fallback.
    """
    fpbpn_world = validate_fpbpn(fpbpn_world)
    points_xz = np.asarray(fpbpn_world[:, :2], dtype=np.float64)

    poly = Polygon(points_xz.tolist())
    if (not poly.is_valid) or poly.area <= 1e-12:
        poly = MultiPoint(points_xz.tolist()).convex_hull

    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)

    if (not poly.is_valid) or poly.area <= 1e-12:
        raise ValueError("Failed to reconstruct a valid floor polygon from fpbpn")

    coords = np.asarray(poly.exterior.coords[:-1], dtype=np.float32)
    return coords


def center_floor_polygon_xz(floor_polygon_xz: np.ndarray) -> Dict[str, np.ndarray]:
    """Center floor polygon in XZ frame to match encoded scene coordinates.

    Returns:
      - polygon_centered_xz: centroid-centered polygon used by boundary reward/render
      - centroid_xz: original centroid in XZ
    """
    poly = np.asarray(floor_polygon_xz, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError(f"floor polygon must have shape (num_vertices, 2), got {poly.shape}")
    centroid = poly.mean(axis=0)
    centered = poly - centroid[None, :]
    return {
        "polygon_centered_xz": centered.astype(np.float32),
        "centroid_xz": centroid.astype(np.float32),
    }


def build_floor_tri_mesh_inputs(floor_polygon_xz: np.ndarray) -> Dict[str, np.ndarray]:
    """Return floor vertices/faces arrays suitable for rendering a flat floor mesh."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import triangulate

    polygon_xz = np.asarray(floor_polygon_xz, dtype=np.float64)
    poly = ShapelyPolygon(polygon_xz.tolist())
    if (not poly.is_valid) or poly.area <= 1e-12:
        raise ValueError("Invalid polygon for floor triangulation")

    triangles = triangulate(poly)
    vertices = []
    faces = []
    vertex_index = {}

    def get_vid(pt):
        key = (float(pt[0]), float(pt[1]))
        if key not in vertex_index:
            vertex_index[key] = len(vertices)
            vertices.append([key[0], 0.0, key[1]])
        return vertex_index[key]

    for tri in triangles:
        coords = list(tri.exterior.coords)[:3]
        f = [get_vid(c) for c in coords]
        if len(set(f)) == 3:
            faces.append(f)

    if len(vertices) < 3 or len(faces) == 0:
        raise ValueError("Failed to triangulate floor polygon")

    return {
        "vertices": np.asarray(vertices, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int32),
    }
