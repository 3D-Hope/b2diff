"""Out-of-boundary reward.

Two public reward functions:

  * ``compute_boundary_violation_reward`` (legacy signature, used elsewhere in
    the repo) — kept for backward compatibility; routes to the new smooth SDF
    based formulation.
  * ``compute_oob_reward_sdf`` — new RL-friendly reward. For each scene we
    grid-sample points inside each object's oriented xz footprint, query a
    signed-distance field of the floor polygon (positive inside), accumulate
    ``sum(max(0, -sdf) * cell_area)`` to get an outside-area estimate, then
    apply ``r = exp(-violation / sigma)`` so the reward lives in (0, 1].

The SDF cache machinery is preserved from the previous file: if a cache is
found on disk it is loaded; otherwise the SDF is computed on the fly (and the
result optionally written back to disk for reuse).
"""

import os
import pickle

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
from shapely.geometry import Polygon

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EROSION_M = 0.02  # tighter than the old 5 cm tolerance
DEFAULT_GRID_RESOLUTION = 0.05
DEFAULT_FOOTPRINT_SAMPLES = 7  # K x K samples per object footprint


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _oriented_corners(cx, cz, hx, hz, cos_t, sin_t, erosion: float = DEFAULT_EROSION_M):
    hx = max(float(hx) - erosion, 0.0)
    hz = max(float(hz) - erosion, 0.0)
    local = np.array(
        [[-hx, -hz], [-hx, hz], [hx, hz], [hx, -hz]], dtype=np.float64
    )
    rot = np.array(
        [[float(cos_t), float(sin_t)], [-float(sin_t), float(cos_t)]],
        dtype=np.float64,
    )
    return local @ rot.T + np.array([float(cx), float(cz)], dtype=np.float64)


def _sample_footprint_xz(cx, cz, hx, hz, cos_t, sin_t,
                         erosion: float = DEFAULT_EROSION_M,
                         k: int = DEFAULT_FOOTPRINT_SAMPLES):
    """Return (K*K, 2) world-frame samples spread uniformly inside the eroded
    footprint, plus the area-per-cell."""
    hx = max(float(hx) - erosion, 0.0)
    hz = max(float(hz) - erosion, 0.0)
    if hx <= 1e-6 or hz <= 1e-6:
        return np.zeros((0, 2)), 0.0
    us = (np.arange(k) + 0.5) / k * 2.0 - 1.0  # cell centers in [-1, 1]
    grid_u, grid_v = np.meshgrid(us, us, indexing="xy")
    local = np.stack([grid_u.ravel() * hx, grid_v.ravel() * hz], axis=1)
    rot = np.array(
        [[float(cos_t), float(sin_t)], [-float(sin_t), float(cos_t)]],
        dtype=np.float64,
    )
    world = local @ rot.T + np.array([float(cx), float(cz)], dtype=np.float64)
    cell_area = (2 * hx / k) * (2 * hz / k)
    return world, cell_area


# ---------------------------------------------------------------------------
# Floor SDF
# ---------------------------------------------------------------------------


class SDFBoundaryChecker:
    """Signed-distance field of a floor polygon (positive inside).

    Re-export-compatible with the previous module (same cache schema).
    """

    def __init__(self, floor_vertices, grid_resolution: float = DEFAULT_GRID_RESOLUTION):
        self.floor_vertices = np.asarray(floor_vertices, dtype=np.float32)
        self.grid_resolution = grid_resolution
        self._validate_polygon()
        padding = max(2.0, self.grid_resolution * 10)
        min_x = self.floor_vertices[:, 0].min() - padding
        max_x = self.floor_vertices[:, 0].max() + padding
        min_z = self.floor_vertices[:, 1].min() - padding
        max_z = self.floor_vertices[:, 1].max() + padding
        self.world_bounds = (min_x, max_x, min_z, max_z)
        self.sdf_grid, self.x_range, self.z_range = self._compute_sdf_grid()

    # --- validation -----------------------------------------------------
    def _validate_polygon(self):
        if len(self.floor_vertices) < 3:
            raise ValueError("Polygon must have >= 3 vertices")
        area = 0.0
        n = len(self.floor_vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self.floor_vertices[i, 0] * self.floor_vertices[j, 1]
            area -= self.floor_vertices[j, 0] * self.floor_vertices[i, 1]
        area *= 0.5
        if abs(area) < 1e-6:
            raise ValueError("Polygon has zero area")
        self.is_ccw = area > 0

    # --- caching --------------------------------------------------------
    def get_cache_data(self) -> Dict:
        return {
            "sdf_grid": self.sdf_grid,
            "world_bounds": self.world_bounds,
            "grid_resolution": self.grid_resolution,
            "x_range": self.x_range,
            "z_range": self.z_range,
            "is_ccw": self.is_ccw,
        }

    @classmethod
    def from_cache_data(cls, cache_data: Dict) -> "SDFBoundaryChecker":
        obj = cls.__new__(cls)
        obj.sdf_grid = cache_data["sdf_grid"]
        obj.world_bounds = cache_data["world_bounds"]
        obj.grid_resolution = cache_data["grid_resolution"]
        obj.x_range = cache_data["x_range"]
        obj.z_range = cache_data["z_range"]
        obj.is_ccw = cache_data.get("is_ccw", True)
        obj.floor_vertices = None
        return obj

    # --- queries --------------------------------------------------------
    def query_sdf(self, points: np.ndarray) -> np.ndarray:
        """Vectorised bilinear SDF query. ``points`` has shape (M, 2) in (x, z)."""
        if points.size == 0:
            return np.zeros(0, dtype=np.float64)
        x = points[:, 0]
        z = points[:, 1]
        min_x, max_x, min_z, max_z = self.world_bounds
        margin = self.grid_resolution * 5
        far_outside = (
            (x < min_x - margin)
            | (x > max_x + margin)
            | (z < min_z - margin)
            | (z > max_z + margin)
            | ~np.isfinite(x)
            | ~np.isfinite(z)
        )
        x_idx = (x - min_x) / self.grid_resolution
        z_idx = (z - min_z) / self.grid_resolution
        x_idx = np.clip(x_idx, 0, len(self.x_range) - 1.001)
        z_idx = np.clip(z_idx, 0, len(self.z_range) - 1.001)
        x0 = np.floor(x_idx).astype(np.int64)
        z0 = np.floor(z_idx).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, len(self.x_range) - 1)
        z1 = np.clip(z0 + 1, 0, len(self.z_range) - 1)
        wx = x_idx - x0
        wz = z_idx - z0
        s00 = self.sdf_grid[z0, x0]
        s10 = self.sdf_grid[z0, x1]
        s01 = self.sdf_grid[z1, x0]
        s11 = self.sdf_grid[z1, x1]
        sdf = (
            s00 * (1 - wx) * (1 - wz)
            + s10 * wx * (1 - wz)
            + s01 * (1 - wx) * wz
            + s11 * wx * wz
        )
        # NOTE: the SDF grid is already built with ray-cast inside/outside,
        # which is winding-independent. The legacy code applied a sign flip
        # based on ``is_ccw`` that double-inverted the value and made all
        # interior points read as outside; we keep ``is_ccw`` for cache
        # compatibility but do not apply it here.
        sdf = np.where(far_outside, -999.0, sdf)
        return sdf

    def check_violation(self, point: Tuple[float, float]) -> float:
        return float(max(0.0, -self.query_sdf(np.asarray([point], dtype=np.float64))[0]))

    # --- grid construction ---------------------------------------------
    def _compute_sdf_grid(self):
        min_x, max_x, min_z, max_z = self.world_bounds
        x_range = np.arange(min_x, max_x + self.grid_resolution, self.grid_resolution)
        z_range = np.arange(min_z, max_z + self.grid_resolution, self.grid_resolution)
        verts = self.floor_vertices
        edges = [(verts[i], verts[(i + 1) % len(verts)]) for i in range(len(verts))]
        sdf_grid = np.zeros((len(z_range), len(x_range)), dtype=np.float32)
        # Vectorised distance from grid points to each edge.
        gx, gz = np.meshgrid(x_range, z_range)
        gp = np.stack([gx.ravel(), gz.ravel()], axis=1)  # (M, 2)
        min_d = np.full(gp.shape[0], np.inf, dtype=np.float64)
        for a, b in edges:
            ab = b - a
            ap = gp - a
            denom = float(ab @ ab) + 1e-12
            t = np.clip((ap @ ab) / denom, 0.0, 1.0)
            closest = a + t[:, None] * ab
            d = np.linalg.norm(gp - closest, axis=1)
            min_d = np.minimum(min_d, d)
        inside = self._points_in_polygon(gp)
        sdf_grid = np.where(inside, min_d, -min_d).reshape(len(z_range), len(x_range)).astype(np.float32)
        return sdf_grid, x_range, z_range

    def _points_in_polygon(self, pts: np.ndarray) -> np.ndarray:
        verts = self.floor_vertices
        n = len(verts)
        inside = np.zeros(pts.shape[0], dtype=bool)
        j = n - 1
        for i in range(n):
            xi, zi = verts[i]
            xj, zj = verts[j]
            crosses = ((zi > pts[:, 1]) != (zj > pts[:, 1]))
            x_int = (xj - xi) * (pts[:, 1] - zi) / (zj - zi + 1e-12) + xi
            cond = crosses & (pts[:, 0] < x_int)
            inside ^= cond
            j = i
        return inside


# ---------------------------------------------------------------------------
# SDF cache (load-once, RAM-resident)
# ---------------------------------------------------------------------------


class SDFCache:
    """Lightweight wrapper around a directory of per-scene SDF pickles."""

    def __init__(self, cache_dir: Optional[str], split: str = "train_val",
                 grid_resolution: float = DEFAULT_GRID_RESOLUTION,
                 lazy: bool = True, verbose: bool = False):
        self.base_cache_dir = Path(cache_dir) if cache_dir else None
        self.grid_resolution = grid_resolution
        self.split = split
        if split in ("train", "val", "train_val"):
            sub = "train_val"
        elif split == "test":
            sub = "test"
        else:
            sub = split
        self.cache_dir = (self.base_cache_dir / sub) if self.base_cache_dir else None
        self._memory_cache: Dict[int, Dict] = {}
        if self.cache_dir is not None and self.cache_dir.exists() and not lazy:
            self._load_all_to_memory(verbose=verbose)

    def _load_all_to_memory(self, verbose: bool = False):
        sdf_files = sorted(self.cache_dir.glob("sdf_*.pkl"))
        it = tqdm(sdf_files, desc=f"Loading {self.split} SDFs") if verbose else sdf_files
        for f in it:
            try:
                idx = int(f.stem.split("_")[1])
                with open(f, "rb") as fh:
                    self._memory_cache[idx] = pickle.load(fh)
            except Exception:
                continue

    def load(self, scene_idx: int) -> Optional[Dict]:
        if scene_idx in self._memory_cache:
            return self._memory_cache[scene_idx]
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"sdf_{int(scene_idx)}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            self._memory_cache[int(scene_idx)] = data
            return data
        return None

    def save(self, scene_idx: int, cache_data: Dict):
        self._memory_cache[int(scene_idx)] = cache_data
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_dir / f"sdf_{int(scene_idx)}.pkl", "wb") as fh:
            pickle.dump(cache_data, fh)


def _ensure_checker(scene_idx: Optional[int], floor_polygon_xz: np.ndarray,
                    cache: Optional[SDFCache], grid_resolution: float) -> SDFBoundaryChecker:
    if cache is not None and scene_idx is not None:
        data = cache.load(scene_idx)
        if data is not None:
            return SDFBoundaryChecker.from_cache_data(data)
    checker = SDFBoundaryChecker(floor_polygon_xz, grid_resolution=grid_resolution)
    if cache is not None and scene_idx is not None:
        cache.save(scene_idx, checker.get_cache_data())
    return checker


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def compute_oob_reward_sdf(
    parsed_scene: Dict[str, torch.Tensor],
    floor_polygons: Sequence[np.ndarray],
    indices: Optional[Sequence[int]] = None,
    sdf_cache: Optional[SDFCache] = None,
    sdf_cache_dir: Optional[str] = None,
    cache_split: str = "train_val",
    sigma_area: float = 0.05,
    erosion: float = DEFAULT_EROSION_M,
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
    footprint_samples: int = DEFAULT_FOOTPRINT_SAMPLES,
    return_diagnostics: bool = False,
    **kwargs,
):
    """SDF-based, smooth-bounded OOB reward.

    Args:
        parsed_scene: dict from ``parse_and_descale_scenes``.
        floor_polygons: sequence of length B with floor polygon (M, 2) per scene
            in the same world frame as ``positions[:, :, [0,2]]``.
        indices: optional per-scene dataset indices used to address the SDF
            cache. If None, scenes are addressed by batch position only when a
            cache is provided.
        sdf_cache: pre-built ``SDFCache`` instance (preferred). If None and
            ``sdf_cache_dir`` is given, a cache is opened lazily.
        sdf_cache_dir: directory containing ``train_val/`` and ``test/`` SDF
            pickles.
        cache_split: split sub-directory to use.
        sigma_area: bounded-reward scale (m^2). A normalised violation of
            sigma yields reward ~ 0.37.
        erosion: half-extent erosion (m) to forgive geometry float noise.
        grid_resolution: resolution used if SDFs need to be computed.
        footprint_samples: K so that K*K points are queried per object.
        return_diagnostics: if True, also return per-scene diagnostics.

    Returns:
        rewards: (B,) tensor in (0, 1].
        diagnostics (optional): list of dicts.
    """
    positions = parsed_scene["positions"]
    sizes = parsed_scene["sizes"]
    orientations = parsed_scene["orientations"]
    is_empty = parsed_scene["is_empty"]
    device = parsed_scene["device"]

    B, N = positions.shape[0], positions.shape[1]
    rewards = torch.zeros(B, device=device)

    if sdf_cache is None and sdf_cache_dir is not None:
        sdf_cache = SDFCache(sdf_cache_dir, split=cache_split,
                             grid_resolution=grid_resolution, lazy=True)

    pos_np = positions.detach().cpu().numpy()
    siz_np = sizes.detach().cpu().numpy()
    ori_np = orientations.detach().cpu().numpy()
    emp_np = is_empty.detach().cpu().numpy()

    diagnostics: List[Dict] = []

    for b in range(B):
        floor = np.asarray(floor_polygons[b], dtype=np.float64)
        if floor.ndim != 2 or floor.shape[0] < 3:
            rewards[b] = 0.0
            diagnostics.append({"total_outside_area": float("inf"), "per_object": []})
            continue
        try:
            checker = _ensure_checker(
                scene_idx=(int(indices[b]) if indices is not None else None),
                floor_polygon_xz=floor,
                cache=sdf_cache,
                grid_resolution=grid_resolution,
            )
        except Exception:
            rewards[b] = 0.0
            diagnostics.append({"total_outside_area": float("inf"), "per_object": []})
            continue

        per_obj: List[Dict] = []
        total_outside_area = 0.0
        num_active = 0
        for n in range(N):
            if emp_np[b, n]:
                continue
            num_active += 1
            pts, cell_area = _sample_footprint_xz(
                pos_np[b, n, 0], pos_np[b, n, 2],
                siz_np[b, n, 0], siz_np[b, n, 2],
                ori_np[b, n, 0], ori_np[b, n, 1],
                erosion=erosion, k=footprint_samples,
            )
            if pts.shape[0] == 0:
                per_obj.append({"index": n, "outside_area": 0.0})
                continue
            sdf = checker.query_sdf(pts)
            outside = np.clip(-sdf, 0.0, None)
            outside_area = float((outside > 0).astype(np.float64).sum() * cell_area)
            total_outside_area += outside_area
            per_obj.append({"index": n, "outside_area": outside_area})

        denom = max(1, num_active)
        violation = total_outside_area / denom
        rewards[b] = float(np.exp(-violation / max(sigma_area, 1e-6)))
        diagnostics.append({
            "total_outside_area": total_outside_area,
            "per_object": per_obj,
            "num_active": num_active,
        })

    if return_diagnostics:
        return rewards, diagnostics
    return rewards


def compute_boundary_violation_reward(
    parsed_scene: Dict[str, torch.Tensor],
    floor_polygons,
    area_tol: float = 1e-5,  # legacy arg, ignored by the smooth path
    erosion: float = DEFAULT_EROSION_M,
    use_area_ratio: bool = False,  # legacy arg, ignored
    indices: Optional[Sequence[int]] = None,
    sdf_cache: Optional[SDFCache] = None,
    sdf_cache_dir: Optional[str] = None,
    cache_split: str = "train_val",
    sigma_area: float = 0.05,
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
    footprint_samples: int = DEFAULT_FOOTPRINT_SAMPLES,
    return_diagnostics: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Backward-compatible name. Routes to the SDF-based smooth reward.

    Returns rewards in (0, 1] suitable for RL.
    """
    return compute_oob_reward_sdf(
        parsed_scene=parsed_scene,
        floor_polygons=floor_polygons,
        indices=indices,
        sdf_cache=sdf_cache,
        sdf_cache_dir=sdf_cache_dir,
        cache_split=cache_split,
        sigma_area=sigma_area,
        erosion=erosion,
        grid_resolution=grid_resolution,
        footprint_samples=footprint_samples,
        return_diagnostics=return_diagnostics,
    )


# ---------------------------------------------------------------------------
# Precompute helpers (kept from the previous module, in slimmed form)
# ---------------------------------------------------------------------------


def _compute_single_sdf(args):
    idx, floor_verts, grid_resolution = args
    try:
        checker = SDFBoundaryChecker(
            floor_vertices=np.asarray(floor_verts).tolist(),
            grid_resolution=grid_resolution,
        )
        return idx, checker.get_cache_data(), None
    except Exception as e:
        return idx, None, str(e)


def precompute_sdf_cache_from_polygons(
    polygons: Sequence[np.ndarray],
    out_dir: str,
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
    num_workers: Optional[int] = None,
    verbose: bool = True,
):
    """Compute and save an SDF per polygon under ``out_dir`` (no nested split)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metadata.pkl", "wb") as fh:
        pickle.dump({"num_scenes": len(polygons),
                     "grid_resolution": grid_resolution}, fh)

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) // 2)

    args = [(i, p, grid_resolution) for i, p in enumerate(polygons)]
    it = tqdm(total=len(args), desc="precomputing SDFs") if verbose else None
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for fut in as_completed([ex.submit(_compute_single_sdf, a) for a in args]):
            idx, data, err = fut.result()
            if data is not None:
                with open(out / f"sdf_{idx}.pkl", "wb") as fh:
                    pickle.dump(data, fh)
            if it is not None:
                it.update(1)
    if it is not None:
        it.close()
