import os
import pickle

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

_CEILING_OBJECTS = ["ceiling_lamp", "pendant_lamp"]
_IDX_TO_LABELS = {
    "bedroom": {
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
}


def compute_accessibility_reward(
    parsed_scenes: Dict[str, torch.Tensor],
    indices: List[int],  # (B,) - indices to lookup cached grids
    accessibility_cache=None,  
    grid_resolution: float = 0.1,
    agent_radius: float = 0.3,
    save_viz: bool = False,  # NEW: option to save visualizations
    viz_dir: str = "./viz",  # NEW: directory for visualizations
    floor_polygons: torch.Tensor=None,  # (B, num_vertices, 2)
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Compute accessibility reward using cached floor grids or computing on-the-fly.

    Returns dict with 3 components:
    - coverage_ratio: [0, 1] - fraction of floor reachable from largest region
    - num_regions: [1, ∞) - number of disconnected regions
    - avg_clearance: meters - average distance to nearest obstacle in reachable area

    Args:
        parsed_scenes: Dictionary with positions, sizes, is_empty, device, object_types
        floor_polygons: (B, num_vertices, 2) - floor polygon vertices
        is_val: Whether this is validation split
        indices: (B,) - scene indices for cache lookup
        accessibility_cache: Pre-loaded AccessibilityCache instance (optional)
        grid_resolution: Grid resolution in meters (default 0.2m = 20cm)
        agent_radius: Agent radius in meters (default 0.15m = 15cm)
        save_viz: Whether to save visualization images
        viz_dir: Directory to save visualizations

    Returns:
        Dictionary with:
        - 'coverage_ratio': (B,) - reachable area ratio [0, 1]
        - 'num_regions': (B,) - number of disconnected regions [1, ∞)
        - 'avg_clearance': (B,) - average clearance in meters
    """
    room_type = kwargs["room_type"]
    positions = parsed_scenes["positions"]  # (B, N, 3)
    sizes = parsed_scenes["sizes"]  # (B, N, 3) - half-extents
    is_empty = parsed_scenes["is_empty"]  # (B, N)
    orientations = parsed_scenes["orientations"]  # (B, N, 2) - [cos_theta, sin_theta]
    device = parsed_scenes["device"]
    labels_map = kwargs.get("idx_to_labels") or _IDX_TO_LABELS.get(room_type, _IDX_TO_LABELS["bedroom"])
    ceiling_indices = [
        int(idx) for idx, label in labels_map.items() if label in _CEILING_OBJECTS
    ]
    B, N = positions.shape[0], positions.shape[1]
    # Create viz directory if needed
    if save_viz:
        os.makedirs(viz_dir, exist_ok=True)

    coverage_ratios = torch.zeros(B, device=device)
    num_regions = torch.zeros(B, device=device)
    avg_clearances = torch.zeros(B, device=device)

    for b in range(B):
        scene_idx = indices[b] if indices is not None else b
        # Try to load from cache first
        grid_data = None
        if accessibility_cache is not None:
            grid_data = accessibility_cache.load(scene_idx)

        # If cache miss, compute on-the-fly
        if grid_data is None:
            raise ValueError(f"Cache miss for scene {scene_idx} and on-the-fly computation is disabled.")
            print("cache miss for scene ", scene_idx, ", computing on-the-fly")
            try:
                try:
                    floor_verts = floor_polygons[b].cpu().numpy()
                except:
                    floor_verts = np.array(floor_polygons[b])
                # Remove padding vertices (marked as -1000 or similar)
                valid_mask = np.all(np.abs(floor_verts) < 999, axis=1)
                floor_verts = floor_verts[valid_mask]

                if len(floor_verts) < 3:
                    print(f"Warning: Scene {scene_idx} has invalid floor polygon")
                    coverage_ratios[b] = 0.0
                    num_regions[b] = 1.0  # Reduced penalty
                    avg_clearances[b] = 0.0
                    continue

                # Compute grid on-the-fly
                checker = AccessibilityGridBuilder(
                    floor_vertices=floor_verts.tolist(), grid_resolution=grid_resolution
                )
                grid_data = checker.get_cache_data()

            except Exception as e:
                print(
                    f"Warning: Failed to compute accessibility grid for scene {scene_idx}: {e}"
                )
                coverage_ratios[b] = 0.0
                num_regions[b] = 1.0  # Reduced penalty
                avg_clearances[b] = 0.0
                continue

        # Start with empty floor grid (1 = free, 0 = occupied/outside)
        occupancy_grid = grid_data["floor_grid"].copy()
        x_coords = grid_data["x_coords"]
        z_coords = grid_data["z_coords"]

        # Mark occupied cells (inflate by agent_radius)
        for n in range(N):
            if is_empty[b, n]:
                continue

            # FILTER OUT CEILING OBJECTS - they don't occupy floor space
            obj_idx = parsed_scenes["object_indices"][b, n]
            if int(obj_idx) in ceiling_indices:
                continue  # Skip ceiling objects

            obj_pos = positions[b, n].cpu().detach().numpy()
            obj_size = sizes[b, n].cpu().detach().numpy()

            # Get orientation
            cos_theta = orientations[b, n, 0].cpu().detach().numpy()
            sin_theta = orientations[b, n, 1].cpu().detach().numpy()

            # Object center and half-extents
            x_center, z_center = obj_pos[0], obj_pos[2]
            x_half = obj_size[0]
            z_half = obj_size[2]

            # Create 4 corners of oriented bounding box (before inflation)
            corners_local = np.array(
                [
                    [-x_half, -z_half],
                    [+x_half, -z_half],
                    [+x_half, +z_half],
                    [-x_half, +z_half],
                ]
            )

            # Rotation matrix
            R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

            # Rotate corners to world coordinates
            corners_world = corners_local @ R.T + np.array([x_center, z_center])

            # Find axis-aligned bounding box of rotated corners (then inflate by agent_radius)
            x_min = corners_world[:, 0].min() - agent_radius
            x_max = corners_world[:, 0].max() + agent_radius
            z_min = corners_world[:, 1].min() - agent_radius
            z_max = corners_world[:, 1].max() + agent_radius

            # Find grid cells to mark as occupied
            x_min_idx = np.searchsorted(x_coords, x_min, side="left")
            x_max_idx = np.searchsorted(x_coords, x_max, side="right")
            z_min_idx = np.searchsorted(z_coords, z_min, side="left")
            z_max_idx = np.searchsorted(z_coords, z_max, side="right")

            # Clamp to grid bounds
            x_min_idx = max(0, x_min_idx)
            x_max_idx = min(len(x_coords), x_max_idx)
            z_min_idx = max(0, z_min_idx)
            z_max_idx = min(len(z_coords), z_max_idx)

            occupancy_grid[z_min_idx:z_max_idx, x_min_idx:x_max_idx] = 0

        # STEP 1: Count total free cells
        total_free = occupancy_grid.sum()

        if total_free == 0:
            # Completely blocked
            coverage_ratios[b] = 0.0
            num_regions[b] = 1.0  # Reduced penalty
            avg_clearances[b] = 0.0
            continue

        # STEP 2: Find all connected regions and pick largest
        regions = _find_all_regions(occupancy_grid)

        if len(regions) == 0:
            coverage_ratios[b] = 0.0
            num_regions[b] = 1.0  # Reduced penalty
            avg_clearances[b] = 0.0
            continue

        # Find largest region
        largest_region = max(regions, key=lambda r: len(r["cells"]))

        # Get guaranteed-free starting point from largest region
        start_z, start_x = largest_region["center"]

        # STEP 3: Flood fill from largest region to find reachable area
        reachable = _flood_fill_bfs(occupancy_grid, start_z, start_x)
        reachable_count = reachable.sum()
        coverage_ratio = reachable_count / total_free

        # STEP 4: Count total connected regions
        num_disconnected_regions = len(regions)

        # STEP 5: Distance transform (clearance map)
        dist_grid = distance_transform_edt(occupancy_grid)
        dist_meters = dist_grid * grid_resolution

        # STEP 6: Extract reachable distances
        reachable_distances = dist_meters[reachable]
        avg_clearance = (
            reachable_distances.mean() if len(reachable_distances) > 0 else 0.0
        )

        # Store results
        coverage_ratios[b] = coverage_ratio
        num_regions[b] = num_disconnected_regions
        avg_clearances[b] = avg_clearance

        # SAVE VISUALIZATION
        if save_viz:
            _save_region_visualization(
                scene_idx=scene_idx,
                occupancy_grid=occupancy_grid,
                regions=regions,
                reachable=reachable,
                viz_dir=viz_dir,
                x_coords=x_coords,
                z_coords=z_coords,
            )

    # print(
    #     f"Accessibility rewards - Coverage: {coverage_ratios.mean().item():.3f}, Regions: {num_regions.mean().item():.1f}, Clearance: {avg_clearances.mean().item():.2f}m"
    # )

    return 1 * coverage_ratios - 0.1 * num_regions + 0.5 * avg_clearances


def _save_region_visualization(
    scene_idx: int,
    occupancy_grid: np.ndarray,
    regions: List[Dict],
    reachable: np.ndarray,
    viz_dir: str,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
):
    """
    Save visualization showing:
    - Each disconnected region in a different color
    - Reachable area highlighted
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: All regions with different colors
    ax1 = axes[0]
    region_map = np.zeros_like(occupancy_grid, dtype=np.int32)

    # Assign each region a unique ID
    for region_id, region in enumerate(regions, start=1):
        for z, x in region["cells"]:
            region_map[z, x] = region_id

    # Create colormap with distinct colors
    num_regions = len(regions)
    cmap = plt.cm.get_cmap("tab20", num_regions + 1)

    # Show regions
    im1 = ax1.imshow(
        region_map,
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
        extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
    )
    ax1.set_title(
        f"Scene {scene_idx}: All Regions (Total: {num_regions})",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Z (meters)")
    plt.colorbar(im1, ax=ax1, label="Region ID")

    # Plot 2: Reachable area
    ax2 = axes[1]
    reachable_viz = np.zeros_like(occupancy_grid, dtype=float)
    reachable_viz[occupancy_grid == 1] = 0.3  # Light gray for all free space
    reachable_viz[reachable] = 1.0  # Bright for reachable

    im2 = ax2.imshow(
        reachable_viz,
        cmap="RdYlGn",
        interpolation="nearest",
        origin="upper",
        extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
        vmin=0,
        vmax=1,
    )

    coverage = reachable.sum() / (
        occupancy_grid.sum() if occupancy_grid.sum() > 0 else 1
    )
    ax2.set_title(
        f"Scene {scene_idx}: Reachable Area (Coverage: {coverage:.1%})",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Z (meters)")
    plt.colorbar(im2, ax=ax2, label="Reachability")

    plt.tight_layout()
    viz_path = os.path.join(viz_dir, f"scene_{scene_idx}_accessibility.png")
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved visualization: {viz_path}")


def _find_all_regions(grid: np.ndarray) -> List[Dict]:
    """
    Find all connected regions in the occupancy grid.
    Returns list of regions with their cells and guaranteed-free center point.
    """
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    regions = []

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1 and not visited[i, j]:
                region = _flood_fill_region(grid, visited, i, j)
                regions.append(region)

    return regions


def _flood_fill_region(
    grid: np.ndarray, visited: np.ndarray, start_z: int, start_x: int
) -> Dict:
    """
    Flood fill a single region and return its cells and a guaranteed-free center point.
    """
    rows, cols = grid.shape
    queue = deque([(start_z, start_x)])
    cells = []
    visited[start_z, start_x] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        z, x = queue.popleft()
        cells.append((z, x))

        for dz, dx in directions:
            nz, nx = z + dz, x + dx

            if (
                0 <= nz < rows
                and 0 <= nx < cols
                and grid[nz, nx] == 1
                and not visited[nz, nx]
            ):
                visited[nz, nx] = True
                queue.append((nz, nx))

    # Calculate centroid
    sum_z = sum(z for z, x in cells)
    sum_x = sum(x for z, x in cells)
    center_z = round(sum_z / len(cells))
    center_x = round(sum_x / len(cells))

    # CRITICAL: Ensure center is actually a free cell
    if grid[center_z, center_x] != 1:
        # Find closest actual free cell from this region
        min_dist = float("inf")
        best_z, best_x = cells[0]  # Fallback to first cell

        for z, x in cells:
            dist = (z - center_z) ** 2 + (x - center_x) ** 2
            if dist < min_dist:
                min_dist = dist
                best_z, best_x = z, x

        center_z, center_x = best_z, best_x

    return {"cells": cells, "center": (center_z, center_x)}


def _flood_fill_bfs(grid: np.ndarray, start_z: int, start_x: int) -> np.ndarray:
    """
    Fast BFS flood fill to find reachable cells from starting point.

    Args:
        grid: (H, W) occupancy grid (1=free, 0=occupied)
        start_z, start_x: Starting position (guaranteed to be free)

    Returns:
        reachable: (H, W) boolean array of reachable cells
    """
    H, W = grid.shape
    reachable = np.zeros((H, W), dtype=bool)

    if grid[start_z, start_x] == 0:
        return reachable  # Starting point is blocked (should never happen)

    queue = deque([(start_z, start_x)])
    reachable[start_z, start_x] = True

    # 4-connected neighbors
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        z, x = queue.popleft()

        for dz, dx in directions:
            nz, nx = z + dz, x + dx

            if (
                0 <= nz < H
                and 0 <= nx < W
                and grid[nz, nx] > 0
                and not reachable[nz, nx]
            ):
                reachable[nz, nx] = True
                queue.append((nz, nx))

    return reachable


def _compute_single_accessibility_grid(args):
    """Worker function for parallel grid computation."""
    idx, floor_verts, grid_resolution, split_name = args
    try:
        checker = AccessibilityGridBuilder(
            floor_vertices=floor_verts.tolist(), grid_resolution=grid_resolution
        )
        return idx, checker.get_cache_data(), None
    except Exception as e:
        return idx, None, str(e)


def precompute_accessibility_cache(
    config,
    accessibility_cache_dir: str = "./accessibility_cache",
    grid_resolution: float = 0.1,
    verbose: bool = True,
    num_workers: int = None,
):
    """
    Precompute and save floor grids for all scenes in dataset.
    Uses multiprocessing for fast parallel computation.

    **Call this ONCE before training** to generate the cache!

    Args:
        config: Config object with dataset parameters
        accessibility_cache_dir: Directory to save cached grids
        grid_resolution: Grid resolution in meters (default 0.2m = 20cm)
        verbose: Print progress
        num_workers: Number of parallel workers (None = all CPUs)

    Example:
        # Before training
        precompute_accessibility_cache(config, accessibility_cache_dir="./accessibility_cache", num_workers=32)
    """
    from steerable_scene_generation.datasets.custom_scene import CustomDataset

    if num_workers is None:
        num_workers = os.cpu_count()

    # Precompute for train/val split
    train_val_dir = os.path.join(accessibility_cache_dir, "train_val")
    os.makedirs(train_val_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print(
            f"Precomputing Accessibility cache for TRAIN/VAL split (using {num_workers} workers)..."
        )
        print("=" * 60)

    train_val_dataset = CustomDataset(
        cfg=config.dataset,
        split=["train", "val"],
        ckpt_path=None,
    )

    _precompute_accessibility_split(
        dataset=train_val_dataset,
        cache_dir=train_val_dir,
        grid_resolution=grid_resolution,
        split_name="train_val",
        verbose=verbose,
        num_workers=num_workers,
    )

    # Precompute for test split
    test_dir = os.path.join(accessibility_cache_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 60)
        print(
            f"Precomputing Accessibility cache for TEST split (using {num_workers} workers)..."
        )
        print("=" * 60)

    test_dataset = CustomDataset(
        cfg=config.dataset,
        split=["test"],
        ckpt_path=None,
    )

    _precompute_accessibility_split(
        dataset=test_dataset,
        cache_dir=test_dir,
        grid_resolution=grid_resolution,
        split_name="test",
        verbose=verbose,
        num_workers=num_workers,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("✓ Accessibility cache precomputation complete!")
        print(f"  Train/Val: {train_val_dir}")
        print(f"  Test: {test_dir}")
        print("=" * 60)


def _precompute_accessibility_split(
    dataset,
    cache_dir: str,
    grid_resolution: float,
    split_name: str,
    verbose: bool,
    num_workers: int,
):
    """Internal helper to precompute accessibility cache for a single split using parallel processing."""
    floor_polygons_list = [
        dataset.get_floor_polygon_points(idx) for idx in range(len(dataset))
    ]

    # Save metadata
    metadata = {
        "num_scenes": len(floor_polygons_list),
        "grid_resolution": grid_resolution,
        "split": split_name,
    }
    with open(os.path.join(cache_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    failed_scenes = []

    # Prepare arguments for parallel processing
    compute_args = [
        (idx, floor_verts, grid_resolution, split_name)
        for idx, floor_verts in enumerate(floor_polygons_list)
    ]

    if verbose:
        print(
            f"[{split_name}] Computing {len(floor_polygons_list)} accessibility grids with {num_workers} workers..."
        )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_compute_single_accessibility_grid, args): args[0]
            for args in compute_args
        }

        if verbose:
            progress_bar = tqdm(
                total=len(futures), desc=f"[{split_name}] Precomputing Grids"
            )

        for future in as_completed(futures):
            idx, cache_data, error = future.result()

            if error is not None:
                print(f"ERROR: [{split_name}] Scene {idx} failed: {error}")
                failed_scenes.append(idx)
            elif cache_data is not None:
                # Save to disk
                cache_path = os.path.join(cache_dir, f"grid_{idx}.pkl")
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)

            if verbose:
                progress_bar.update(1)

        if verbose:
            progress_bar.close()

    if verbose:
        print(
            f"✓ [{split_name}] Precomputed {len(floor_polygons_list)} grids in {cache_dir}"
        )
        if failed_scenes:
            print(
                f"⚠ [{split_name}] {len(failed_scenes)} scenes had issues: {failed_scenes[:10]}..."
            )

    if failed_scenes:
        with open(os.path.join(cache_dir, "failed_scenes.txt"), "w") as f:
            f.write("\n".join(map(str, failed_scenes)))


class AccessibilityCache:
    """Manages loading/saving of cached accessibility grids. Loads all data into RAM on init."""

    def __init__(
        self, cache_dir: str, grid_resolution: float = 0.1, split: str = "train_val"
    ):
        """
        Args:
            cache_dir: Base cache directory (e.g., "./accessibility_cache")
            grid_resolution: Grid resolution for validation
            split: Either "train_val" or "test"
        """
        self.base_cache_dir = Path(cache_dir)
        self.grid_resolution = grid_resolution
        self.split = split

        # Determine actual cache directory based on split
        if split in ["train", "val", "train_val"]:
            self.cache_dir = self.base_cache_dir / "train_val"
        elif split == "test":
            self.cache_dir = self.base_cache_dir / "test"
        else:
            raise ValueError(
                f"Unknown split: {split}. Expected 'train', 'val', 'train_val', or 'test'"
            )

        # Verify cache metadata
        metadata_path = self.cache_dir / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            if metadata["grid_resolution"] != grid_resolution:
                print(
                    f"Warning: Cache resolution {metadata['grid_resolution']} != requested {grid_resolution}"
                )
        else:
            print(
                f"Warning: No metadata found at {metadata_path}. Cache may not exist."
            )

        # Load all grid data into memory
        self._memory_cache = {}
        self._load_all_to_memory()

    def _load_all_to_memory(self):
        """Load all cached grids into RAM for fast access."""
        if not self.cache_dir.exists():
            print(
                f"Warning: Cache directory {self.cache_dir} does not exist. No data loaded."
            )
            return

        grid_files = sorted(self.cache_dir.glob("grid_*.pkl"))
        print(
            f"Loading {len(grid_files)} accessibility grids into memory from {self.cache_dir}..."
        )

        for grid_file in tqdm(
            grid_files, desc=f"Loading {self.split} accessibility grids to RAM"
        ):
            scene_idx = int(grid_file.stem.split("_")[1])

            with open(grid_file, "rb") as f:
                self._memory_cache[scene_idx] = pickle.load(f)

        # Calculate memory usage
        total_bytes = sum(
            data["floor_grid"].nbytes
            + data["x_coords"].nbytes
            + data["z_coords"].nbytes
            + len(pickle.dumps(data["world_bounds"]))
            for data in self._memory_cache.values()
        )
        print(
            f"✓ Loaded {len(self._memory_cache)} accessibility grids into RAM (~{total_bytes / 1024 / 1024:.1f} MB)"
        )

    def load(self, scene_idx: int) -> Optional[Dict]:
        """Load cached grid data for scene_idx from memory."""
        return self._memory_cache.get(int(scene_idx), None)

    def save(self, scene_idx: int, cache_data: Dict):
        """Save grid data for scene_idx to memory and disk."""
        self._memory_cache[scene_idx] = cache_data

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"grid_{scene_idx}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)


class AccessibilityGridBuilder:
    """Builds floor occupancy grids for accessibility checking."""

    def __init__(self, floor_vertices, grid_resolution=0.1):
        self.floor_vertices = np.array(floor_vertices, dtype=np.float32)
        self.grid_resolution = grid_resolution

        # Validate polygon
        if len(self.floor_vertices) < 3:
            raise ValueError(
                f"Polygon must have at least 3 vertices, got {len(self.floor_vertices)}"
            )

        # Compute bounds with padding
        padding = max(1.0, self.grid_resolution * 5)
        min_x = self.floor_vertices[:, 0].min() - padding
        max_x = self.floor_vertices[:, 0].max() + padding
        min_z = self.floor_vertices[:, 1].min() - padding
        max_z = self.floor_vertices[:, 1].max() + padding

        self.world_bounds = (min_x, max_x, min_z, max_z)
        self.floor_grid, self.x_coords, self.z_coords = self._compute_floor_grid()

    def get_cache_data(self) -> Dict:
        """Get data to save to cache."""
        return {
            "floor_grid": self.floor_grid,
            "world_bounds": self.world_bounds,
            "grid_resolution": self.grid_resolution,
            "x_coords": self.x_coords,
            "z_coords": self.z_coords,
        }

    def _point_in_polygon(self, point: np.ndarray) -> bool:
        """Ray casting algorithm to check if point is inside polygon."""
        x, z = point
        vertices = self.floor_vertices
        n = len(vertices)
        inside = False

        j = n - 1
        for i in range(n):
            xi, zi = vertices[i]
            xj, zj = vertices[j]

            if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
                inside = not inside
            j = i

        return inside

    def _compute_floor_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute binary floor grid (1=inside floor, 0=outside)."""
        min_x, max_x, min_z, max_z = self.world_bounds

        x_coords = np.arange(min_x, max_x, self.grid_resolution)
        z_coords = np.arange(min_z, max_z, self.grid_resolution)

        floor_grid = np.zeros((len(z_coords), len(x_coords)), dtype=np.uint8)

        for i, z in enumerate(z_coords):
            for j, x in enumerate(x_coords):
                point = np.array([x, z])
                if self._point_in_polygon(point):
                    floor_grid[i, j] = 1

        return floor_grid, x_coords, z_coords


if __name__ == "__main__":
    args = np.load(
        "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/reward_func_args_for_first_10_scenes.npy",
        allow_pickle=True,
    )
    # print("loaded ", args)
    start = 0
    end = 15
    # for key in args.item().keys():
    #     if isinstance(args.item()[key], np.ndarray):
    #         args.item()[key] = args.item()[key][start:end]
    #     elif isinstance(args.item()[key], list):
    #         args.item()[key] = args.item()[key][start:end]
    #     print(f"{key}: {args.item()[key].shape if hasattr(args.item()[key], 'shape') else len(args.item()[key])}")
    # Enable visualization for testing
    result = compute_accessibility_reward(
        **args.item(),
        # save_viz=True, viz_dir="./viz"
    )
    print(result[start:end])
