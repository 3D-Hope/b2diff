"""Generate 3-D scene layouts with optional reward-based rejection sampling.

This script mirrors ashok_generate_results.py and adds a rejection mode:
- If --reward_file and --reward_threshold are provided, every generated batch is
  scored by the reward function and only accepted scenes are kept.
- Multiple custom reward stages can be applied sequentially via
    --reward_files/--reward_thresholds.
- Generation continues until accepted scenes >= --target_accepted.

Output format remains identical: a ThreedFrontResults object is saved to
results.pkl so downstream evaluation code works unchanged.
"""

import argparse
import importlib.util
import os
import sys
import shutil
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from shapely.geometry import Polygon

from utils import PROJ_DIR, load_config, update_data_file_paths
from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults
from midiffusion.datasets.threed_front_encoding import (
    get_dataset_raw_and_encoded,
    get_encoded_dataset,
)
from midiffusion.ashok_midiffusion import SceneDiffuserMiDiffusion

# Reuse the exact generation implementation to ensure compatibility.
from ashok_generate_results import generate_ashok_layouts


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UNIVERSAL_REWARD_DIR = REPO_ROOT / "core" / "universal_rewards"


def _load_callable_from_python_file(module_path: Path, function_name: str) -> Callable:
    """Load a named callable from a python file without importing parent packages."""
    if not module_path.is_file():
        raise FileNotFoundError(f"Module file not found: {module_path}")

    module_name = f"dynamic_module_{abs(hash(str(module_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"Function '{function_name}' not found in {module_path}")
    return fn


compute_boundary_violation_reward = _load_callable_from_python_file(
    UNIVERSAL_REWARD_DIR / "not_out_of_bound_reward.py",
    "compute_boundary_violation_reward",
)
compute_non_penetration_reward = _load_callable_from_python_file(
    UNIVERSAL_REWARD_DIR / "penetration_reward.py",
    "compute_non_penetration_reward",
)


def _cfg_get(container, key, default=None):
    """Read a config field from dict-like or attribute-like containers."""
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    if hasattr(container, "get"):
        try:
            value = container.get(key, default)
            return default if value is None else value
        except Exception:
            pass
    return getattr(container, key, default)


def _load_reward_function_from_file(reward_file: str) -> Callable:
    """Load a reward callable from a python file.

    Preferred entrypoint is compute_reward(parsed_scene, **kwargs).
    Falls back to a unique callable starting with compute_.
    """
    if not os.path.isfile(reward_file):
        raise FileNotFoundError(f"Reward file not found: {reward_file}")

    module_name = f"reward_module_{abs(hash(os.path.abspath(reward_file)))}"
    spec = importlib.util.spec_from_file_location(module_name, reward_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {reward_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "compute_reward") and callable(module.compute_reward):
        return module.compute_reward

    compute_like = [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("compute_") and callable(getattr(module, name))
    ]
    if len(compute_like) == 1:
        return compute_like[0]

    raise ValueError(
        "Reward file must define compute_reward(parsed_scene, **kwargs), "
        "or exactly one compute_* callable."
    )


def _slice_reward_kwargs_by_positions(
    reward_kwargs: Dict,
    base_batch_size: int,
    keep_positions: List[int],
) -> Dict:
    """Slice list-valued reward kwargs that are batch-aligned."""
    return {
        key: [value[i] for i in keep_positions]
        if isinstance(value, list) and len(value) == base_batch_size
        else value
        for key, value in reward_kwargs.items()
    }


def _build_reward_kwargs_for_batch(
    config,
    idx_to_labels: Dict[int, str],
    sampled_indices: List[int],
    raw_dataset,
) -> Dict:
    """Mirror selection.py custom reward kwargs for one generated batch."""
    midiffusion_cfg = _cfg_get(config, "midiffusion", None)
    reward_kwargs = {
        "room_type": _cfg_get(midiffusion_cfg, "room_type", "bedroom"),
        "idx_to_labels": idx_to_labels,
    }

    try:
        floor_entries = [
            _extract_floor_geometry_from_dataset_scene(raw_dataset[int(idx)], int(idx))
            for idx in sampled_indices
        ]
    except Exception as exc:
        raise ValueError(
            "Could not extract floor geometry from dataset for sampled indices. "
            "Ensure raw_dataset and encoded_dataset use the same split and ordering."
        ) from exc

    reward_kwargs["floor_plan_vertices"] = [
        entry["floor_plan_vertices"] for entry in floor_entries
    ]
    reward_kwargs["floor_plan_faces"] = [
        entry["floor_plan_faces"] for entry in floor_entries
    ]
    reward_kwargs["floor_plan_centroid"] = [
        entry["floor_plan_centroid"] for entry in floor_entries
    ]

    return reward_kwargs


def _extract_floor_geometry_from_dataset_scene(scene, scene_index: int) -> Dict[str, List]:
    """Extract floor geometry (vertices/faces/centroid) from a raw dataset scene."""
    vertices = None
    faces = None
    centroid = None

    if hasattr(scene, "floor_plan_vertices") and hasattr(scene, "floor_plan_faces"):
        vertices = np.asarray(scene.floor_plan_vertices, dtype=np.float64)
        faces = np.asarray(scene.floor_plan_faces, dtype=np.int64)
        centroid = np.asarray(getattr(scene, "floor_plan_centroid", None), dtype=np.float64)
    elif hasattr(scene, "floor_plan"):
        floor_vertices, floor_faces = scene.floor_plan
        vertices = np.asarray(floor_vertices, dtype=np.float64)
        faces = np.asarray(floor_faces, dtype=np.int64)
        if hasattr(scene, "floor_plan_centroid"):
            centroid = np.asarray(scene.floor_plan_centroid, dtype=np.float64)
    elif isinstance(scene, dict):
        if "floor_plan_vertices" in scene and "floor_plan_faces" in scene:
            vertices = np.asarray(scene["floor_plan_vertices"], dtype=np.float64)
            faces = np.asarray(scene["floor_plan_faces"], dtype=np.int64)
            if "floor_plan_centroid" in scene:
                centroid = np.asarray(scene["floor_plan_centroid"], dtype=np.float64)

    if vertices is None or faces is None:
        raise ValueError(
            f"Dataset scene {scene_index} does not expose floor_plan_vertices/faces."
        )

    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError(
            f"Invalid floor_plan_vertices for scene {scene_index}: shape={vertices.shape}"
        )

    if centroid is None or centroid.shape[0] < 3:
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        centroid = (bbox_min + bbox_max) / 2.0

    return {
        "floor_plan_vertices": vertices.astype(np.float64).tolist(),
        "floor_plan_faces": faces.astype(np.int64).tolist(),
        "floor_plan_centroid": centroid.astype(np.float64).tolist(),
    }


def _build_floor_polygon_xz_lookup_from_dataset(raw_dataset) -> Dict[int, List[List[float]]]:
    """Build index -> xz boundary polygon directly from raw dataset floor geometry."""
    lookup = {}
    for idx in range(len(raw_dataset)):
        entry = _extract_floor_geometry_from_dataset_scene(raw_dataset[idx], idx)
        vertices = np.asarray(entry["floor_plan_vertices"], dtype=np.float64)
        faces = np.asarray(entry["floor_plan_faces"], dtype=np.int64)
        centroid = np.asarray(entry["floor_plan_centroid"], dtype=np.float64)

        # Object translations are centroid-centered in encoded data, so center floor geometry.
        centered_vertices = vertices - centroid[None, :]

        poly = None
        if faces.ndim == 2 and faces.shape[1] >= 3 and len(faces) > 0:
            tri_polys = []
            for tri in faces:
                tri_indices = tri[:3]
                tri_vertices = centered_vertices[tri_indices][:, [0, 2]]
                tri_poly = Polygon(tri_vertices.tolist())
                if tri_poly.is_valid and tri_poly.area > 1e-12:
                    tri_polys.append(tri_poly)
            if tri_polys:
                # Avoid shapely vectorized union API incompatibilities across versions.
                try:
                    poly = tri_polys[0]
                    for next_poly in tri_polys[1:]:
                        poly = poly.union(next_poly)
                except Exception:
                    poly = None

        if poly is None or poly.is_empty:
            poly = Polygon(centered_vertices[:, [0, 2]].tolist()).convex_hull

        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)

        if not poly.is_valid or poly.area <= 1e-12:
            raise ValueError(f"Could not reconstruct valid floor polygon for scene {idx}.")

        lookup[idx] = [[float(x), float(z)] for x, z in list(poly.exterior.coords)[:-1]]

    return lookup


def _angles_to_orientation(angles: np.ndarray) -> np.ndarray:
    """Convert angle representation to 2D orientation [cos, sin].

    Supported input shapes for angles:
    - (N,) or (N, 1): radians
    - (N, 2): already [cos, sin]
    """
    if angles.ndim == 1:
        return np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    if angles.ndim == 2 and angles.shape[-1] == 1:
        a = angles[:, 0]
        return np.stack([np.cos(a), np.sin(a)], axis=-1)

    if angles.ndim == 2 and angles.shape[-1] == 2:
        # Normalize to avoid small numeric drift.
        norm = np.linalg.norm(angles, axis=-1, keepdims=True)
        norm = np.clip(norm, 1e-8, None)
        return angles / norm

    raise ValueError(f"Unsupported angles shape: {angles.shape}")


def _build_parsed_scene_from_layouts(
    layouts: List[Dict[str, np.ndarray]],
    max_objects: int,
    n_object_types: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Convert generated layout dicts into parsed_scene format for rewards."""
    batch_size = len(layouts)

    positions = torch.zeros((batch_size, max_objects, 3), dtype=torch.float32, device=device)
    sizes = torch.zeros((batch_size, max_objects, 3), dtype=torch.float32, device=device)
    orientations = torch.zeros((batch_size, max_objects, 2), dtype=torch.float32, device=device)
    angles = torch.zeros((batch_size, max_objects, 1), dtype=torch.float32, device=device)
    object_indices = torch.zeros((batch_size, max_objects), dtype=torch.long, device=device)
    class_labels = torch.zeros(
        (batch_size, max_objects, n_object_types), dtype=torch.float32, device=device
    )
    is_empty = torch.ones((batch_size, max_objects), dtype=torch.bool, device=device)

    for b, scene in enumerate(layouts):
        scene_positions = np.asarray(scene["translations"], dtype=np.float32)
        scene_sizes = np.asarray(scene["sizes"], dtype=np.float32)
        scene_angles = np.asarray(scene["angles"], dtype=np.float32)
        scene_classes = np.asarray(scene["class_labels"], dtype=np.float32)

        if scene_positions.ndim == 1:
            scene_positions = scene_positions.reshape(1, -1)
        if scene_sizes.ndim == 1:
            scene_sizes = scene_sizes.reshape(1, -1)
        if scene_angles.ndim == 0:
            scene_angles = scene_angles.reshape(1)
        if scene_classes.ndim == 1:
            scene_classes = scene_classes.reshape(1, -1)

        n_obj = int(min(scene_positions.shape[0], max_objects))
        if n_obj == 0:
            continue

        scene_orient = _angles_to_orientation(scene_angles)
        scene_idx = np.argmax(scene_classes, axis=-1).astype(np.int64)

        positions[b, :n_obj] = torch.from_numpy(scene_positions[:n_obj]).to(device)
        sizes[b, :n_obj] = torch.from_numpy(scene_sizes[:n_obj]).to(device)
        orientations[b, :n_obj] = torch.from_numpy(scene_orient[:n_obj]).to(device)

        if scene_angles.ndim == 1:
            scene_angles_col = scene_angles.reshape(-1, 1)
        elif scene_angles.shape[-1] == 1:
            scene_angles_col = scene_angles
        else:
            # If angles are already cos/sin, keep angle channel as zero fallback.
            scene_angles_col = np.zeros((scene_angles.shape[0], 1), dtype=np.float32)

        angles[b, :n_obj] = torch.from_numpy(scene_angles_col[:n_obj]).to(device)
        object_indices[b, :n_obj] = torch.from_numpy(scene_idx[:n_obj]).to(device)
        class_labels[b, :n_obj] = torch.from_numpy(scene_classes[:n_obj]).to(device)
        is_empty[b, :n_obj] = False

    return {
        "device": device,
        "positions": positions,
        "sizes": sizes,
        "orientations": orientations,
        "angles": angles,
        "object_indices": object_indices,
        "class_labels": class_labels,
        "is_empty": is_empty,
    }


def _score_layout_batch(
    layouts: List[Dict[str, np.ndarray]],
    reward_fn: Callable,
    max_objects: int,
    n_object_types: int,
    reward_kwargs: Dict,
    device: torch.device,
) -> torch.Tensor:
    """Run reward function on a generated layout batch and return per-scene rewards."""
    parsed = _build_parsed_scene_from_layouts(
        layouts=layouts,
        max_objects=max_objects,
        n_object_types=n_object_types,
        device=device,
    )

    with torch.no_grad():
        rewards = reward_fn(parsed, **reward_kwargs)

    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    else:
        rewards = rewards.to(device=device, dtype=torch.float32)

    if rewards.ndim == 0:
        rewards = rewards.unsqueeze(0)

    if rewards.shape[0] != len(layouts):
        raise ValueError(
            f"Reward function returned {rewards.shape[0]} values for {len(layouts)} scenes."
        )

    return rewards


def _score_parsed_scene(
    parsed_scene: Dict[str, torch.Tensor],
    reward_fn: Callable,
    reward_kwargs: Dict,
    device: torch.device,
) -> torch.Tensor:
    """Run reward function on already-parsed scene tensors and return per-scene rewards."""
    with torch.no_grad():
        rewards = reward_fn(parsed_scene, **reward_kwargs)

    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    else:
        rewards = rewards.to(device=device, dtype=torch.float32)

    if rewards.ndim == 0:
        rewards = rewards.unsqueeze(0)

    return rewards


def _slice_parsed_scene(parsed_scene: Dict[str, torch.Tensor], keep_indices: List[int]):
    """Slice parsed_scene along batch dimension for selected scene indices."""
    if len(keep_indices) == 0:
        return None

    idx_tensor = torch.tensor(keep_indices, dtype=torch.long, device=parsed_scene["device"])
    sliced = {"device": parsed_scene["device"]}
    for key, value in parsed_scene.items():
        if key == "device":
            continue
        sliced[key] = value.index_select(0, idx_tensor)
    return sliced


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using SceneDiffuserMiDiffusion with optional reward rejection"
    )
    parser.add_argument("weight_file", help="Path to a pretrained model (.pt state-dict)")
    parser.add_argument(
        "--config_file",
        default=None,
        help="Path to experiment config yaml (default: config.yaml next to weight_file)",
    )
    parser.add_argument(
        "--output_directory",
        default=PROJ_DIR + "/output/predicted_results/",
        help="Path to output directory",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling")
    parser.add_argument(
        "--n_syn_scenes",
        default=1000,
        type=int,
        help="Number of scenes to synthesize when rejection is disabled",
    )
    parser.add_argument("--batch_size", default=128, type=int, help="Scenes per generation batch")
    parser.add_argument("--result_tag", default=None, help="Result sub-directory name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--overfit_test",
        action="store_true",
        help="Use first training sample floor plan for every generated scene",
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="Optional LoRA weights (.pt). Applies adapters then loads LoRA weights.",
    )
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=20,
        help="Number of DDIM inference steps (default: 20)",
    )

    # Rejection-sampling args
    parser.add_argument(
        "--reward_file",
        default=None,
        help="Path to a reward python file (e.g., core/custom_rewards/tv_bed.py)",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=None,
        help="Minimum reward to accept a scene",
    )
    parser.add_argument(
        "--reward_files",
        nargs="+",
        default=None,
        help=(
            "Optional list of reward python files to apply sequentially as custom "
            "rejection stages."
        ),
    )
    parser.add_argument(
        "--reward_thresholds",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional list of thresholds for --reward_files, matched by position. "
            "Each stage keeps scenes with reward >= threshold."
        ),
    )
    parser.add_argument(
        "--target_accepted",
        type=int,
        default=4000,
        help="Accepted scene target when rejection is enabled (default: 4000)",
    )
    parser.add_argument(
        "--max_rejection_batches",
        type=int,
        default=10000,
        help="Safety cap on number of generated batches in rejection mode",
    )
    parser.add_argument(
        "--floor_geometry_path",
        default=None,
        help=(
            "Deprecated: floor geometry is now taken from the loaded raw dataset. "
            "This flag is ignored."
        ),
    )
    parser.add_argument(
        "--use_universal_pre_rejection",
        action="store_true",
        help=(
            "Enable universal pre-rejection stages (penetration then boundary). "
            "Disabled by default unless this flag is passed."
        ),
    )
    parser.add_argument(
        "--penetration_threshold",
        type=float,
        default=0.0,
        help="Minimum penetration reward for stage-1 pre-rejection (default: 0.0).",
    )
    parser.add_argument(
        "--boundary_threshold",
        type=float,
        default=0.0,
        help="Minimum boundary reward for stage-2 pre-rejection (default: 0.0).",
    )

    args = parser.parse_args(argv)

    has_single_custom_rejection = args.reward_file is not None or args.reward_threshold is not None
    if has_single_custom_rejection and (args.reward_file is None or args.reward_threshold is None):
        raise ValueError("Both --reward_file and --reward_threshold must be provided together.")

    has_multi_custom_rejection = args.reward_files is not None or args.reward_thresholds is not None
    if has_multi_custom_rejection and (args.reward_files is None or args.reward_thresholds is None):
        raise ValueError("Both --reward_files and --reward_thresholds must be provided together.")
    if args.reward_files is not None and len(args.reward_files) != len(args.reward_thresholds):
        raise ValueError(
            "--reward_files and --reward_thresholds must have the same number of values."
        )

    custom_reward_stages: List[Tuple[str, Callable, float]] = []
    if args.reward_file is not None:
        custom_reward_stages.append(
            (
                args.reward_file,
                _load_reward_function_from_file(args.reward_file),
                float(args.reward_threshold),
            )
        )
    if args.reward_files is not None:
        for reward_file, reward_threshold in zip(args.reward_files, args.reward_thresholds):
            custom_reward_stages.append(
                (
                    reward_file,
                    _load_reward_function_from_file(reward_file),
                    float(reward_threshold),
                )
            )

    has_custom_rejection = len(custom_reward_stages) > 0
    use_universal_pre_rejection = args.use_universal_pre_rejection
    rejection_enabled = use_universal_pre_rejection or has_custom_rejection

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # output directory
    if args.result_tag is None:
        result_dir = args.output_directory
    else:
        result_dir = os.path.join(args.output_directory, args.result_tag)

    if os.path.exists(result_dir) and len(os.listdir(result_dir)) > 0:
        input(
            f"{result_dir} directory is non-empty. "
            "Press any key to remove all files..."
        )
        for fi in os.listdir(result_dir):
            os.remove(os.path.join(result_dir, fi))
    else:
        os.makedirs(result_dir, exist_ok=True)

    path_to_config = os.path.join(result_dir, "config.yaml")
    path_to_results = os.path.join(result_dir, "results.pkl")

    # config
    if args.config_file is None:
        args.config_file = os.path.join(os.path.dirname(args.weight_file), "config.yaml")
    config = load_config(args.config_file)
    if "_eval" not in config["data"]["encoding_type"]:
        config["data"]["encoding_type"] += "_eval"
    if not os.path.exists(path_to_config) or not os.path.samefile(args.config_file, path_to_config):
        shutil.copyfile(args.config_file, path_to_config)

    # raw training dataset (for ThreedFrontResults bookkeeping)
    raw_train_dataset = get_raw_dataset(
        update_data_file_paths(config["data"]),
        split=config["training"].get("splits", ["train", "val"]),
        include_room_mask=config["network"].get("room_mask_condition", True),
    )

    # encoded test dataset
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"]),
        split=["train","val"],
        # split=config["validation"].get("splits", ["test"]),
        max_length=config["network"]["sample_num_points"],
        include_room_mask=config["network"].get("room_mask_condition", True),
    )
    print(
        "Loaded {} scenes with {} object types ({} labels):".format(
            len(encoded_dataset),
            encoded_dataset.n_object_types,
            encoded_dataset.n_classes,
        )
    )
    print(encoded_dataset.class_labels)

    # build model and load baseline weights
    network = SceneDiffuserMiDiffusion()
    state = torch.load(args.weight_file, map_location=device)
    network.load_state_dict(state)
    network.to(device)
    print("Loaded baseline weights from", args.weight_file)

    # optionally apply LoRA and load LoRA weights
    if args.lora is not None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("peft is required for LoRA support. Install with: pip install peft") from exc

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        network = get_peft_model(network, lora_config)

        lora_state = torch.load(args.lora, map_location=device)
        network.load_state_dict(lora_state, strict=False)
        network.to(device)
        print("Loaded LoRA weights from", args.lora)

    network.eval()
    print("Model ready for inference")

    # --overfit_test: grab first training sample's floor plan
    overfit_sample = None
    if args.overfit_test:
        print("[overfit_test] using first training-set floor plan for all scenes.")
        train_encoded = get_encoded_dataset(
            update_data_file_paths(config["data"]),
            path_to_bounds=os.path.join(os.path.dirname(args.weight_file), "bounds.npz"),
            augmentations=None,
            split=config["training"].get("splits", ["train", "val"]),
            max_length=config["network"]["sample_num_points"],
            include_room_mask=config["network"].get("room_mask_condition", True),
        )
        first = train_encoded[0]
        overfit_sample = {
            k: torch.from_numpy(v)[None]
            for k, v in first.items()
            if isinstance(v, np.ndarray)
        }

    accepted_indices: List[int] = []
    accepted_layouts: List[Dict[str, np.ndarray]] = []

    if rejection_enabled:
        idx_to_labels = {i: str(label) for i, label in enumerate(encoded_dataset.class_labels)}
        floor_geometry = None
        floor_polygon_lookup = None
        if use_universal_pre_rejection:
            floor_polygon_lookup = _build_floor_polygon_xz_lookup_from_dataset(raw_dataset)

        if args.floor_geometry_path:
            print(
                "[rejection] --floor_geometry_path is deprecated and ignored; "
                "using floor geometry from raw_dataset."
            )

        print(
            f"[rejection] enabled with target_accepted={args.target_accepted}"
        )
        if has_custom_rejection:
            print(f"[rejection] custom stages enabled: count={len(custom_reward_stages)}")
            for stage_idx, (reward_file, _, threshold) in enumerate(custom_reward_stages, start=1):
                print(
                    f"[rejection] custom stage {stage_idx}: reward_file={reward_file}, "
                    f"threshold={threshold}"
                )
        else:
            print("[rejection] custom stage disabled; using universal stages only")
        if use_universal_pre_rejection:
            print(
                "[rejection] universal pre-stages enabled: "
                f"penetration>={args.penetration_threshold}, "
                f"boundary>={args.boundary_threshold}"
            )
        print(
            f"[rejection] using dataset floor geometry entries={len(raw_dataset)} "
            "aligned with encoded dataset indices"
        )

        batch_counter = 0
        while len(accepted_layouts) < args.target_accepted:
            batch_counter += 1
            if batch_counter > args.max_rejection_batches:
                raise RuntimeError(
                    "Reached max rejection batches before hitting target accepted scenes. "
                    f"accepted={len(accepted_layouts)}, target={args.target_accepted}, "
                    f"max_batches={args.max_rejection_batches}"
                )

            sampled_indices, layout_list = generate_ashok_layouts(
                network,
                encoded_dataset,
                config,
                num_syn_scenes=args.batch_size,
                sampling_rule="random",
                batch_size=args.batch_size,
                device=device,
                overfit_sample=overfit_sample,
                num_denoising_steps=args.num_denoising_steps,
            )

            parsed_batch = _build_parsed_scene_from_layouts(
                layouts=layout_list,
                max_objects=config["network"]["sample_num_points"],
                n_object_types=encoded_dataset.n_object_types,
                device=device,
            )

            surviving_positions = list(range(len(layout_list)))

            if use_universal_pre_rejection:
                pen_rewards = _score_parsed_scene(
                    parsed_scene=parsed_batch,
                    reward_fn=compute_non_penetration_reward,
                    reward_kwargs={
                        "room_type": _cfg_get(_cfg_get(config, "midiffusion", None), "room_type", "bedroom")
                    },
                    device=device,
                )
                pen_keep_mask = (
                    pen_rewards >= float(args.penetration_threshold)
                ).detach().cpu().numpy().astype(bool)
                surviving_positions = [i for i, keep in enumerate(pen_keep_mask) if keep]

                if len(surviving_positions) > 0:
                    parsed_after_pen = _slice_parsed_scene(parsed_batch, surviving_positions)
                    surviving_scene_indices = [int(sampled_indices[i]) for i in surviving_positions]
                    floor_polygons = [floor_polygon_lookup[idx] for idx in surviving_scene_indices]

                    boundary_rewards = _score_parsed_scene(
                        parsed_scene=parsed_after_pen,
                        reward_fn=compute_boundary_violation_reward,
                        reward_kwargs={"floor_polygons": floor_polygons},
                        device=device,
                    )
                    boundary_keep_mask = (
                        boundary_rewards >= float(args.boundary_threshold)
                    ).detach().cpu().numpy().astype(bool)
                    surviving_positions = [
                        surviving_positions[i]
                        for i, keep in enumerate(boundary_keep_mask)
                        if keep
                    ]

            final_keep_mask = np.zeros(len(layout_list), dtype=bool)
            if len(surviving_positions) > 0 and has_custom_rejection:
                base_reward_kwargs = _build_reward_kwargs_for_batch(
                    config=config,
                    idx_to_labels=idx_to_labels,
                    sampled_indices=sampled_indices,
                    raw_dataset=raw_dataset,
                )
                for stage_idx, (_, reward_fn, threshold) in enumerate(custom_reward_stages, start=1):
                    if len(surviving_positions) == 0:
                        break

                    parsed_for_stage = _slice_parsed_scene(parsed_batch, surviving_positions)
                    stage_reward_kwargs = _slice_reward_kwargs_by_positions(
                        reward_kwargs=base_reward_kwargs,
                        base_batch_size=len(layout_list),
                        keep_positions=surviving_positions,
                    )

                    stage_rewards = _score_parsed_scene(
                        parsed_scene=parsed_for_stage,
                        reward_fn=reward_fn,
                        reward_kwargs=stage_reward_kwargs,
                        device=device,
                    )
                    stage_keep_mask = (
                        stage_rewards >= threshold
                    ).detach().cpu().numpy().astype(bool)
                    surviving_positions = [
                        surviving_positions[i]
                        for i, keep in enumerate(stage_keep_mask)
                        if keep
                    ]
                    print(
                        f"[rejection] custom stage={stage_idx} "
                        f"survivors={len(surviving_positions)}"
                    )

                for pos_idx in surviving_positions:
                    final_keep_mask[pos_idx] = True
            else:
                for pos_idx in surviving_positions:
                    final_keep_mask[pos_idx] = True

            kept_this_batch = int(np.sum(final_keep_mask))

            for keep, scene_idx, layout in zip(final_keep_mask, sampled_indices, layout_list):
                if keep:
                    accepted_indices.append(int(scene_idx))
                    accepted_layouts.append(layout)

            print(
                f"[rejection] batch={batch_counter} generated={len(layout_list)} "
                f"after_universal={len(surviving_positions)} "
                f"accepted_batch={kept_this_batch} accepted_total={len(accepted_layouts)}"
            )

        # Truncate to exact target_accepted to keep deterministic output size.
        accepted_indices = accepted_indices[: args.target_accepted]
        accepted_layouts = accepted_layouts[: args.target_accepted]
        sampled_indices = accepted_indices
        layout_list = accepted_layouts
        print(f"[rejection] done: final accepted scenes={len(layout_list)}")

    else:
        sampled_indices, layout_list = generate_ashok_layouts(
            network,
            encoded_dataset,
            config,
            num_syn_scenes=args.n_syn_scenes,
            sampling_rule="random",
            batch_size=args.batch_size,
            device=device,
            overfit_sample=overfit_sample,
            num_denoising_steps=args.num_denoising_steps,
        )

    threed_front_results = ThreedFrontResults(
        raw_train_dataset,
        raw_dataset,
        config,
        sampled_indices,
        layout_list,
    )
    pickle.dump(threed_front_results, open(path_to_results, "wb"))
    print("Saved result to:", path_to_results)


if __name__ == "__main__":
    main(sys.argv[1:])
