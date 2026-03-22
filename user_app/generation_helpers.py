from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers import DDIMScheduler

from .furniture_constraints import (
    apply_fixed_classes_to_scene_tensor,
    enforce_fixed_class_order_in_layout,
    expand_furniture_slots,
    furniture_filter_hard_exact,
    furniture_filter_soft_weighted,
    labels_to_indices_in_order,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MIDIFFUSION_DIR = REPO_ROOT / "3d_layout_generation" / "MiDiffusion"
MIDIFFUSION_SCRIPTS_DIR = MIDIFFUSION_DIR / "scripts"
THREEDFRONT_DIR = REPO_ROOT / "3d_layout_generation" / "ThreedFront"

import sys

for p in [str(MIDIFFUSION_DIR), str(MIDIFFUSION_SCRIPTS_DIR), str(THREEDFRONT_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from midiffusion.ashok_midiffusion import SceneDiffuserMiDiffusion  # noqa: E402
from midiffusion.datasets.threed_front_encoding import get_dataset_raw_and_encoded  # noqa: E402
from utils import load_config, update_data_file_paths  # noqa: E402


DEFAULT_WEIGHT_FILE = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000"
DEFAULT_LORA_FILE = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/universal_only_oob_area/stage92/checkpoints/checkpoint_1/lora_weights.pt"


def _load_callable_from_python_file(module_path: Path, function_name: str):
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


UNIVERSAL_REWARD_DIR = REPO_ROOT / "core" / "universal_rewards"
compute_boundary_violation_reward = _load_callable_from_python_file(
    UNIVERSAL_REWARD_DIR / "not_out_of_bound_reward.py",
    "compute_boundary_violation_reward",
)
compute_non_penetration_reward = _load_callable_from_python_file(
    UNIVERSAL_REWARD_DIR / "penetration_reward.py",
    "compute_non_penetration_reward",
)


@dataclass
class RuntimeContext:
    config: Dict
    raw_dataset: object
    encoded_dataset: object
    network: torch.nn.Module
    device: torch.device
    room_type: str


def _infer_room_type_from_config(config: Dict) -> str:
    dataset_dir = str(config["data"]["dataset_directory"])
    for t in ["bedroom", "livingroom", "diningroom", "library"]:
        if t in dataset_dir:
            return t
    return "bedroom"


def _angles_to_orientation(angles: np.ndarray) -> np.ndarray:
    if angles.ndim == 1:
        return np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    if angles.ndim == 2 and angles.shape[-1] == 1:
        return np.stack([np.cos(angles[:, 0]), np.sin(angles[:, 0])], axis=-1)
    if angles.ndim == 2 and angles.shape[-1] == 2:
        norm = np.linalg.norm(angles, axis=-1, keepdims=True)
        return angles / np.clip(norm, 1e-8, None)
    raise ValueError(f"Unsupported angles shape: {angles.shape}")


def _build_parsed_scene_from_layouts(
    layouts: List[Dict[str, np.ndarray]],
    max_objects: int,
    n_object_types: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch_size = len(layouts)
    positions = torch.zeros((batch_size, max_objects, 3), dtype=torch.float32, device=device)
    sizes = torch.zeros((batch_size, max_objects, 3), dtype=torch.float32, device=device)
    orientations = torch.zeros((batch_size, max_objects, 2), dtype=torch.float32, device=device)
    object_indices = torch.zeros((batch_size, max_objects), dtype=torch.long, device=device)
    class_labels = torch.zeros((batch_size, max_objects, n_object_types), dtype=torch.float32, device=device)
    is_empty = torch.ones((batch_size, max_objects), dtype=torch.bool, device=device)

    for b, scene in enumerate(layouts):
        p = np.asarray(scene["translations"], dtype=np.float32)
        s = np.asarray(scene["sizes"], dtype=np.float32)
        a = np.asarray(scene["angles"], dtype=np.float32)
        c = np.asarray(scene["class_labels"], dtype=np.float32)

        if p.ndim == 1:
            p = p.reshape(1, -1)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        if a.ndim == 0:
            a = a.reshape(1)
        if c.ndim == 1:
            c = c.reshape(1, -1)

        n_obj = int(min(p.shape[0], max_objects))
        if n_obj == 0:
            continue

        o = _angles_to_orientation(a)
        idx = np.argmax(c, axis=-1).astype(np.int64)

        positions[b, :n_obj] = torch.from_numpy(p[:n_obj]).to(device)
        sizes[b, :n_obj] = torch.from_numpy(s[:n_obj]).to(device)
        orientations[b, :n_obj] = torch.from_numpy(o[:n_obj]).to(device)
        object_indices[b, :n_obj] = torch.from_numpy(idx[:n_obj]).to(device)
        class_labels[b, :n_obj] = torch.from_numpy(c[:n_obj]).to(device)
        is_empty[b, :n_obj] = False

    return {
        "device": device,
        "positions": positions,
        "sizes": sizes,
        "orientations": orientations,
        "object_indices": object_indices,
        "class_labels": class_labels,
        "is_empty": is_empty,
    }


def _delete_empty_from_samples(x0: torch.Tensor, n_object_types: int) -> List[Dict[str, torch.Tensor]]:
    bbox_dim = 8
    class_dim = x0.shape[-1] - bbox_dim

    translations = x0[..., 0:3]
    sizes = x0[..., 3:6]
    angles = x0[..., 6:8]
    class_raw = x0[..., bbox_dim:]

    class_scores = class_raw[..., :n_object_types]
    obj_max, obj_max_ind = torch.max(class_scores, dim=-1)

    empty_logit = class_raw[..., class_dim - 1]
    is_empty = empty_logit > obj_max

    class_onehot = torch.nn.functional.one_hot(obj_max_ind, num_classes=n_object_types).float()

    boxes_list = []
    B, N, _ = x0.shape
    for b in range(B):
        box = {
            "translations": torch.zeros(1, 0, 3),
            "sizes": torch.zeros(1, 0, 3),
            "angles": torch.zeros(1, 0, 2),
            "class_labels": torch.zeros(1, 0, n_object_types),
        }
        for i in range(N):
            if is_empty[b, i]:
                continue
            box["translations"] = torch.cat([box["translations"], translations[b : b + 1, i : i + 1, :].cpu()], dim=1)
            box["sizes"] = torch.cat([box["sizes"], sizes[b : b + 1, i : i + 1, :].cpu()], dim=1)
            box["angles"] = torch.cat([box["angles"], angles[b : b + 1, i : i + 1, :].cpu()], dim=1)
            box["class_labels"] = torch.cat([box["class_labels"], class_onehot[b : b + 1, i : i + 1, :].cpu()], dim=1)
        boxes_list.append(box)
    return boxes_list


@torch.no_grad()
def _ddim_sample(
    network,
    fpbpn: torch.Tensor,
    num_timesteps: int,
    beta_start: float,
    beta_end: float,
    num_objects: int,
    scene_dim: int,
    device: torch.device,
    num_denoising_steps: int,
    x_t_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size = fpbpn.shape[0]

    scheduler = DDIMScheduler(
        num_train_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False,
        prediction_type="epsilon",
        steps_offset=1 if num_denoising_steps < num_timesteps else 0,
    )
    scheduler.set_timesteps(num_denoising_steps, device=device)

    if x_t_init is None:
        x_t = torch.randn(batch_size, num_objects, scene_dim, device=device)
    else:
        x_t = x_t_init.clone().to(device)

    for t in scheduler.timesteps:
        t_tensor = torch.full((batch_size,), int(t.item()), dtype=torch.long, device=device)
        eps_pred = network.predict_noise(x_t, t_tensor, fpbpn)
        x_t = scheduler.step(eps_pred, t, x_t, eta=0.0).prev_sample

    return x_t


def load_runtime_context(
    weight_file: str = DEFAULT_WEIGHT_FILE,
    lora_file: Optional[str] = DEFAULT_LORA_FILE,
    config_file: Optional[str] = None,
    gpu: int = 0,
    split: Sequence[str] = ("test",),
) -> RuntimeContext:
    if config_file is None:
        config_file = str(Path(weight_file).with_name("config.yaml"))

    config = load_config(config_file)
    if "_eval" not in config["data"]["encoding_type"]:
        config["data"]["encoding_type"] += "_eval"

    if gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"]),
        split=list(split),
        max_length=config["network"]["sample_num_points"],
        include_room_mask=config["network"].get("room_mask_condition", True),
    )

    network = SceneDiffuserMiDiffusion()
    state = torch.load(weight_file, map_location=device)
    network.load_state_dict(state)

    if lora_file is not None:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        network = get_peft_model(network, lora_config)
        lora_state = torch.load(lora_file, map_location=device)
        network.load_state_dict(lora_state, strict=False)

    network.to(device)
    network.eval()

    return RuntimeContext(
        config=config,
        raw_dataset=raw_dataset,
        encoded_dataset=encoded_dataset,
        network=network,
        device=device,
        room_type=_infer_room_type_from_config(config),
    )


def generate_layouts_with_mandatory_filters(
    context: RuntimeContext,
    fpbpn_scaled: np.ndarray,
    floor_polygon_centered_xz: np.ndarray,
    target_accepted: int,
    furniture_objects: Optional[Dict[str, int]] = None,
    batch_size: int = 128,
    num_denoising_steps: int = 20,
    penetration_threshold: float = 0.0,
    boundary_threshold: float = 0.0,
    furniture_filter_mode: str = "hard",
    furniture_soft_weights: Optional[Dict[str, float]] = None,
    furniture_soft_max_distance: float = 0.0,
    max_rejection_batches: int = 5000,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """Generate scenes from one floor condition with universal filtering and optional furniture constraints."""

    np.random.seed(seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))

    cfg = context.config
    net = context.network
    enc = context.encoded_dataset
    device = context.device

    n_object_types = int(enc.n_object_types)
    num_objects = int(cfg["network"]["sample_num_points"])
    scene_dim = 30

    if furniture_objects is None:
        furniture_objects = {}
    else:
        furniture_objects = {k: int(v) for k, v in furniture_objects.items() if int(v) > 0}

    has_furniture_constraints = len(furniture_objects) > 0
    if has_furniture_constraints:
        fixed_labels = expand_furniture_slots(furniture_objects)
        fixed_class_indices = labels_to_indices_in_order(fixed_labels, list(enc.object_types))
        if len(fixed_class_indices) > num_objects:
            raise ValueError(
                f"Requested fixed furniture slots ({len(fixed_class_indices)}) exceed sample_num_points ({num_objects})"
            )

        required_class_counts = {
            labels_to_indices_in_order([label], list(enc.object_types))[0]: int(count)
            for label, count in furniture_objects.items()
            if int(count) > 0
        }
    else:
        fixed_class_indices = []
        required_class_counts = {}

    soft_weights_idx = {}
    if has_furniture_constraints and furniture_soft_weights:
        for label, weight in furniture_soft_weights.items():
            idx = labels_to_indices_in_order([label], list(enc.object_types))[0]
            soft_weights_idx[idx] = float(weight)

    diff_geo = cfg["network"].get("diffusion_geometric_kwargs", {})
    num_timesteps = int(cfg["network"].get("time_num", 1000))
    beta_start = float(diff_geo.get("beta_start", 1e-4))
    beta_end = float(diff_geo.get("beta_end", 0.02))

    fpbpn_scaled = np.asarray(fpbpn_scaled, dtype=np.float32)
    fpbpn_batch_one = torch.from_numpy(fpbpn_scaled[None, :, :]).to(device).float()

    accepted_layouts: List[Dict[str, np.ndarray]] = []
    batch_counter = 0

    while len(accepted_layouts) < int(target_accepted):
        batch_counter += 1
        if batch_counter > int(max_rejection_batches):
            raise RuntimeError(
                "Reached max_rejection_batches before collecting enough accepted scenes: "
                f"accepted={len(accepted_layouts)} target={target_accepted}"
            )

        cur_bsz = int(batch_size)
        fpbpn_batch = fpbpn_batch_one.expand(cur_bsz, -1, -1)

        x_t_init = torch.randn(cur_bsz, num_objects, scene_dim, device=device)
        if has_furniture_constraints:
            x_t_init = apply_fixed_classes_to_scene_tensor(
                x_t_init,
                fixed_class_indices=fixed_class_indices,
                class_start=8,
                n_object_types=n_object_types,
            )

        x0 = _ddim_sample(
            net,
            fpbpn=fpbpn_batch,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            num_objects=num_objects,
            scene_dim=scene_dim,
            device=device,
            num_denoising_steps=num_denoising_steps,
            x_t_init=x_t_init,
        )

        if has_furniture_constraints:
            x0 = apply_fixed_classes_to_scene_tensor(
                x0,
                fixed_class_indices=fixed_class_indices,
                class_start=8,
                n_object_types=n_object_types,
            )

        boxes_list = _delete_empty_from_samples(x0, n_object_types=n_object_types)
        layouts: List[Dict[str, np.ndarray]] = []
        for b in boxes_list:
            boxes = enc.post_process(b)
            layout = {k: v.numpy()[0] for k, v in boxes.items()}
            if has_furniture_constraints:
                layout = enforce_fixed_class_order_in_layout(
                    layout,
                    fixed_class_indices=fixed_class_indices,
                    n_object_types=n_object_types,
                )
            layouts.append(layout)

        parsed = _build_parsed_scene_from_layouts(
            layouts,
            max_objects=num_objects,
            n_object_types=n_object_types,
            device=device,
        )

        with torch.no_grad():
            pen = compute_non_penetration_reward(parsed, room_type=context.room_type).to(device=device, dtype=torch.float32)
            bnd = compute_boundary_violation_reward(
                parsed,
                floor_polygons=[floor_polygon_centered_xz.tolist() for _ in range(len(layouts))],
            ).to(device=device, dtype=torch.float32)

        keep_pen = (pen >= float(penetration_threshold)).detach().cpu().numpy().astype(bool)
        keep_bnd = (bnd >= float(boundary_threshold)).detach().cpu().numpy().astype(bool)

        keep_universal = keep_pen & keep_bnd

        for i, layout in enumerate(layouts):
            if not keep_universal[i]:
                continue

            if not has_furniture_constraints:
                keep_furniture = True
            elif furniture_filter_mode == "hard":
                keep_furniture = furniture_filter_hard_exact(layout, required_class_counts)
            elif furniture_filter_mode == "soft":
                keep_furniture = furniture_filter_soft_weighted(
                    layout,
                    required_class_counts=required_class_counts,
                    class_weights=soft_weights_idx,
                    max_distance=furniture_soft_max_distance,
                )
            else:
                raise ValueError(f"Unknown furniture_filter_mode={furniture_filter_mode}")

            if keep_furniture:
                accepted_layouts.append(layout)
                if len(accepted_layouts) >= int(target_accepted):
                    break

    return accepted_layouts[: int(target_accepted)]
