import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .floor_condition_utils import (
    center_floor_polygon_xz,
    normalize_user_fpbpn,
    reconstruct_floor_polygon_xz_from_fpbpn,
)
from .generation_helpers import (
    RuntimeContext,
    generate_layouts_with_mandatory_filters,
    load_runtime_context,
)
from .render_helpers import render_layouts_to_png_and_glb


REPO_ROOT = Path(__file__).resolve().parents[1]
MIDIFFUSION_DIR = REPO_ROOT / "3d_layout_generation" / "MiDiffusion"
THREEDFRONT_DIR = REPO_ROOT / "3d_layout_generation" / "ThreedFront"

import sys

for p in [str(MIDIFFUSION_DIR), str(THREEDFRONT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from threed_front.datasets import ThreedFutureDataset  # noqa: E402


DEFAULT_WEIGHT_FILE = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000"
DEFAULT_LORA_FILE = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/universal_only_oob_area/stage92/checkpoints/checkpoint_1/lora_weights.pt"
DEFAULT_3D_FUTURE_PKL = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl"


class UserSceneService:
    """Server-friendly helper service for conditioned 3D scene generation."""

    def __init__(
        self,
        weight_file: str = DEFAULT_WEIGHT_FILE,
        lora_file: Optional[str] = DEFAULT_LORA_FILE,
        path_to_pickled_3d_future_model: str = DEFAULT_3D_FUTURE_PKL,
        config_file: Optional[str] = None,
        gpu: int = 0,
        split=("test",),
    ):
        self.context: RuntimeContext = load_runtime_context(
            weight_file=weight_file,
            lora_file=lora_file,
            config_file=config_file,
            gpu=gpu,
            split=split,
        )

        room_type = self.context.room_type
        pkl_path = str(path_to_pickled_3d_future_model)
        if "{}" in pkl_path:
            pkl_path = pkl_path.format(room_type)

        self.objects_dataset = ThreedFutureDataset.from_pickled_dataset(pkl_path)
        self.path_to_pickled_3d_future_model = pkl_path

    @property
    def object_types(self) -> List[str]:
        return list(self.context.encoded_dataset.object_types)

    @property
    def fpbpn_bounds(self):
        return self.context.encoded_dataset.bounds["fpbpn"]

    def generate_and_render(
        self,
        fpbpn: np.ndarray,
        num_scenes: int,
        output_dir: str,
        furniture_objects: Optional[Dict[str, int]] = None,
        input_space: str = "scaled",
        furniture_filter_mode: str = "hard",
        furniture_soft_weights: Optional[Dict[str, float]] = None,
        furniture_soft_max_distance: float = 0.0,
        penetration_threshold: float = 0.0,
        boundary_threshold: float = 0.0,
        batch_size: int = 128,
        num_denoising_steps: int = 20,
        max_rejection_batches: int = 5000,
        export_glb: bool = True,
        add_floor: bool = True,
        seed: int = 42,
    ) -> Dict[str, object]:
        normalized = normalize_user_fpbpn(
            fpbpn=fpbpn,
            input_space=input_space,
            fpbpn_bounds=self.fpbpn_bounds,
        )
        fpbpn_scaled = normalized["fpbpn_scaled"]
        fpbpn_world = normalized["fpbpn_world"]

        floor_polygon_world_xz = reconstruct_floor_polygon_xz_from_fpbpn(fpbpn_world)
        floor_centered = center_floor_polygon_xz(floor_polygon_world_xz)
        floor_polygon_centered_xz = floor_centered["polygon_centered_xz"]

        layouts = generate_layouts_with_mandatory_filters(
            context=self.context,
            fpbpn_scaled=fpbpn_scaled,
            floor_polygon_centered_xz=floor_polygon_centered_xz,
            furniture_objects=furniture_objects,
            target_accepted=int(num_scenes),
            batch_size=batch_size,
            num_denoising_steps=num_denoising_steps,
            penetration_threshold=penetration_threshold,
            boundary_threshold=boundary_threshold,
            furniture_filter_mode=furniture_filter_mode,
            furniture_soft_weights=furniture_soft_weights,
            furniture_soft_max_distance=furniture_soft_max_distance,
            max_rejection_batches=max_rejection_batches,
            seed=seed,
        )

        render_out = render_layouts_to_png_and_glb(
            layouts=layouts,
            floor_polygon_centered_xz=floor_polygon_centered_xz,
            objects_dataset=self.objects_dataset,
            object_types=self.object_types,
            output_dir=output_dir,
            export_glb=export_glb,
            add_floor=add_floor,
            room_side=3.1 if self.context.room_type in {"bedroom", "library"} else 6.1,
        )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        layouts_path = out_path / "layouts.pkl"
        with open(layouts_path, "wb") as f:
            pickle.dump(layouts, f)

        return {
            "output_dir": str(out_path),
            "layouts_path": str(layouts_path),
            "png_paths": render_out["png_paths"],
            "glb_paths": render_out["glb_paths"],
            "accepted_count": len(layouts),
            "room_type": self.context.room_type,
            "object_types": self.object_types,
        }
