from datetime import datetime
from pathlib import Path
import random
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import (
    DEFAULT_3D_FUTURE_PKL,
    DEFAULT_LORA_FILE,
    DEFAULT_WEIGHT_FILE,
    UserSceneService,
)


# TODO: Replace these default LoRA checkpoints with your final task-specific paths.
DEFAULT_LORA_BY_ENDPOINT = {
    "generate-free-space": DEFAULT_LORA_FILE,
    "generate-study": DEFAULT_LORA_FILE,
    "generate-entertainment": DEFAULT_LORA_FILE,
}


class GenerateRequest(BaseModel):
    floor_vertices: List[List[float]] = Field(
        ..., description="2D polygon vertices [[x, z], ...] for floor condition"
    )
    num_scenes: int = Field(4, ge=1)
    specific_objects: Optional[Dict[str, int]] = None
    output_root: str = "user_app_outputs/api"
    seed: int = 42
    batch_size: int = 128
    num_denoising_steps: int = 20


class GenerateResponse(BaseModel):
    output_dir: str
    glb_folder: str
    glb_paths: List[str]
    accepted_count: int
    floor_index_used: int
    room_type: str


def _get_default_task_objects(task_name: str, object_types: List[str]) -> Dict[str, int]:
    preferred = {
        "generate-free-space": ["double_bed", "single_bed", "desk", "chair"],
        "generate-study": ["desk", "chair", "bookshelf", "cabinet"],
        "generate-entertainment": ["tv_stand", "double_bed", "sofa", "chair"],
        "generate-normal": ["double_bed", "nightstand"],
    }.get(task_name, ["double_bed", "nightstand"])

    for label in preferred:
        if label in object_types:
            return {label: 1}

    non_empty = [x for x in object_types if x.lower() != "empty"]
    if not non_empty:
        raise ValueError("No non-empty object classes available in encoded dataset")
    return {non_empty[0]: 1}


def _build_service(lora_file: Optional[str]) -> UserSceneService:
    return UserSceneService(
        weight_file=DEFAULT_WEIGHT_FILE,
        lora_file=lora_file,
        path_to_pickled_3d_future_model=DEFAULT_3D_FUTURE_PKL,
        config_file=None,
        gpu=0,
        split=("test",),
    )


def _pick_random_floor_index(service: UserSceneService, seed: int) -> int:
    rng = random.Random(seed)
    dataset_len = len(service.context.encoded_dataset)
    return rng.randint(0, max(0, dataset_len - 1))


def _run_generation(task_name: str, request: GenerateRequest, lora_file: Optional[str]) -> GenerateResponse:
    service = _build_service(lora_file=lora_file)

    # TODO: Convert request.floor_vertices into fpbpn conditioning.
    # Demo behavior for now: ignore provided polygon and pick a random floor from dataset.
    floor_idx = _pick_random_floor_index(service, request.seed)
    fpbpn = service.context.encoded_dataset[floor_idx]["fpbpn"]

    furniture_objects = request.specific_objects
    if not furniture_objects:
        # furniture_objects = _get_default_task_objects(task_name, service.object_types)
        furniture_objects = None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(request.output_root) / task_name / ts

    result = service.generate_and_render(
        fpbpn=fpbpn,
        furniture_objects=furniture_objects,
        num_scenes=request.num_scenes,
        output_dir=str(out_dir),
        input_space="scaled",
        furniture_filter_mode="hard",
        batch_size=request.batch_size,
        num_denoising_steps=request.num_denoising_steps,
        penetration_threshold=0.0,
        boundary_threshold=0.0,
        seed=request.seed,
        export_glb=True,
        add_floor=True,
    )

    return GenerateResponse(
        output_dir=result["output_dir"],
        glb_folder=result["output_dir"],
        glb_paths=result["glb_paths"],
        accepted_count=result["accepted_count"],
        floor_index_used=floor_idx,
        room_type=result["room_type"],
    )


def create_app() -> FastAPI:
    app = FastAPI(title="User Scene Generation API", version="0.1.0")

    @app.post("/generate-free-space", response_model=GenerateResponse)
    def generate_free_space(request: GenerateRequest):
        # TODO: Set final LoRA path for free-space once selected.
        return _run_generation(
            task_name="generate-free-space",
            request=request,
            lora_file=DEFAULT_LORA_BY_ENDPOINT["generate-free-space"],
        )

    @app.post("/generate-study", response_model=GenerateResponse)
    def generate_study(request: GenerateRequest):
        # TODO: Set final LoRA path for study once selected.
        return _run_generation(
            task_name="generate-study",
            request=request,
            lora_file=DEFAULT_LORA_BY_ENDPOINT["generate-study"],
        )

    @app.post("/generate-entertainment", response_model=GenerateResponse)
    def generate_entertainment(request: GenerateRequest):
        # TODO: Set final LoRA path for entertainment once selected.
        return _run_generation(
            task_name="generate-entertainment",
            request=request,
            lora_file=DEFAULT_LORA_BY_ENDPOINT["generate-entertainment"],
        )

    @app.post("/generate-normal", response_model=GenerateResponse)
    def generate_normal(request: GenerateRequest):
        # Base model only (no LoRA) as requested.
        return _run_generation(
            task_name="generate-normal",
            request=request,
            lora_file=None,
        )

    return app


app = create_app()
