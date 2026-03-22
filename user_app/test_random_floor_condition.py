import argparse
import random
from pathlib import Path
import sys


# Allow running this file both as:
# 1) python user_app/test_random_floor_condition.py  (from repo root)
# 2) python test_random_floor_condition.py           (from user_app dir)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_app import UserSceneService


DEFAULT_WEIGHT_FILE = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000"
DEFAULT_LORA_FILE = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/universal_only_oob_area/stage92/checkpoints/checkpoint_1/lora_weights.pt"
DEFAULT_3D_FUTURE_PKL = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl"


def main():
    parser = argparse.ArgumentParser(description="Test user_app pipeline with random real floor condition")
    parser.add_argument(
        "weight_file",
        nargs="?",
        default=DEFAULT_WEIGHT_FILE,
        help="Path to baseline model weights",
    )
    parser.add_argument("--lora", default=DEFAULT_LORA_FILE, help="Optional LoRA weights")
    parser.add_argument(
        "--path_to_pickled_3d_future_model",
        default=DEFAULT_3D_FUTURE_PKL,
        help="Path to threed future model pickle (or template with {} room_type)",
    )
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_scenes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_denoising_steps", type=int, default=20)
    parser.add_argument("--output_dir", default="user_app_outputs/test_random_floor")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    service = UserSceneService(
        weight_file=args.weight_file,
        lora_file=args.lora,
        path_to_pickled_3d_future_model=args.path_to_pickled_3d_future_model,
        config_file=args.config_file,
        gpu=args.gpu,
        split=("test",),
    )

    rng = random.Random(args.seed)
    dataset_len = len(service.context.encoded_dataset)
    chosen_idx = rng.randint(0, max(0, dataset_len - 1))

    fpbpn = service.context.encoded_dataset[chosen_idx]["fpbpn"]

    # Example furniture requirement; replace as needed.
    furniture_objects = {
        "double_bed": 1,
        "nightstand": 1,
    }

    out_dir = Path(args.output_dir) / f"scene_{chosen_idx:03d}"

    result = service.generate_and_render(
        fpbpn=fpbpn,
        furniture_objects=furniture_objects,
        num_scenes=args.num_scenes,
        output_dir=str(out_dir),
        input_space="scaled",
        furniture_filter_mode="hard",
        batch_size=args.batch_size,
        num_denoising_steps=args.num_denoising_steps,
        penetration_threshold=0.0,
        boundary_threshold=0.0,
        seed=args.seed,
        export_glb=True,
        add_floor=True,
    )

    print("Random floor index:", chosen_idx)
    print("Accepted scenes:", result["accepted_count"])
    print("Layouts:", result["layouts_path"])
    print("PNGs:")
    for p in result["png_paths"]:
        print("  ", p)
    print("GLBs:")
    for p in result["glb_paths"]:
        print("  ", p)


if __name__ == "__main__":
    main()
