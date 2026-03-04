"""Visualise intermediate DDIM denoising steps for a small set of scenes.

Denoises N samples (default 4) from pure noise, saving a separate
ThreedFrontResults pkl **per sample** at every intermediate timestep
(using the x0 prediction extracted inside ddim_step_with_logprob –
exactly the same as the resampling code in fk_sampling.py).

Output layout
-------------
<output_dir>/
  sample_000.pkl   ← ThreedFrontResults with T entries, one per DDIM step
                     (renderable with render_results.py – renders all T scenes)
  sample_001.pkl
  ...
  scene_indices.json

Usage
-----
  cd 3d_layout_generation/MiDiffusion
  python scripts/visualize_intermediate_steps.py \\
      /path/to/model_weights.pt \\
      --num_samples 4 \\
      --num_denoising_steps 20 \\
      --output_directory /path/to/out \\
      --gpu 0

Then render any step with:
  python ../ThreedFront/scripts/render_results.py \\
      /path/to/out/sample_000/step_009.pkl \\
      --no_texture \\
      --path_to_pickled_3d_future_model .../threed_future_model_bedroom.pkl
"""

import argparse
import os
import sys
import pickle

import numpy as np
import torch
from diffusers import DDIMScheduler
from tqdm import tqdm

# ── make b2diff root importable so we can use diffusion.ddim_with_logprob ────
_SCRIPT_DIR    = os.path.dirname(os.path.realpath(__file__))
_MIDIFF_DIR    = os.path.dirname(_SCRIPT_DIR)                       # .../MiDiffusion
_3D_LAYOUT_DIR = os.path.dirname(_MIDIFF_DIR)                       # .../3d_layout_generation
_THREEDFRONT_DIR = os.path.join(_3D_LAYOUT_DIR, "ThreedFront")      # .../ThreedFront
_B2DIFF_ROOT   = os.path.dirname(_3D_LAYOUT_DIR)                    # .../b2diff
for _p in [_SCRIPT_DIR, _MIDIFF_DIR, _THREEDFRONT_DIR, _B2DIFF_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import PROJ_DIR, load_config, update_data_file_paths   # MiDiffusion/scripts/utils.py
from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults
from midiffusion.datasets.threed_front_encoding import get_dataset_raw_and_encoded
from midiffusion.ashok_midiffusion import SceneDiffuserMiDiffusion


# ---------------------------------------------------------------------------
# helpers – identical to ashok_generate_results.py
# ---------------------------------------------------------------------------

def _delete_empty_from_x0(x0, n_object_types):
    """Convert (B, N, C) raw x0 tensor → list[B] of layout dicts.

    Removes empty-object slots and returns dicts ready for
    encoded_dataset.post_process().
    """
    bbox_dim  = 8   # 3 trans + 3 size + 2 angle
    class_dim = x0.shape[-1] - bbox_dim

    translations = x0[..., 0:3]
    sizes        = x0[..., 3:6]
    angles       = x0[..., 6:8]          # [cos θ, sin θ]
    class_raw    = x0[..., bbox_dim:]

    class_scores         = class_raw[..., :n_object_types]
    obj_max, obj_max_ind = torch.max(class_scores, dim=-1)
    empty_logit          = class_raw[..., class_dim - 1]
    is_empty             = empty_logit > obj_max

    class_onehot = torch.nn.functional.one_hot(
        obj_max_ind, num_classes=n_object_types
    ).float()

    B, N, _ = x0.shape
    boxes_list = []
    for b in range(B):
        box = {
            "translations": torch.zeros(1, 0, 3),
            "sizes":        torch.zeros(1, 0, 3),
            "angles":       torch.zeros(1, 0, 2),
            "class_labels": torch.zeros(1, 0, n_object_types),
        }
        for i in range(N):
            if is_empty[b, i]:
                continue
            box["translations"] = torch.cat(
                [box["translations"], translations[b:b+1, i:i+1, :].cpu()], dim=1)
            box["sizes"]        = torch.cat(
                [box["sizes"],        sizes[b:b+1, i:i+1, :].cpu()],        dim=1)
            box["angles"]       = torch.cat(
                [box["angles"],       angles[b:b+1, i:i+1, :].cpu()],       dim=1)
            box["class_labels"] = torch.cat(
                [box["class_labels"], class_onehot[b:b+1, i:i+1, :].cpu()], dim=1)
        boxes_list.append(box)
    return boxes_list


def x0_to_layout(x0_single, encoded_dataset):
    """Convert a single (1, N, C) x0 prediction to a post-processed layout dict."""
    n_obj_types = encoded_dataset.n_object_types
    boxes     = _delete_empty_from_x0(x0_single.float().cpu(), n_obj_types)  # list of 1
    bbox_dict = boxes[0]
    processed = encoded_dataset.post_process(bbox_dict)
    return {k: v.numpy()[0] for k, v in processed.items()}


# ---------------------------------------------------------------------------
# main denoising loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise_and_save_steps(
    network,
    scene_indices,          # list[int] – indices into encoded_dataset
    encoded_dataset,
    raw_train_dataset,
    raw_test_dataset,
    midiff_config,
    output_dir,
    num_denoising_steps=20,
    eta=0.0,
    device="cpu",
):
    """Denoise each sample step-by-step, collecting x0_pred at every step.

    Saves ONE pkl per sample containing all T intermediate layouts so that
    render_results.py renders all denoising steps for that sample at once.
    """
    num_timesteps = midiff_config["network"].get("time_num", 1000)
    diff_geo      = midiff_config["network"].get("diffusion_geometric_kwargs", {})
    beta_start    = diff_geo.get("beta_start", 1e-4)
    beta_end      = diff_geo.get("beta_end",   0.02)
    num_objects   = midiff_config["network"]["sample_num_points"]
    scene_dim     = 30   # 3 trans + 3 size + 2 angle + 22 class

    scheduler = DDIMScheduler(
        num_train_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False,
        prediction_type="epsilon",
        steps_offset=1,
    )
    scheduler.set_timesteps(num_denoising_steps, device=device)

    B = len(scene_indices)

    # floor plan features for the chosen scenes
    fpbpn = torch.from_numpy(
        np.stack([encoded_dataset[i]["fpbpn"] for i in scene_indices], axis=0)
    ).to(device).float()              # (B, 256, 4)

    # initial pure noise
    x_t = torch.randn(B, num_objects, scene_dim, device=device)

    # accumulate layouts per sample: step_layouts[s_i] = list[layout_dict]
    step_layouts = [[] for _ in range(B)]

    print(f"Denoising {B} samples over {num_denoising_steps} DDIM steps ...")
    for t in tqdm(scheduler.timesteps, total=num_denoising_steps, desc="DDIM step"):
        t_tensor = torch.full((B,), t.item(), dtype=torch.long, device=device)
        eps_pred = network.predict_noise(x_t, t_tensor, fpbpn)

        # step() returns DDIMSchedulerOutput with:
        #   .prev_sample          – x_{t-1}
        #   .pred_original_sample – x0 estimate (same quantity FK resampling uses)
        out     = scheduler.step(eps_pred, t, x_t, eta=eta)
        x0_pred = out.pred_original_sample   # (B, N, C)

        for s_i in range(B):
            layout = x0_to_layout(x0_pred[s_i:s_i+1], encoded_dataset)
            step_layouts[s_i].append(layout)

        x_t = out.prev_sample   # advance

    # save one pkl per sample – all T steps packed into a single ThreedFrontResults
    for s_i, scene_idx in enumerate(scene_indices):
        T       = len(step_layouts[s_i])
        result  = ThreedFrontResults(
            raw_train_dataset,
            raw_test_dataset,
            midiff_config,
            scene_indices=[scene_idx] * T,   # same floor plan for every step
            predicted_layouts=step_layouts[s_i],
        )
        out_path = os.path.join(output_dir, f"sample_{s_i:03d}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved sample_{s_i:03d}.pkl  ({T} steps, scene_idx={scene_idx})")

    print(f"Done. PKL files saved under: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Denoise N scenes from pure noise and save per-sample ThreedFrontResults "
            "pkls at every DDIM timestep (x0 prediction). "
            "Renderable with render_results.py."
        )
    )
    parser.add_argument(
        "weight_file",
        help="Path to a pretrained model .pt state-dict (baseline or LoRA weights)."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        help="Path to experiment config yaml "
             "(default: config.yaml next to weight_file)."
    )
    parser.add_argument(
        "--output_directory",
        default=os.path.join(PROJ_DIR, "output", "intermediate_vis"),
        help="Root directory for output pkls."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of scenes to generate (default: 4)."
    )
    parser.add_argument(
        "--scene_indices",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Explicit test-dataset indices to use as floor plans. "
            "If omitted, --num_samples random indices are chosen."
        ),
    )
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=20,
        help="Number of DDIM inference steps (default: 20)."
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="DDIM eta (0 = deterministic, 1 = DDPM-like). Default: 0."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID (default: 0)."
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="Path to LoRA weights (.pt). If given, LoRA adapters are applied."
    )
    args = parser.parse_args(argv)

    # ── reproducibility ──────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu < torch.cuda.device_count()
        else torch.device("cpu")
    )
    print(f"Running on: {device}")

    # ── config ───────────────────────────────────────────────────────────────
    if args.config_file is None:
        args.config_file = os.path.join(
            os.path.dirname(args.weight_file), "config.yaml"
        )
    config = load_config(args.config_file)
    if "_eval" not in config["data"]["encoding_type"]:
        config["data"]["encoding_type"] += "_eval"

    # ── datasets ─────────────────────────────────────────────────────────────
    data_cfg = update_data_file_paths(config["data"])

    raw_train_dataset = get_raw_dataset(
        data_cfg,
        split=config["training"].get("splits", ["train", "val"]),
        include_room_mask=config["network"].get("room_mask_condition", True),
    )
    raw_test_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        data_cfg,
        split=config["validation"].get("splits", ["test"]),
        max_length=config["network"]["sample_num_points"],
        include_room_mask=config["network"].get("room_mask_condition", True),
    )
    n_test = len(encoded_dataset)
    print(f"Test dataset: {n_test} scenes, {encoded_dataset.n_object_types} object types")

    # ── scene indices ─────────────────────────────────────────────────────────
    if args.scene_indices is not None:
        scene_indices = args.scene_indices
        for idx in scene_indices:
            if not (0 <= idx < n_test):
                raise ValueError(f"scene_index {idx} out of range [0, {n_test})")
    else:
        scene_indices = np.random.choice(n_test, args.num_samples, replace=False).tolist()
    print(f"Using scene indices: {scene_indices}")

    # ── model ─────────────────────────────────────────────────────────────────
    network = SceneDiffuserMiDiffusion()
    state   = torch.load(args.weight_file, map_location=device)
    network.load_state_dict(state)
    network.to(device)
    print(f"Loaded baseline weights from {args.weight_file}")

    if args.lora is not None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        network = get_peft_model(network, lora_cfg)
        lora_state = torch.load(args.lora, map_location=device)
        network.load_state_dict(lora_state, strict=False)
        network.to(device)
        print(f"Loaded LoRA weights from {args.lora}")

    network.eval()

    # ── output dir ───────────────────────────────────────────────────────────
    os.makedirs(args.output_directory, exist_ok=True)

    # save scene-index mapping for reference
    import json
    with open(os.path.join(args.output_directory, "scene_indices.json"), "w") as f:
        json.dump({"scene_indices": scene_indices}, f, indent=2)
    print(f"Scene index mapping saved to {args.output_directory}/scene_indices.json")

    # ── run ───────────────────────────────────────────────────────────────────
    denoise_and_save_steps(
        network        = network,
        scene_indices  = scene_indices,
        encoded_dataset= encoded_dataset,
        raw_train_dataset = raw_train_dataset,
        raw_test_dataset  = raw_test_dataset,
        midiff_config  = config,
        output_dir     = args.output_directory,
        num_denoising_steps = args.num_denoising_steps,
        eta            = args.eta,
        device         = device,
    )


if __name__ == "__main__":
    main()
