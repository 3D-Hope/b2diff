"""Script for generating 3-D scene layouts using SceneDiffuserMiDiffusion.

Mirrors generate_results.py but drives the custom model defined in
midiffusion/ashok_midiffusion.py instead of the original build_network path.
Saving format is identical so downstream evaluation code works unchanged.
"""
import argparse
import os
import sys
import shutil
import pickle

import numpy as np
import torch
from diffusers import DDIMScheduler
from tqdm import tqdm

from utils import PROJ_DIR, load_config, update_data_file_paths
from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults
from midiffusion.datasets.threed_front_encoding import (
    get_dataset_raw_and_encoded,
    get_encoded_dataset,
)
from midiffusion.ashok_midiffusion import SceneDiffuserMiDiffusion


# ---------------------------------------------------------------------------
# DDIM reverse (denoising) sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(
    network, fpbpn,
    num_timesteps, beta_start, beta_end,
    num_objects, scene_dim, device,
    num_denoising_steps=20, eta=0.0,
):
    """Run DDIM reverse chain for one batch.

    Args:
        network             : SceneDiffuserMiDiffusion (eval mode, on device)
        fpbpn               : (B, nfpbp, 4) floor-plan boundary-point-normals tensor
        num_timesteps       : total training timesteps (e.g. 1000)
        beta_start          : diffusion beta_start
        beta_end            : diffusion beta_end
        num_objects         : N (max objects per scene)
        scene_dim           : C (per-object feature dimension, e.g. 30)
        device              : torch.device
        num_denoising_steps : number of DDIM inference steps (default 20)
        eta                 : stochasticity (0.0 = deterministic DDIM)

    Returns:
        x0_pred : (B, N, C) denoised scene tensor (still in scaled/encoded space)
    """
    B = fpbpn.shape[0]

    scheduler = DDIMScheduler(
        num_train_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False,
        prediction_type="epsilon",
        steps_offset=1 if num_denoising_steps < num_timesteps else 0,  # offset for 1000 steps (mirrors original code)
    )
    scheduler.set_timesteps(num_denoising_steps, device=device)

    print(f"Running DDIM sampling with {num_denoising_steps} steps")
    # print(f"Timesteps: {scheduler.timesteps.cpu().numpy()}")

    x_t = torch.randn(B, num_objects, scene_dim, device=device)

    for t in scheduler.timesteps:
        t_tensor = torch.full((B,), t.item(), dtype=torch.long, device=device)
        eps_pred = network.predict_noise(x_t, t_tensor, fpbpn)
        x_t = scheduler.step(eps_pred, t, x_t, eta=eta).prev_sample

    return x_t   # (B, N, C)


# ---------------------------------------------------------------------------
# Layout generation  (replaces midiffusion.evaluation.utils.generate_layouts)
# ---------------------------------------------------------------------------

def _delete_empty_from_samples(x0, n_object_types):
    """Replicate DiffusionSceneLayout_DDPM.delete_empty_from_network_samples.

    Args:
        x0            : (B, N, C) raw network output tensor (on any device)
        n_object_types: number of non-empty object classes
                        (= class_dim - 1, where last dim is "empty")

    Returns:
        list[dict] of length B – each dict has keys translations/sizes/angles/
        class_labels with tensors of shape (1, N_i, ?) on CPU, ready for
        encoded_dataset.post_process().
    """
    # layout: [trans(3), size(3), angle(2), class_raw(n_obj_types+1)]
    bbox_dim   = 8  # 3 + 3 + 2
    class_dim  = x0.shape[-1] - bbox_dim          # total class output dims
    # class_dim = n_object_types + 1  (last = empty)

    translations = x0[..., 0:3]
    sizes        = x0[..., 3:6]
    angles       = x0[..., 6:8]
    class_raw    = x0[..., bbox_dim:]              # (B, N, class_dim)

    # argmax over the NON-empty class dims only (mirror original code)
    class_scores = class_raw[..., :n_object_types]         # (B, N, n_obj_types)
    obj_max, obj_max_ind = torch.max(class_scores, dim=-1)  # (B, N)

    # "empty" slot: empty-class logit > best object-class logit
    empty_logit = class_raw[..., class_dim - 1]            # (B, N)
    is_empty    = empty_logit > obj_max                     # (B, N) bool

    # one-hot class labels with n_object_types classes  (no empty column)
    class_onehot = torch.nn.functional.one_hot(
        obj_max_ind, num_classes=n_object_types
    ).float()                                               # (B, N, n_obj_types)

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
            box["sizes"] = torch.cat(
                [box["sizes"],        sizes[b:b+1, i:i+1, :].cpu()],        dim=1)
            box["angles"] = torch.cat(
                [box["angles"],       angles[b:b+1, i:i+1, :].cpu()],       dim=1)
            box["class_labels"] = torch.cat(
                [box["class_labels"], class_onehot[b:b+1, i:i+1, :].cpu()], dim=1)
        boxes_list.append(box)
    return boxes_list


def generate_ashok_layouts(
    network,
    encoded_dataset,
    config,
    num_syn_scenes,
    sampling_rule="random",
    batch_size=16,
    device="cpu",
    overfit_sample=None,
    num_denoising_steps=20,
):
    """Generate scene layouts with SceneDiffuserMiDiffusion.

    Returns
    -------
    sampled_indices : list[int]  – dataset indices whose floor plans were used
    layout_list     : list[dict] – post-processed bbox dicts (one per scene)
    """
    num_timesteps  = config["network"].get("time_num", 1000)
    diff_geo       = config["network"].get("diffusion_geometric_kwargs", {})
    beta_start     = diff_geo.get("beta_start", 1e-4)
    beta_end       = diff_geo.get("beta_end", 0.02)

    num_objects    = config["network"]["sample_num_points"]
    scene_dim      = 30   # 3 trans + 3 size + 2 angle + 22 class
    # n_object_types = non-empty classes (class_dim - 1 = 21)
    n_object_types = encoded_dataset.n_object_types
    print(f"[generate] n_object_types={n_object_types}, scene_dim={scene_dim}, "
          f"num_objects={num_objects}")

    # sample floor plan indices
    if sampling_rule == "random":
        sampled_indices = np.random.choice(
            len(encoded_dataset), num_syn_scenes
        ).tolist()
    elif sampling_rule == "uniform":
        sampled_indices = (
            np.arange(len(encoded_dataset)).tolist()
            * (num_syn_scenes // len(encoded_dataset))
        )
        sampled_indices += np.random.choice(
            len(encoded_dataset),
            num_syn_scenes - len(sampled_indices),
        ).tolist()
    else:
        raise NotImplementedError(f"sampling_rule={sampling_rule}")

    network.to(device)
    network.eval()
    layout_list = []

    for i in tqdm(range(0, num_syn_scenes, batch_size)):
        scene_indices = sampled_indices[i : min(i + batch_size, num_syn_scenes)]
        B_cur = len(scene_indices)

        if overfit_sample is not None:
            # repeat the fixed overfit floor plan for every scene in the batch
            fpbpn = (
                overfit_sample["fpbpn"][:1]
                .expand(B_cur, -1, -1)
                .to(device)
                .float()
            )
        else:
            fpbpn = torch.from_numpy(
                np.stack(
                    [encoded_dataset[ind]["fpbpn"] for ind in scene_indices], axis=0
                )
            ).to(device).float()

        # run reverse diffusion → (B_cur, N, 30)
        x0 = ddim_sample(
            network, fpbpn,
            num_timesteps, beta_start, beta_end,
            num_objects, scene_dim, device,
            num_denoising_steps=num_denoising_steps,
        )

        # remove "empty" slots and convert class_labels to one-hot (n_object_types)
        # mirrors DiffusionSceneLayout_DDPM.delete_empty_from_network_samples
        boxes_list = _delete_empty_from_samples(x0, n_object_types)

        for bbox_params_dict in boxes_list:
            # post_process handles angle descaling (cos/sin→angle) and bounds
            boxes = encoded_dataset.post_process(bbox_params_dict)
            bbox_params = {k: v.numpy()[0] for k, v in boxes.items()}
            layout_list.append(bbox_params)

    return sampled_indices, layout_list


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using SceneDiffuserMiDiffusion"
    )
    parser.add_argument(
        "weight_file",
        help="Path to a pretrained model (.pt state-dict)"
    )
    parser.add_argument(
        "--config_file",
        default=None,
        help="Path to the experiment config yaml "
             "(default: config.yaml next to weight_file)"
    )
    parser.add_argument(
        "--output_directory",
        default=PROJ_DIR + "/output/predicted_results/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sampling floor plans"
    )
    parser.add_argument(
        "--n_syn_scenes",
        default=1000,
        type=int,
        help="Number of scenes to synthesize"
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Scenes per generation batch"
    )
    parser.add_argument(
        "--result_tag",
        default=None,
        help="Result sub-directory name"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
    )
    parser.add_argument(
        "--overfit_test",
        action="store_true",
        help=(
            "Use the first training sample's floor plan for every generated "
            "scene (mirrors the overfit_test mode in ashok_train.py)."
        )
    )
    parser.add_argument(
        "--lora",
        default=None,
        help=(
            "Path to LoRA weights (.pt file). If provided, LoRA adapters are applied "
            "to the baseline model and these weights are loaded. If not provided, "
            "only the baseline model weights are used."
        )
    )
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=20,
        help="Number of DDIM inference steps (default: 20)"
    )

    args = parser.parse_args(argv)

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if args.gpu < torch.cuda.device_count():
        device = torch.device("cuda:{}".format(args.gpu))
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
            "{} directory is non-empty. "
            "Press any key to remove all files...".format(result_dir)
        )
        for fi in os.listdir(result_dir):
            os.remove(os.path.join(result_dir, fi))
    else:
        os.makedirs(result_dir, exist_ok=True)

    path_to_config  = os.path.join(result_dir, "config.yaml")
    path_to_results = os.path.join(result_dir, "results.pkl")

    # config
    if args.config_file is None:
        args.config_file = os.path.join(
            os.path.dirname(args.weight_file), "config.yaml"
        )
    config = load_config(args.config_file)
    # ensure eval encoding is used (same convention as generate_results.py)
    if "_eval" not in config["data"]["encoding_type"]:
        config["data"]["encoding_type"] += "_eval"
    if not os.path.exists(path_to_config) or \
            not os.path.samefile(args.config_file, path_to_config):
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
        split=config["validation"].get("splits", ["test"]),
        max_length=config["network"]["sample_num_points"],
        include_room_mask=config["network"].get("room_mask_condition", True),
    )
    print("Loaded {} scenes with {} object types ({} labels):".format(
        len(encoded_dataset), encoded_dataset.n_object_types,
        encoded_dataset.n_classes,
    ))
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
        except ImportError:
            raise ImportError("peft is required for LoRA support. Install with: pip install peft")

        # Apply LoRA config (matching train_pipeline.py)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        network = get_peft_model(network, lora_config)
        
        # Load LoRA weights
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
            path_to_bounds=os.path.join(
                os.path.dirname(args.weight_file), "bounds.npz"
            ),
            augmentations=None,
            split=config["training"].get("splits", ["train", "val"]),
            max_length=config["network"]["sample_num_points"],
            include_room_mask=config["network"].get("room_mask_condition", True),
        )
        first = train_encoded[0]
        overfit_sample = {
            k: torch.from_numpy(v)[None]   # add batch dim
            for k, v in first.items()
            if isinstance(v, np.ndarray)
        }

    # generate
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

    # save – identical structure to generate_results.py
    threed_front_results = ThreedFrontResults(
        raw_train_dataset, raw_dataset, config, sampled_indices, layout_list
    )
    pickle.dump(threed_front_results, open(path_to_results, "wb"))
    print("Saved result to:", path_to_results)

    # kl_divergence = threed_front_results.kl_divergence()
    # print("Object category KL divergence:", kl_divergence)


if __name__ == "__main__":
    main(sys.argv[1:])
