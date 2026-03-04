"""
Core sampling module for B2Diff training pipeline.
Extracts the sampling logic from run_sample.py into a callable function.
"""

import os
import sys
import torch
import contextlib
import pickle
import json
import random
import numpy as np
from functools import partial
from tqdm import tqdm as tqdm_lib
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
from utils.utils import post_processing, seed_everything

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def _load_midiff_cfg(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PATH_TO_DATASET_FILES = os.path.join(PROJ_DIR, "../ThreedFront/dataset_files/")
PATH_TO_PROCESSED_DATA = os.path.join(PROJ_DIR, "../ThreedFront/output/3d_front_processed/")

def _update_data_paths(config_data):
    config_data["dataset_directory"] = \
        os.path.join(PATH_TO_PROCESSED_DATA, config_data["dataset_directory"])
    config_data["annotation_file"] = \
        os.path.join(PATH_TO_DATASET_FILES, config_data["annotation_file"])
    return config_data

tqdm = partial(tqdm_lib, dynamic_ncols=True)


def descale_to_origin(x, minimum, maximum):
    """x: (B, N, 3); minimum/maximum: (3,)"""
    x = (x + 1) / 2
    x = x * (maximum - minimum)[None, None, :] + minimum[None, None, :]
    return x


def descale_pos(positions, pos_min=None, pos_max=None, device="cuda", room_type="livingroom"):
    """Descale positions (B, N, 3) to world coordinates."""
    if pos_min is None:
        if room_type == "bedroom":
            pos_min = torch.tensor([-2.7625005,  0.045,     -2.75275  ], device=device)
        elif room_type == "livingroom":
            pos_min = torch.tensor([-5.672918693230125, 0.0375, -5.716401580065309], device=device)
        else:
            raise ValueError(f"Unknown room type: {room_type}")
    if pos_max is None:
        if room_type == "bedroom":
            pos_max = torch.tensor([2.7784417, 3.6248395, 2.8185427], device=device)
        elif room_type == "livingroom":
            pos_max = torch.tensor([5.09667921844729, 3.3577405149437496, 5.4048500000000015], device=device)
        else:
            raise ValueError(f"Unknown room type: {room_type}")
    return descale_to_origin(positions, pos_min, pos_max)


def descale_size(sizes, size_min=None, size_max=None, device="cuda", room_type="livingroom"):
    """Descale sizes (B, N, 3) to world coordinates (returns HALF-EXTENTS)."""
    if size_min is None:
        if room_type == "bedroom":
            size_min = torch.tensor([0.03998289, 0.02000002, 0.012772  ], device=device)
        elif room_type == "livingroom":
            size_min = torch.tensor([0.03998999999999997, 0.020000020334800084, 0.0328434999999998], device=device)
        else:
            raise ValueError(f"Unknown room type: {room_type}")
    if size_max is None:
        if room_type == "bedroom":
            size_max = torch.tensor([2.8682, 1.770065, 1.698315], device=device)
        elif room_type == "livingroom":
            size_max = torch.tensor([2.3802699999999994, 1.7700649999999998, 1.3224289999999996], device=device)
        else:
            raise ValueError(f"Unknown room type: {room_type}")
    return descale_to_origin(sizes, size_min, size_max)


def parse_and_descale_scenes(scenes, num_classes=22, parse_only=False, room_type="livingroom"):
    """Parse (B, N, 30) scene tensor and descale positions/sizes to world coordinates.

    Returns dict with keys: one_hot, positions, sizes (HALF-EXTENTS), orientations,
    object_indices, is_empty, device.
    """
    device = scenes.device
    positions_normalized = scenes[:, :, 0:3]
    sizes_normalized     = scenes[:, :, 3:6]
    orientations         = scenes[:, :, 6:8]   # [cos_theta, sin_theta]
    one_hot              = scenes[:, :, 8:8 + num_classes]

    if not parse_only:
        positions = descale_pos(positions_normalized, device=device, room_type=room_type)
        sizes     = descale_size(sizes_normalized,    device=device, room_type=room_type)
    else:
        positions = positions_normalized
        sizes     = sizes_normalized

    object_indices  = torch.argmax(one_hot, dim=-1)
    empty_class_idx = num_classes - 1
    is_empty        = object_indices == empty_class_idx

    return {
        "one_hot":        one_hot,
        "positions":      positions,
        "sizes":          sizes,
        "orientations":   orientations,
        "object_indices": object_indices,
        "is_empty":       is_empty,
        "device":         device,
    }


def run_sampling(config, stage_idx=None, logger=None, wandb_run=None, pipeline=None, trainable_layers=None, resume_from_ckpt=False):
    """
    Run the sampling phase for a given stage.
    
    Args:
        config: Configuration object (can be OmegaConf or dict)
        stage_idx: Current stage index (optional, for logging)
        logger: Logger instance (optional)
        wandb_run: Existing wandb run to log to (optional)
        pipeline: Pre-loaded StableDiffusionPipeline (avoids reloading)
        trainable_layers: Pre-initialized trainable layers
        
    Returns:
        save_dir: Directory where samples were saved
    """
    # Convert OmegaConf to dict if needed
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config
        
    if logger:
        logger.info(f"Starting sampling for stage {stage_idx}")
    else:
        print(f"Starting sampling for stage {stage_idx}")
    
    print(f'========== seed: {config.seed} ==========') 
    torch.cuda.set_device(config.dev_id)
    # Setup directories
    unique_id = config.exp_name
    os.makedirs(os.path.join(config.save_path, unique_id), exist_ok=True)
    
    # Use stage index for directory, not run_name
    stage_id = f"stage{stage_idx}"
    save_dir = os.path.join(config.save_path, unique_id, stage_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Use pre-loaded pipeline (no loading!)
    if pipeline is None:
        raise ValueError("Pipeline must be provided - should not reload model each stage!")
    
    # Setup accelerator
    accelerator_config = ProjectConfiguration(
        project_dir=save_dir,
        automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config
    )
    
    # IMPORTANT: Do NOT use log_with="wandb" - it tries to init its own run
    # All logging goes through the parent pipeline's wandb_run
    
    # # Setup inference dtype
    # inference_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     inference_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     inference_dtype = torch.bfloat16
    if getattr(config, 'threed_scene_layout', False):
        # MiDiffusion save/load hooks – LoRA-aware.
        # When use_lora=True  → save only the lora_ parameters (tiny delta weights).
        # When use_lora=False → save only the denoising transformer (pipeline.model),
        #                       the floor encoder is frozen so it never needs saving.
        def save_model_hook(models, weights, output_dir):
            assert len(models) == 1
            if config.use_lora:
                # Extract only parameters whose names contain "lora_"
                lora_state = {
                    k: v for k, v in models[0].state_dict().items()
                    if "lora_" in k
                }
                torch.save(lora_state, os.path.join(output_dir, "lora_weights.pt"))
            else:
                # Full fine-tune: trainable_layers == pipeline.model (denoising transformer)
                torch.save(models[0].state_dict(), os.path.join(output_dir, "model.pt"))
            weights.pop()

        def load_model_hook(models, input_dir):
            assert len(models) == 1
            if config.use_lora:
                lora_state = torch.load(
                    os.path.join(input_dir, "lora_weights.pt"), map_location="cpu"
                )
                # Load into the full model; strict=False so frozen params are untouched
                models[0].load_state_dict(lora_state, strict=False)
            else:
                state = torch.load(
                    os.path.join(input_dir, "model.pt"), map_location="cpu"
                )
                models[0].load_state_dict(state)
            models.pop()
    else:
        def save_model_hook(models, weights, output_dir):
            assert len(models) == 1
            if config.use_lora and isinstance(models[0], AttnProcsLayers):
                pipeline.unet.save_attn_procs(output_dir)
            elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
                models[0].save_pretrained(os.path.join(output_dir, "unet"))
            else:
                raise ValueError(f"Unknown model type {type(models[0])}")
            weights.pop()

        def load_model_hook(models, input_dir):
            assert len(models) == 1
            if config.use_lora and isinstance(models[0], AttnProcsLayers):
                tmp_unet = UNet2DConditionModel.from_pretrained(
                    config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
                )
                tmp_unet.load_attn_procs(input_dir)
                models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
                del tmp_unet
            elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                models[0].register_to_config(**load_model.config)
                models[0].load_state_dict(load_model.state_dict())
                del load_model
            else:
                raise ValueError(f"Unknown model type {type(models[0])}")
            models.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    

    
    # Prepare trainable layers (if not already prepared)
    if trainable_layers is not None and not hasattr(trainable_layers, '_hf_hook'):
        trainable_layers = accelerator.prepare(trainable_layers)
    
    # Load checkpoint if resume_from_ckpt is True
    if resume_from_ckpt:
        print("loading model. Please Wait.")
        # Build checkpoint path from previous stage
        prev_stage = stage_idx - 1
        checkpoint_num = config.train.num_epochs // config.train.save_interval
        checkpoint_path = os.path.join(
            config.save_path,
            config.exp_name,
            f"stage{prev_stage}",
            "checkpoints",
            f"checkpoint_{checkpoint_num}"
        )
        
        checkpoint_path = os.path.normpath(os.path.expanduser(checkpoint_path))
        if "checkpoint_" not in os.path.basename(checkpoint_path):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(checkpoint_path)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {checkpoint_path}")
            checkpoint_path = os.path.join(
                checkpoint_path,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        print(f"Resuming from {checkpoint_path}")
        if getattr(config, 'threed_scene_layout', False):
            # MiDiffusion: load LoRA deltas or full transformer weights directly
            if config.use_lora:
                lora_pt = os.path.join(checkpoint_path, "lora_weights.pt")
                state = torch.load(lora_pt, map_location=accelerator.device)
                pipeline.load_state_dict(state, strict=False)
            else:
                model_pt = os.path.join(checkpoint_path, "model.pt")
                state = torch.load(model_pt, map_location=accelerator.device)
                pipeline.model.load_state_dict(state)
        else:
            accelerator.load_state(checkpoint_path)
        print("load successfully!")
    
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Set seed
    seed_everything(config.seed)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    if config.sample.always_branch_at > 0:
        split_steps = [config.sample.always_branch_at]
    else:
        split_steps = [config.split_step]
    split_times = [config.split_time]

    # ------------------------------------------------------------------
    # 3D scene layout path  (MiDiffusion + DDIM)
    # ------------------------------------------------------------------
    if getattr(config, 'threed_scene_layout', False):
        # Build a DDIM scheduler for 3D (DDIMScheduler is already imported).
        # ddim_step_with_logprob is shape-agnostic via _left_broadcast so it
        # handles (B, N, C) scene tensors the same way as (B, C, H, W) latents.
        ddim_3d = DDIMScheduler(
            num_train_timesteps=getattr(config.midiffusion, 'num_timesteps', 1000),
            beta_start=getattr(config.midiffusion, 'beta_start', 1e-4),
            beta_end=getattr(config.midiffusion, 'beta_end', 0.02),
            clip_sample=False,
            prediction_type="epsilon",
            steps_offset=1,
        )

        # Load floor conditions  [F, 256, 4]
        with open(config.midiffusion.floor_conditions) as _f:
            _fc = json.load(_f)
        floor_cond_np = np.array(_fc, dtype=np.float32)  # (F, 256, 4)
        floor_cnt = len(floor_cond_np)

        num_objects = getattr(config.midiffusion, 'num_objects', 12)
        scene_dim   = getattr(config.midiffusion, 'scene_dim',   30)

        pipeline.eval()
        samples = []

        total_fpbpn   = []   # tracks which floor condition each sample used
        total_samples = None

        if os.path.exists(os.path.join(save_dir, 'fpbpn_list.json')):
            with open(os.path.join(save_dir, 'fpbpn_list.json'), 'r') as _f:
                total_fpbpn = json.load(_f)
        if os.path.exists(os.path.join(save_dir, 'sample.pkl')):
            with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as _f:
                total_samples = pickle.load(_f)

        global_idx = len(total_fpbpn)
        local_idx  = 0

        # ---- Load MiDiffusion datasets for ThreedFrontResults-compatible saving ----
        # import sys as _sys
        # _midiff_scripts_dir = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        #     "3d_layout_generation", "MiDiffusion", "scripts"
        # )
        # if _midiff_scripts_dir not in _sys.path:
        #     _sys.path.insert(0, _midiff_scripts_dir)
        # from threed_front.evaluation import ThreedFrontResults
        # from midiffusion.datasets.threed_front_encoding import get_dataset_raw_and_encoded as _get_ds_raw_enc
        # from threed_front.datasets import get_raw_dataset as _get_raw_ds

        # _midiff_cnf = _load_midiff_cfg(config.midiffusion.config_path)
        # if "_eval" not in _midiff_cnf["data"]["encoding_type"]:
        #     _midiff_cnf["data"]["encoding_type"] += "_eval"
        # _data_cfg = _update_data_paths(_midiff_cnf["data"])
        # print("[intermediate saving] Loading train/test datasets for ThreedFrontResults ...")
        # _raw_train_ds = _get_raw_ds(
        #     _data_cfg,
        #     split=_midiff_cnf["training"].get("splits", ["train", "val"]),
        #     include_room_mask=_midiff_cnf["network"].get("room_mask_condition", True),
        # )
        # _raw_test_ds, _enc_dataset = _get_ds_raw_enc(
        #     _data_cfg,
        #     split=_midiff_cnf["validation"].get("splits", ["test"]),
        #     max_length=_midiff_cnf["network"]["sample_num_points"],
        #     include_room_mask=_midiff_cnf["network"].get("room_mask_condition", True),
        # )
        # _n_obj_types = _enc_dataset.n_object_types
        # print(f"[intermediate saving] Datasets loaded. n_object_types={_n_obj_types}")

        # def _delete_empty_from_x0(x0_batch, n_object_types):
        #     """Convert (B, N, 30) raw x0 tensor → list[B] of layout dicts
        #     with empty objects removed (same logic as ashok_generate_results.py)."""
        #     bbox_dim   = 8
        #     class_dim  = x0_batch.shape[-1] - bbox_dim
        #     N          = x0_batch.shape[1]
        #     translations = x0_batch[..., 0:3]
        #     sizes        = x0_batch[..., 3:6]
        #     angles       = x0_batch[..., 6:8]   # [cos θ, sin θ]
        #     class_raw    = x0_batch[..., bbox_dim:]
        #     class_scores = class_raw[..., :n_object_types]
        #     obj_max, obj_max_ind = torch.max(class_scores, dim=-1)
        #     empty_logit  = class_raw[..., class_dim - 1]
        #     is_empty     = empty_logit > obj_max
        #     class_onehot = torch.nn.functional.one_hot(
        #         obj_max_ind, num_classes=n_object_types
        #     ).float()
        #     B_loc = x0_batch.shape[0]
        #     boxes_list = []
        #     for b_loc in range(B_loc):
        #         box = {
        #             "translations": torch.zeros(1, 0, 3),
        #             "sizes":        torch.zeros(1, 0, 3),
        #             "angles":       torch.zeros(1, 0, 2),
        #             "class_labels": torch.zeros(1, 0, n_object_types),
        #         }
        #         for i in range(N):
        #             if is_empty[b_loc, i]:
        #                 continue
        #             box["translations"] = torch.cat(
        #                 [box["translations"], translations[b_loc:b_loc+1, i:i+1, :].cpu()], dim=1)
        #             box["sizes"]        = torch.cat(
        #                 [box["sizes"],        sizes[b_loc:b_loc+1, i:i+1, :].cpu()],        dim=1)
        #             box["angles"]       = torch.cat(
        #                 [box["angles"],       angles[b_loc:b_loc+1, i:i+1, :].cpu()],       dim=1)
        #             box["class_labels"] = torch.cat(
        #                 [box["class_labels"], class_onehot[b_loc:b_loc+1, i:i+1, :].cpu()], dim=1)
        #         boxes_list.append(box)
        #     return boxes_list

        for idx in tqdm(
            range(config.sample.num_batches_per_epoch),
            disable=not accelerator.is_local_main_process,
            position=0,
            desc="Sampling batches",
        ):
            # Randomly sample floor conditions for each item in the batch
            batch_indices = [random.randrange(floor_cnt) for _ in range(config.sample.batch_size)]
            fpbpn_batch   = torch.tensor(
                floor_cond_np[batch_indices], dtype=torch.float32, device=accelerator.device
            )  # (B, 256, 4)

            # Initial pure noise  (B, N, C)
            x_noise = torch.randn(
                config.sample.batch_size, num_objects, scene_dim,
                dtype=torch.float32, device=accelerator.device,
            )

            ddim_3d.set_timesteps(config.sample.num_steps, device=accelerator.device)
            ts = ddim_3d.timesteps
            # print(f"DDIM timesteps: {ts.cpu().numpy()}")
            extra_step_kwargs = {"eta": getattr(config.sample, 'eta', 0.0)}
            # print(f"DDIM extra step kwargs: {extra_step_kwargs}")
            if config.sample.no_branching:
                branch_num  = split_times[0]
                noise_list  = [x_noise] + [torch.randn_like(x_noise) for _ in range(branch_num - 1)]
                scenes      = [[n] for n in noise_list]
                log_probs   = [[] for _ in range(branch_num)]
            else:
                scenes    = [[x_noise]]
                log_probs = [[]]

            for i, t in tqdm(
                enumerate(ts),
                desc="Timestep",
                position=3,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):
                with autocast():
                    with torch.no_grad():
                        if not config.sample.no_branching and ((config.sample.num_steps - i) in split_steps):
                            branch_num = split_times[split_steps.index(config.sample.num_steps - i)]
                            cur_num    = len(scenes)
                            scenes    = [[s for s in scenes[k // branch_num]] for k in range(cur_num * branch_num)]
                            log_probs = [[lp for lp in log_probs[k // branch_num]] for k in range(cur_num * branch_num)]

                        for k in range(len(scenes)):
                            x_t     = scenes[k][i]
                            t_batch = t.expand(config.sample.batch_size)  # (B,)
                            noise_pred = pipeline.predict_noise(x_t, t_batch, fpbpn_batch)
                            # ddim_step_with_logprob is shape-agnostic; works on (B, N, C)
                            x_prev, log_prob, _ = ddim_step_with_logprob(
                                ddim_3d, noise_pred, t, x_t, **extra_step_kwargs
                            )
                            scenes[k].append(x_prev)
                            log_probs[k].append(log_prob)

            sample_num = len(scenes)
            total_fpbpn.extend([batch_indices] * sample_num)

            for k in range(sample_num):
                store_scenes    = torch.stack(scenes[k], dim=1)     # (B, T+1, N, C)
                store_log_probs = torch.stack(log_probs[k], dim=1)  # (B, T)
                timesteps_rep   = ts.unsqueeze(0).expand(config.sample.batch_size, -1)  # (B, T)

                samples.append({
                    "fpbpn":       fpbpn_batch.cpu().detach(),
                    "timesteps":   timesteps_rep.cpu().detach(),
                    "log_probs":   store_log_probs.cpu().detach(),
                    "scenes":      store_scenes[:, :-1].cpu().detach(),   # (B, T, N, C)
                    "next_scenes": store_scenes[:, 1:].cpu().detach(),   # (B, T, N, C)
                })

            if (idx + 1) % config.sample.save_interval == 0 or idx == (config.sample.num_batches_per_epoch - 1):
                keys = ["fpbpn", "timesteps", "log_probs", "scenes", "next_scenes"]
                new_tensors = {k: torch.cat([s[k] for s in samples]) for k in keys}
                local_idx   = len(new_tensors["scenes"])

                with open(os.path.join(save_dir, 'fpbpn_list.json'), 'w') as _f:
                    json.dump(total_fpbpn, _f)
                with open(os.path.join(save_dir, 'sample.pkl'), 'wb') as _f:
                    if total_samples is None:
                        pickle.dump(new_tensors, _f)
                    else:
                        pickle.dump(
                            {k: torch.cat([total_samples[k], new_tensors[k]]) for k in keys},
                            _f
                        )

        # ---------------------------------------------------------------
        # Save final scenes as ThreedFrontResults for rendering
        # ---------------------------------------------------------------
        # if getattr(config.sample, 'save_train_samples_no_train', False):
        #     print("[save_train_samples_no_train] Building ThreedFrontResults from final scenes …")

        #     all_sampled_indices = []
        #     all_layouts         = []

        #     # `samples` has one dict per (batch, branch-k) appended in order.
        #     # `total_fpbpn[global_idx + j]` gives the B floor-plan indices for samples[j].
        #     for sample_j, batch_s in enumerate(samples):
        #         # Final fully-denoised scenes: (B, N, C)
        #         final_scenes_batch  = batch_s["next_scenes"][:, -1].float()
        #         B_cur               = final_scenes_batch.shape[0]
        #         fpbpn_batch_indices = total_fpbpn[global_idx + sample_j]

        #         boxes_list = _delete_empty_from_x0(final_scenes_batch, _n_obj_types)
        #         for b_loc in range(B_cur):
        #             fpbpn_idx = fpbpn_batch_indices[b_loc]
        #             bbox_dict = boxes_list[b_loc]
        #             processed = _enc_dataset.post_process(bbox_dict)
        #             all_layouts.append({k: v.numpy()[0] for k, v in processed.items()})
        #             all_sampled_indices.append(fpbpn_idx)

        #     result = ThreedFrontResults(
        #         _raw_train_ds, _raw_test_ds, _midiff_cnf,
        #         all_sampled_indices, all_layouts,
        #     )
        #     out_path = os.path.join(save_dir, 'final_best_samples.pkl')
        #     with open(out_path, 'wb') as _pf:
        #         pickle.dump(result, _pf)
        #     print(f"[save_train_samples_no_train] Saved {len(all_layouts)} layouts → {out_path}")
        #     if logger:
        #         logger.info(
        #             f"[save_train_samples_no_train] Saved {len(all_layouts)} layouts → {out_path}"
        #         )

        if wandb_run:
            wandb_run.log({
                f"sampling/stage_{stage_idx}/total_samples": len(total_fpbpn),
                f"sampling/stage_{stage_idx}/batches": config.sample.num_batches_per_epoch,
            })
        if logger:
            logger.info(f"Sampling completed for stage {stage_idx}")
        return save_dir

    # ------------------------------------------------------------------
    # 2D image (Stable Diffusion) path
    # ------------------------------------------------------------------
    # Generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)

    # SAMPLING LOOP
    pipeline.unet.eval()
    samples = []

    total_prompts = []
    total_samples = None

    # Load existing prompts and samples if available
    if os.path.exists(os.path.join(save_dir, 'prompt.json')):
        with open(os.path.join(save_dir, 'prompt.json'), 'r') as f:
            total_prompts = json.load(f)

    if os.path.exists(os.path.join(save_dir, 'sample.pkl')):
        with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as f:
            total_samples = pickle.load(f)

    global_idx = len(total_prompts)
    local_idx = 0

    # Load prompts
    prompt_list = []
    if len(config.prompt) == 0:  # prompt = "" by default
        prompt_file_path = config.prompt_file
        if not os.path.isabs(prompt_file_path):
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            prompt_file_path = os.path.join(project_root, prompt_file_path)
        with open(prompt_file_path) as f:
            prompt_list = json.load(f)
    prompt_idx = 0
    prompt_cnt = len(prompt_list)

    # Main sampling loop
    for idx in tqdm(
        range(config.sample.num_batches_per_epoch),
        disable=not accelerator.is_local_main_process,
        position=0,
        desc="Sampling batches"
    ):
        # generate prompts
        if len(config.prompt) != 0:
            prompts1 = [config.prompt for _ in range(config.sample.batch_size)]
        elif config.prompt_random_choose:
            prompts1 = [random.choice(prompt_list) for _ in range(config.sample.batch_size)]
        else:
            prompts1 = [prompt_list[(prompt_idx+i)%prompt_cnt] for i in range(config.sample.batch_size)]
            prompt_idx += config.sample.batch_size

        # Encode prompts
        prompt_ids1 = pipeline.tokenizer(
            prompts1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]

        # Combine prompt and negative prompt
        prompt_embeds1_combine = pipeline._encode_prompt(
            None,
            accelerator.device,
            1,
            config.sample.cfg,
            None,
            prompt_embeds=prompt_embeds1,
            negative_prompt_embeds=sample_neg_prompt_embeds
        )

        # Prepare latents
        noise_latents1 = pipeline.prepare_latents(
            config.sample.batch_size,
            pipeline.unet.config.in_channels,
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
            prompt_embeds1.dtype,
            accelerator.device,
            None
        )

        pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
        ts = pipeline.scheduler.timesteps
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, config.sample.eta)

        # For no_branching mode, prepare multiple different initial noises
        if config.sample.no_branching:
            branch_num = split_times[0]
            noise_latents_list = [noise_latents1]
            for _ in range(branch_num - 1):
                additional_noise = pipeline.prepare_latents(
                    config.sample.batch_size,
                    pipeline.unet.config.in_channels,
                    pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                    pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                    prompt_embeds1.dtype,
                    accelerator.device,
                    None
                )
                noise_latents_list.append(additional_noise)
            latents = [[noise] for noise in noise_latents_list]
            log_probs = [[] for _ in range(branch_num)]
        else:
            latents = [[noise_latents1]]
            log_probs = [[]]

        for i, t in tqdm(
            enumerate(ts),
            desc="Timestep",
            position=3,
            leave=False,
            disable=not accelerator.is_local_main_process,
        ):
            with autocast():
                with torch.no_grad():
                    if not config.sample.no_branching and ((config.sample.num_steps-i) in split_steps):
                        branch_num = split_steps.index(config.sample.num_steps-i)
                        branch_num = split_times[branch_num]
                        cur_sample_num = len(latents)
                        latents = [
                            [latent for latent in latents[k//branch_num]]
                            for k in range(cur_sample_num*branch_num)
                        ]
                        log_probs = [
                            [log_prob for log_prob in log_probs[k//branch_num]]
                            for k in range(cur_sample_num*branch_num)
                        ]

                    for k in range(len(latents)):
                        latents_t = latents[k][i]
                        latents_input = torch.cat([latents_t] * 2) if config.sample.cfg else latents_t
                        latents_input = pipeline.scheduler.scale_model_input(latents_input, t)

                        noise_pred = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds1_combine,
                            return_dict=False,
                        )[0]

                        if config.sample.cfg:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        latents_t_1, log_prob, latents_0 = ddim_step_with_logprob(pipeline.scheduler, noise_pred, t, latents_t, **extra_step_kwargs)

                        latents[k].append(latents_t_1)
                        log_probs[k].append(log_prob)

        sample_num = len(latents)
        total_prompts.extend(prompts1*sample_num)

        for k in range(sample_num):
            images = latents_decode(pipeline, latents[k][config.sample.num_steps], accelerator.device, prompt_embeds1.dtype)
            store_latents = torch.stack(latents[k], dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            store_log_probs = torch.stack(log_probs[k], dim=1)  # (batch_size, num_steps)
            prompt_embeds = prompt_embeds1
            current_latents = store_latents[:, :-1]
            next_latents = store_latents[:, 1:]
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            samples.append({
                "prompt_embeds": prompt_embeds.cpu().detach(),
                "timesteps":     timesteps.cpu().detach(),
                "log_probs":     store_log_probs.cpu().detach(),
                "latents":       current_latents.cpu().detach(),
                "next_latents":  next_latents.cpu().detach(),
                "images":        images.cpu().detach()
            })

        if (idx+1) % config.sample.save_interval == 0 or idx == (config.sample.num_batches_per_epoch-1):
            os.makedirs(os.path.join(save_dir, "images/"), exist_ok=True)
            print(f'-----------{accelerator.process_index} save image start-----------')
            new_samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            images = new_samples['images'][local_idx:]
            for j, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil.save(os.path.join(save_dir, f"images/{(j+global_idx):05}.png"))

            global_idx += len(images)
            local_idx  += len(images)

            with open(os.path.join(save_dir, 'prompt.json'), 'w') as f:
                json.dump(total_prompts, f)
            with open(os.path.join(save_dir, 'sample.pkl'), 'wb') as f:
                if total_samples is None:
                    pickle.dump({
                        "prompt_embeds": new_samples["prompt_embeds"],
                        "timesteps":     new_samples["timesteps"],
                        "log_probs":     new_samples["log_probs"],
                        "latents":       new_samples["latents"],
                        "next_latents":  new_samples["next_latents"],
                    }, f)
                else:
                    pickle.dump({
                        "prompt_embeds": torch.cat([total_samples["prompt_embeds"], new_samples["prompt_embeds"]]),
                        "timesteps":     torch.cat([total_samples["timesteps"],     new_samples["timesteps"]]),
                        "log_probs":     torch.cat([total_samples["log_probs"],     new_samples["log_probs"]]),
                        "latents":       torch.cat([total_samples["latents"],       new_samples["latents"]]),
                        "next_latents":  torch.cat([total_samples["next_latents"],  new_samples["next_latents"]]),
                    }, f)

    # Log sampling completion to parent wandb run
    if wandb_run:
        wandb_run.log({
            f"sampling/stage_{stage_idx}/total_samples": len(total_prompts),
            f"sampling/stage_{stage_idx}/batches": config.sample.num_batches_per_epoch,
        })

    if logger:
        logger.info(f"Sampling completed for stage {stage_idx}")

    return save_dir
