#!/usr/bin/env python
"""
Annealed Importance Guidance (AIG) inference.

At each denoising timestep the noise prediction is a weighted mix of:
  - the finetuned (LoRA) UNet score  — drives reward
  - the base (pretrained) UNet score — preserves diversity

Mixing weight γ(t) follows the annealing schedule from Jena et al. WACV 2025:

    t_norm  = t / num_train_timesteps        (1 at pure noise, 0 at clean image)
    γ(t)    = (1 - t_norm) ** aig_beta       (0 at start → 1 at end)

    noise_pred = γ(t) * noise_pred_lora + (1 - γ(t)) * noise_pred_base

Higher aig_beta → sharper annealing → more reward-oriented.
Paper evaluates β ∈ {1, 2, 3, 4}; default β=2 is a good starting point.

Usage (from repo root):
    python scripts/inference/inference_lora_aig.py \
        --checkpoint_path outputs/incremental_branch_lambda_2_fk_4particles/stage97 \
        --output_dir     outputs/template1_aig/stage0 \
        --prompt_file    configs/prompt/template1_train.json \
        --num_images     1080 \
        --batch_size     4 \
        --aig_beta       2.0
"""

import os
import sys
import json
import gc
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np

script_path  = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(project_root)

from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
from utils.utils import seed_everything
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from accelerate import Accelerator
from diffusers.loaders import AttnProcsLayers
import contextlib


def generate_and_evaluate_aig(
    checkpoint_path,
    output_dir,
    prompt_file,
    num_images=1080,
    base_model="CompVis/stable-diffusion-v1-4",
    batch_size=4,
    num_inference_steps=20,
    guidance_scale=5.0,
    eta=1.0,
    seed=300,
    aig_beta=2.0,
):
    """
    Generate images using Annealed Importance Guidance and compute CLIP rewards.

    Args:
        checkpoint_path : Path to LoRA checkpoint directory (accelerate format)
        output_dir      : Directory to save generated images and results
        prompt_file     : Path to JSON prompt list
        num_images      : Total images to generate
        base_model      : HuggingFace base model id
        batch_size      : Generation batch size
        num_inference_steps : DDIM steps
        guidance_scale  : CFG weight
        eta             : DDIM eta
        seed            : RNG seed
        aig_beta        : Annealing exponent β (higher → more reward-biased)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16

    print("=" * 80)
    print("AIG INFERENCE + CLIP REWARD COMPUTATION")
    print("=" * 80)
    print(f"Device           : {device}")
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Images to gen    : {num_images}")
    print(f"AIG beta         : {aig_beta}")
    print()

    seed_everything(seed)

    # Load prompts
    print("Loading prompts...")
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    print(f"  Loaded {len(prompts)} prompts")

    os.makedirs(output_dir, exist_ok=True)

    # ── Accelerator (needed for LoRA loading) ─────────────────────────────────
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        project_config=ProjectConfiguration(project_dir="."),
    )

    # ── [STEP 1] Load finetuned (LoRA) pipeline ───────────────────────────────
    print("\n[STEP 1] Load Finetuned LoRA Pipeline")
    pipeline_lora = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=dtype
    )
    pipeline_lora.vae.requires_grad_(False)
    pipeline_lora.text_encoder.requires_grad_(False)
    pipeline_lora.unet.requires_grad_(False)
    pipeline_lora.safety_checker = None
    pipeline_lora.scheduler = DDIMScheduler.from_config(pipeline_lora.scheduler.config)

    pipeline_lora.vae.to(accelerator.device, dtype=dtype)
    pipeline_lora.text_encoder.to(accelerator.device, dtype=dtype)
    pipeline_lora.unet.to(accelerator.device, dtype=dtype)

    # Attach LoRA processors
    lora_attn_procs = {}
    for name in pipeline_lora.unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor")
            else pipeline_lora.unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = pipeline_lora.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id   = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipeline_lora.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id   = int(name[len("down_blocks.")])
            hidden_size = pipeline_lora.unet.config.block_out_channels[block_id]
        else:
            hidden_size = pipeline_lora.unet.config.block_out_channels[0]
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
        )
    pipeline_lora.unet.set_attn_processor(lora_attn_procs)
    trainable_layers = AttnProcsLayers(pipeline_lora.unet.attn_processors)

    def save_model_hook(models, weights, output_dir_):
        assert len(models) == 1
        if isinstance(models[0], AttnProcsLayers):
            pipeline_lora.unet.save_attn_procs(output_dir_)
        weights.pop()

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                base_model, revision="main", subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        models.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    trainable_layers = accelerator.prepare(trainable_layers)
    accelerator.load_state(checkpoint_path)
    pipeline_lora.unet.eval()
    pipeline_lora.text_encoder.eval()
    pipeline_lora.vae.eval()
    print("  LoRA pipeline loaded")

    # ── [STEP 2] Load base (pretrained, no LoRA) UNet ─────────────────────────
    print("\n[STEP 2] Load Base (Pretrained) UNet for AIG")
    unet_base = UNet2DConditionModel.from_pretrained(
        base_model, revision="main", subfolder="unet",
        torch_dtype=dtype,
    )
    unet_base.requires_grad_(False)
    unet_base.eval()
    unet_base.to(accelerator.device)
    print("  Base UNet loaded")

    # ── [STEP 3] Generate Images with AIG ─────────────────────────────────────
    print("\n[STEP 3] Generate Images with AIG")

    pipeline_lora.scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
    timesteps        = pipeline_lora.scheduler.timesteps   # descending: T→0
    num_train_steps  = pipeline_lora.scheduler.config.num_train_timesteps
    extra_step_kwargs = pipeline_lora.prepare_extra_step_kwargs(None, eta)

    # Negative prompt embeddings (once)
    neg_ids = pipeline_lora.tokenizer(
        [""],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline_lora.tokenizer.model_max_length,
    ).input_ids.to(accelerator.device)
    neg_embeds = pipeline_lora.text_encoder(neg_ids)[0]

    num_batches   = (num_images + batch_size - 1) // batch_size
    all_images    = []
    all_prompts   = []
    global_idx    = 0
    autocast      = contextlib.nullcontext

    for batch_idx in tqdm(range(num_batches), desc="AIG generate"):
        current_bs    = min(batch_size, num_images - batch_idx * batch_size)
        batch_prompts = [
            prompts[(batch_idx * batch_size + i) % len(prompts)]
            for i in range(current_bs)
        ]

        # Encode text
        prompt_ids = pipeline_lora.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline_lora.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = pipeline_lora.text_encoder(prompt_ids)[0]

        # CFG embeddings (uncond + cond) for both models
        sample_neg = neg_embeds.repeat(current_bs, 1, 1)
        embeds_cfg  = pipeline_lora._encode_prompt(
            None, accelerator.device, 1, True, None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=sample_neg,
        )

        # Initial latent noise
        latents = pipeline_lora.prepare_latents(
            current_bs,
            pipeline_lora.unet.config.in_channels,
            pipeline_lora.unet.config.sample_size * pipeline_lora.vae_scale_factor,
            pipeline_lora.unet.config.sample_size * pipeline_lora.vae_scale_factor,
            prompt_embeds.dtype,
            accelerator.device,
            None,
        )

        with autocast():
            with torch.no_grad():
                for t in timesteps:
                    # ── AIG mixing weight ────────────────────────────────────
                    # t is in [0, num_train_timesteps]; high t = noisy
                    t_norm = float(t.item()) / num_train_steps   # ∈ (0,1]
                    gamma  = (1.0 - t_norm) ** aig_beta          # 0→1 as t→0

                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = pipeline_lora.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # ── Finetuned (LoRA) score ───────────────────────────────
                    noise_pred_lora = pipeline_lora.unet(
                        latent_model_input, t,
                        encoder_hidden_states=embeds_cfg,
                        return_dict=False,
                    )[0]
                    noise_uncond_lora, noise_text_lora = noise_pred_lora.chunk(2)
                    noise_pred_lora = noise_uncond_lora + guidance_scale * (
                        noise_text_lora - noise_uncond_lora
                    )

                    # ── Base score ───────────────────────────────────────────
                    noise_pred_base = unet_base(
                        latent_model_input, t,
                        encoder_hidden_states=embeds_cfg,
                        return_dict=False,
                    )[0]
                    noise_uncond_base, noise_text_base = noise_pred_base.chunk(2)
                    noise_pred_base = noise_uncond_base + guidance_scale * (
                        noise_text_base - noise_uncond_base
                    )

                    # ── AIG interpolation ────────────────────────────────────
                    noise_pred_aig = gamma * noise_pred_lora + (1.0 - gamma) * noise_pred_base

                    # ── DDIM step ────────────────────────────────────────────
                    latents, _, _ = ddim_step_with_logprob(
                        pipeline_lora.scheduler,
                        noise_pred_aig, t, latents,
                        **extra_step_kwargs,
                    )

        # Decode to images
        images = latents_decode(
            pipeline_lora, latents, accelerator.device, prompt_embeds.dtype
        )

        for i, img_tensor in enumerate(images):
            img_arr = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_arr)
            img_path = os.path.join(output_dir, f"{global_idx:05d}.png")
            pil_img.save(img_path)
            all_images.append(img_tensor)
            all_prompts.append(batch_prompts[i])
            global_idx += 1

    print(f"  Generated {len(all_images)} images")

    # Save prompts
    with open(os.path.join(output_dir, "prompts.json"), "w") as f:
        json.dump(all_prompts, f, indent=2)

    # Free generation models
    del pipeline_lora, unet_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── [STEP 4] Compute CLIP Rewards ─────────────────────────────────────────
    print("\n[STEP 4] Compute CLIP Rewards")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2B-s32B-b79K"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    clip_model     = clip_model.to(accelerator.device)
    clip_model.eval()
    print("  CLIP model loaded")

    similarity_scores = []
    eval_bs           = 8
    num_eval_batches  = (len(all_images) + eval_bs - 1) // eval_bs

    for batch_idx in tqdm(range(num_eval_batches), desc="CLIP eval"):
        start = batch_idx * eval_bs
        end   = min(start + eval_bs, len(all_images))

        batch_imgs = []
        batch_txts = all_prompts[start:end]
        for idx in range(start, end):
            img = Image.open(os.path.join(output_dir, f"{idx:05d}.png"))
            batch_imgs.append(clip_preprocess(img))

        img_input  = torch.stack(batch_imgs).to(accelerator.device)
        txt_input  = clip_tokenizer(batch_txts).to(accelerator.device)

        with torch.no_grad():
            img_feat = clip_model.encode_image(img_input)
            txt_feat = clip_model.encode_text(txt_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            n = len(img_feat)
            sims = (img_feat @ txt_feat.T)[torch.arange(n), torch.arange(n)]
            similarity_scores.extend(sims.cpu().tolist())

    scores_tensor = torch.tensor(similarity_scores)
    mean_reward   = scores_tensor.mean().item()
    std_reward    = scores_tensor.std().item()

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Images          : {len(all_images)}")
    print(f"AIG beta        : {aig_beta}")
    print(f"CLIP Mean       : {mean_reward:.4f}")
    print(f"CLIP Std        : {std_reward:.4f}")
    print(f"CLIP Min/Max    : {scores_tensor.min():.4f} / {scores_tensor.max():.4f}")
    print("=" * 80)

    results = {
        "checkpoint":        checkpoint_path,
        "aig_beta":          aig_beta,
        "num_images":        len(all_images),
        "clip_reward_mean":  mean_reward,
        "clip_reward_std":   std_reward,
        "clip_reward_min":   float(scores_tensor.min()),
        "clip_reward_max":   float(scores_tensor.max()),
        "all_scores":        similarity_scores,
        "prompts":           all_prompts,
    }
    results_file = os.path.join(output_dir, "clip_rewards.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    return mean_reward, std_reward


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AIG inference: interpolate finetuned+base scores at inference time"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to LoRA checkpoint directory (accelerate format)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write generated images and clip_rewards.json",
    )
    parser.add_argument(
        "--prompt_file", type=str,
        default="configs/prompt/template1_train.json",
        help="Path to JSON prompt list",
    )
    parser.add_argument("--num_images",          type=int,   default=1080)
    parser.add_argument("--batch_size",          type=int,   default=4)
    parser.add_argument("--num_inference_steps", type=int,   default=20)
    parser.add_argument("--guidance_scale",      type=float, default=5.0)
    parser.add_argument("--eta",                 type=float, default=1.0)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument(
        "--aig_beta", type=float, default=2.0,
        help="Annealing exponent β. Higher = more reward-biased. Paper tests 1–4.",
    )

    args = parser.parse_args()

    mean_reward, _ = generate_and_evaluate_aig(
        checkpoint_path      = args.checkpoint_path,
        output_dir           = args.output_dir,
        prompt_file          = args.prompt_file,
        num_images           = args.num_images,
        batch_size           = args.batch_size,
        num_inference_steps  = args.num_inference_steps,
        guidance_scale       = args.guidance_scale,
        eta                  = args.eta,
        seed                 = args.seed,
        aig_beta             = args.aig_beta,
    )

    from run_inception_score import get_inception_score
    get_inception_score(args.output_dir)
