#!/usr/bin/env python
"""
Baseline evaluation: generate images from the pretrained SD model (NO LoRA / checkpoint)
and compute reward scores (geometric or CLIP).

Usage:
    # Geometric reward
    python scripts/inference/eval_pretrained_geometric_reward.py \
        --output_dir outputs/pretrained_baseline_geometric \
        --prompt_file configs/prompt/template4_train.json \
        --reward_fn geometric --num_images 120 --batch_size 8

    # CLIP reward
    python scripts/inference/eval_pretrained_geometric_reward.py \
        --output_dir outputs/pretrained_baseline_clip \
        --prompt_file configs/prompt/template4_train.json \
        --reward_fn clip --num_images 120 --batch_size 8
"""

import os
import sys
import json
import gc
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Add project root to path
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.insert(0, project_root)

from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
from utils.utils import seed_everything
from core.utils.geometric_rewards import ImageGeometricReward


def generate_and_evaluate_pretrained(
    output_dir,
    prompt_file,
    reward_fn="geometric",       # "geometric" or "clip"
    num_images=120,
    base_model="CompVis/stable-diffusion-v1-4",
    batch_size=8,
    num_inference_steps=20,
    guidance_scale=5.0,
    eta=1.0,
    seed=42,
    # geometric reward params
    num_samples=5,
    min_length=20.0,
    threshold_c=0.03,
):
    assert reward_fn in ("geometric", "clip"), "--reward_fn must be 'geometric' or 'clip'"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print("=" * 80)
    print(f"PRETRAINED BASELINE — {reward_fn.upper()} REWARD EVALUATION")
    print("=" * 80)
    print(f"Device:       {device}")
    print(f"Base model:   {base_model}")
    print(f"Reward fn:    {reward_fn}")
    print(f"Prompt file:  {prompt_file}")
    print(f"Num images:   {num_images}")
    print(f"Output dir:   {output_dir}")
    print()

    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    print(f"✓ Loaded {len(prompts)} prompts")

    # ------------------------------------------------------------------ #
    # STEP 1: Load pretrained pipeline — NO LoRA, NO checkpoint loading  #
    # ------------------------------------------------------------------ #
    print("\n[STEP 1] Loading pretrained model (no fine-tuning)...")
    pipeline = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    pipeline.vae.to(device, dtype=dtype)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.unet.to(device, dtype=dtype)

    pipeline.unet.eval()
    pipeline.text_encoder.eval()
    pipeline.vae.eval()
    print("✓ Pretrained model ready (vanilla, no LoRA)")

    # ------------------------------------------------------------------ #
    # STEP 2: Generate images                                             #
    # ------------------------------------------------------------------ #
    print("\n[STEP 2] Generating images...")

    neg_prompt_ids = pipeline.tokenizer(
        [""],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    neg_prompt_embeds = pipeline.text_encoder(neg_prompt_ids)[0]

    all_prompts = []
    global_idx = 0
    num_batches = (num_images + batch_size - 1) // batch_size

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, eta)

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        cur_bs = min(batch_size, num_images - batch_idx * batch_size)

        batch_prompts = [
            prompts[(batch_idx * batch_size + i) % len(prompts)]
            for i in range(cur_bs)
        ]

        # Encode text
        prompt_ids = pipeline.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

        neg_embeds = neg_prompt_embeds.repeat(cur_bs, 1, 1)
        combined_embeds = pipeline._encode_prompt(
            None, device, 1, True, None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
        )

        # Prepare latents
        latents = pipeline.prepare_latents(
            cur_bs,
            pipeline.unet.config.in_channels,
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
            prompt_embeds.dtype,
            device,
            None,
        )

        # Denoising loop
        with torch.no_grad():
            for t in timesteps:
                latent_input = torch.cat([latents] * 2)
                latent_input = pipeline.scheduler.scale_model_input(latent_input, t)

                noise_pred = pipeline.unet(
                    latent_input, t,
                    encoder_hidden_states=combined_embeds,
                    return_dict=False,
                )[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                latents, _, _ = ddim_step_with_logprob(
                    pipeline.scheduler, noise_pred, t, latents, **extra_step_kwargs
                )

        # Decode and save
        images = latents_decode(pipeline, latents, device, prompt_embeds.dtype)
        for i, img_tensor in enumerate(images):
            arr = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(output_dir, f"{global_idx:05d}.png"))
            all_prompts.append(batch_prompts[i])
            global_idx += 1

    print(f"✓ Generated {global_idx} images → {output_dir}")

    # Save prompts list
    with open(os.path.join(output_dir, "prompts.json"), "w") as f:
        json.dump(all_prompts, f, indent=2)

    # Free GPU memory before reward computation
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------ #
    # STEP 3: Compute reward                                              #
    # ------------------------------------------------------------------ #
    eval_list = sorted(f for f in os.listdir(output_dir) if f.endswith(".png"))

    if reward_fn == "geometric":
        print("\n[STEP 3] Computing geometric reward scores...")
        reward_calculator = ImageGeometricReward(min_length=min_length)

        raw_rewards = []
        for img_name in tqdm(eval_list, desc="Scoring"):
            try:
                image = np.array(Image.open(os.path.join(output_dir, img_name)).convert("RGB"))
                reward = reward_calculator.get_algebraic_intersection_reward(
                    image, num_samples=num_samples, threshold_c=threshold_c
                )
            except Exception as e:
                print(f"  Warning: failed on {img_name}: {e}")
                reward = 0.0
            raw_rewards.append(float(reward))

    else:  # clip
        print("\n[STEP 3] Computing CLIP reward scores...")
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2B-s32B-b79K"
        )
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        clip_model = clip_model.to(device).eval()
        print("✓ CLIP model loaded")

        raw_rewards = []
        eval_batch = 8
        for i in tqdm(range(0, len(eval_list), eval_batch), desc="Scoring"):
            batch_imgs = eval_list[i: i + eval_batch]
            batch_prompts = all_prompts[i: i + eval_batch]
            images_t = torch.stack([
                clip_preprocess(Image.open(os.path.join(output_dir, f)))
                for f in batch_imgs
            ]).to(device)
            text_t = clip_tokenizer(batch_prompts).to(device)
            with torch.no_grad():
                img_feat = clip_model.encode_image(images_t)
                txt_feat = clip_model.encode_text(text_t)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                n = len(img_feat)
                sims = (img_feat @ txt_feat.T)[torch.arange(n), torch.arange(n)]
            raw_rewards.extend(sims.cpu().tolist())

    scores = torch.tensor(raw_rewards)

    # ------------------------------------------------------------------ #
    # STEP 4: Print & save results                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print(f"RESULTS — PRETRAINED BASELINE ({reward_fn.upper()} Reward)")
    print("=" * 80)
    print(f"  Num images : {len(raw_rewards)}")
    print(f"  Mean       : {scores.mean():.4f}")
    print(f"  Std        : {scores.std():.4f}")
    print(f"  Min        : {scores.min():.4f}")
    print(f"  Max        : {scores.max():.4f}")
    print("=" * 80)

    # Flat format matching inference_lora_clip_reward.py output
    results = {
        "checkpoint": "pretrained (no LoRA)",
        "num_images": len(raw_rewards),
        f"{reward_fn}_reward_mean": float(scores.mean()),
        f"{reward_fn}_reward_std": float(scores.std()),
        f"{reward_fn}_reward_min": float(scores.min()),
        f"{reward_fn}_reward_max": float(scores.max()),
        "all_scores": raw_rewards,
        "prompts": all_prompts,
    }

    out_path = os.path.join(output_dir, f"{reward_fn}_rewards.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {out_path}")

    return float(scores.mean()), float(scores.std())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate from pretrained SD and evaluate with geometric or CLIP reward"
    )
    parser.add_argument("--output_dir", type=str,
                        default="pretrained_outputs/pretrained_baseline",
                        help="Where to save images and results")
    parser.add_argument("--prompt_file", type=str,
                        default="configs/prompt/template4_train.json",
                        help="JSON list of prompts")
    parser.add_argument("--reward_fn", type=str, default="geometric",
                        choices=["geometric", "clip"],
                        help="Reward function to use for evaluation")
    parser.add_argument("--num_images", type=int, default=120,
                        help="Total images to generate (aim for ≥5x num prompts)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    # Geometric reward params — only used when --reward_fn geometric
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--min_length", type=float, default=20.0)
    parser.add_argument("--threshold_c", type=float, default=0.03)

    args = parser.parse_args()

    generate_and_evaluate_pretrained(
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        reward_fn=args.reward_fn,
        num_images=args.num_images,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_samples=args.num_samples,
        min_length=args.min_length,
        threshold_c=args.threshold_c,
    )
