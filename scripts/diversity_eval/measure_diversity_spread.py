#!/usr/bin/env python
"""
Diversity-spread evaluation script for the iADD rebuttal.

Measures three metrics at a given checkpoint with FIXED seeds so results
are directly comparable across methods and training stages:

  1. CLIP score mean + std        — reward distribution spread
  2. Intra-class LPIPS            — mean pairwise LPIPS within each prompt group
  3. Intra-class CLIP distance    — mean pairwise cosine distance in CLIP image
                                    embedding space (proxy for Jacobian volume
                                    preservation; directly answers R1 MW2)

Usage:
    python scripts/diversity_eval/measure_diversity_spread.py \
        --checkpoint_path model/lora/iadd_full/stage20/checkpoints/checkpoint_1 \
        --method_name iadd_full \
        --stage 20 \
        --output_dir outputs/diversity_eval \
        --num_images_per_prompt 24 \
        --prompt_file configs/prompt/template1_train.json
"""

import os
import sys
import json
import gc
import argparse
import itertools
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.insert(0, project_root)

from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
from utils.utils import seed_everything


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_pipeline_base(base_model: str, device: torch.device):
    """Load the frozen SD pipeline (no LoRA yet)."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.vae.to(device, dtype=torch.float16)
    pipeline.text_encoder.to(device, dtype=torch.float16)
    pipeline.unet.to(device, dtype=torch.float16)
    return pipeline


def _load_lora_peft(pipeline, checkpoint_path: str):
    """Load LoRA weights saved by the PEFT-based training (pytorch_lora_weights.pt)."""
    from peft import LoraConfig, set_peft_model_state_dict
    lora_cfg = LoraConfig(
        r=4, lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipeline.unet.add_adapter(lora_cfg)
    weights_path = os.path.join(checkpoint_path, "pytorch_lora_weights.pt")
    state_dict = torch.load(weights_path, map_location="cpu")
    set_peft_model_state_dict(pipeline.unet, state_dict)
    print(f"  [LoRA] loaded PEFT weights from {weights_path}")
    return pipeline


def _load_lora_attnprocs(pipeline, checkpoint_path: str, base_model: str, device: torch.device):
    """Load LoRA weights saved by the old AttnProcsLayers-based training."""
    from diffusers import UNet2DConditionModel
    from diffusers.loaders import AttnProcsLayers
    from diffusers.models.attention_processor import LoRAAttnProcessor
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration

    lora_attn_procs = {}
    for name in pipeline.unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor")
            else pipeline.unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = pipeline.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipeline.unet.config.block_out_channels[block_id]
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )
    pipeline.unet.set_attn_processor(lora_attn_procs)
    trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)

    # Use accelerator just for loading state
    acc_cfg = ProjectConfiguration(
        project_dir=checkpoint_path, automatic_checkpoint_naming=True, total_limit=1
    )
    accelerator = Accelerator(mixed_precision="fp16", project_config=acc_cfg,
                               gradient_accumulation_steps=1)

    def _save_hook(models, weights, output_dir):
        pipeline.unet.save_attn_procs(output_dir)
        weights.pop()

    def _load_hook(models, input_dir):
        tmp = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
        tmp.load_attn_procs(input_dir)
        models[0].load_state_dict(AttnProcsLayers(tmp.attn_processors).state_dict())
        del tmp
        models.pop()

    accelerator.register_save_state_pre_hook(_save_hook)
    accelerator.register_load_state_pre_hook(_load_hook)
    trainable_layers = accelerator.prepare(trainable_layers)
    accelerator.load_state(checkpoint_path)
    print(f"  [LoRA] loaded AttnProcs weights from {checkpoint_path}")
    return pipeline


def load_model(checkpoint_path: str, base_model: str, device: torch.device):
    """Auto-detect checkpoint format and load accordingly."""
    pipeline = _load_pipeline_base(base_model, device)

    peft_file   = os.path.join(checkpoint_path, "pytorch_lora_weights.pt")
    attn_file   = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")

    if os.path.exists(peft_file):
        pipeline = _load_lora_peft(pipeline, checkpoint_path)
    elif os.path.exists(attn_file):
        pipeline = _load_lora_attnprocs(pipeline, checkpoint_path, base_model, device)
    else:
        print(f"  [LoRA] WARNING: no lora weights found in {checkpoint_path}. "
              "Using pretrained base model.")
    pipeline.unet.eval()
    return pipeline


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_images(pipeline, prompts, num_per_prompt: int, device: torch.device,
                    num_steps: int = 20, guidance_scale: float = 5.0,
                    eta: float = 1.0, base_seed: int = 0):
    """
    Generate `num_per_prompt` images per prompt using fixed, reproducible seeds.
    Returns:
        images_by_prompt : dict[prompt_str -> list[PIL.Image]]
        global_images    : list of all PIL.Image (same order as global seeds)
        global_prompts   : list[str] parallel to global_images
    """
    import contextlib
    pipeline.scheduler.set_timesteps(num_steps, device=device)
    ts = pipeline.scheduler.timesteps
    extra_kwargs = pipeline.prepare_extra_step_kwargs(None, eta)

    # Pre-compute negative prompt embed once
    neg_ids = pipeline.tokenizer(
        [""], return_tensors="pt", padding="max_length", truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    neg_embed = pipeline.text_encoder(neg_ids)[0]  # (1, seq, d)

    images_by_prompt = {p: [] for p in prompts}
    global_images, global_prompts = [], []

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        # Encode prompt
        p_ids = pipeline.tokenizer(
            [prompt], return_tensors="pt", padding="max_length",
            truncation=True, max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)
        p_embed = pipeline.text_encoder(p_ids)[0]   # (1, seq, d)

        for k in range(num_per_prompt):
            # Fixed, deterministic seed: ensures same noise for every method
            seed = base_seed + prompt_idx * 1000 + k
            torch.manual_seed(seed)

            latent = pipeline.prepare_latents(
                1,
                pipeline.unet.config.in_channels,
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                p_embed.dtype, device, None,
            )

            combined = pipeline._encode_prompt(
                None, device, 1, True, None,
                prompt_embeds=p_embed,
                negative_prompt_embeds=neg_embed,
            )

            with torch.no_grad():
                for t in ts:
                    inp = torch.cat([latent] * 2)
                    inp = pipeline.scheduler.scale_model_input(inp, t)
                    noise_pred = pipeline.unet(inp, t,
                                              encoder_hidden_states=combined,
                                              return_dict=False)[0]
                    uncond, cond = noise_pred.chunk(2)
                    noise_pred = uncond + guidance_scale * (cond - uncond)
                    latent, _, _ = ddim_step_with_logprob(
                        pipeline.scheduler, noise_pred, t, latent, **extra_kwargs
                    )

            # Decode
            decoded = latents_decode(pipeline, latent, device, p_embed.dtype)
            arr = (decoded[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            images_by_prompt[prompt].append(img)
            global_images.append(img)
            global_prompts.append(prompt)

    return images_by_prompt, global_images, global_prompts


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_clip_metrics(global_images, global_prompts, device: torch.device):
    """Compute CLIP text-image similarity scores → mean, std."""
    import open_clip
    clip_model, _, clip_prep = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2B-s32B-b79K"
    )
    clip_tok = open_clip.get_tokenizer("ViT-H-14")
    clip_model = clip_model.to(device).eval()

    scores = []
    batch_size = 16
    for i in tqdm(range(0, len(global_images), batch_size), desc="CLIP scores"):
        imgs   = global_images[i:i+batch_size]
        txts   = global_prompts[i:i+batch_size]
        img_t  = torch.stack([clip_prep(im) for im in imgs]).to(device)
        txt_t  = clip_tok(txts).to(device)
        with torch.no_grad():
            img_f = clip_model.encode_image(img_t)
            txt_f = clip_model.encode_text(txt_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            s = (img_f * txt_f).sum(dim=-1)
        scores.extend(s.cpu().tolist())
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std()), scores.tolist()


def compute_clip_image_diversity(images_by_prompt, device: torch.device):
    """
    Mean pairwise cosine distance between CLIP image embeddings within each
    prompt group, then averaged across prompts.

    This is the proxy for Jacobian volume preservation:
    high value = output distribution volume maintained.
    """
    import open_clip
    clip_model, _, clip_prep = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2B-s32B-b79K"
    )
    clip_model = clip_model.to(device).eval()

    per_prompt_dists = []
    for prompt, imgs in tqdm(images_by_prompt.items(), desc="CLIP diversity"):
        img_t = torch.stack([clip_prep(im) for im in imgs]).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(img_t)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # (N, d)

        # Pairwise cosine similarity → distance = 1 - sim
        sim_mat = (feats @ feats.T).cpu().numpy()  # (N, N)
        N = len(imgs)
        # Take upper triangle (exclude diagonal)
        idx = np.triu_indices(N, k=1)
        pairwise_dists = 1.0 - sim_mat[idx]
        per_prompt_dists.append(float(pairwise_dists.mean()))

    mean_dist = float(np.mean(per_prompt_dists))
    std_dist  = float(np.std(per_prompt_dists))
    return mean_dist, std_dist, per_prompt_dists


def compute_lpips_diversity(images_by_prompt, device: torch.device):
    """
    Intra-class LPIPS: mean pairwise LPIPS distance within each prompt group,
    averaged across prompts.  Higher = more diverse.
    """
    import lpips
    loss_fn = lpips.LPIPS(net="vgg").to(device)

    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    per_prompt_lpips = []
    for prompt, imgs in tqdm(images_by_prompt.items(), desc="LPIPS diversity"):
        tensors = [pil_to_tensor(im).to(device) for im in imgs]
        N = len(tensors)
        vals = []
        for i, j in itertools.combinations(range(N), 2):
            with torch.no_grad():
                d = loss_fn(tensors[i], tensors[j]).item()
            vals.append(d)
        per_prompt_lpips.append(float(np.mean(vals)))

    mean_lpips = float(np.mean(per_prompt_lpips))
    std_lpips  = float(np.std(per_prompt_lpips))
    return mean_lpips, std_lpips, per_prompt_lpips


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure diversity spread at a given checkpoint (fixed seeds)."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--method_name", type=str, required=True,
                        help="Label for this method, e.g. 'iadd_full'")
    parser.add_argument("--stage", type=int, required=True,
                        help="Stage number (for logging/output naming)")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/diversity_eval",
                        help="Root output directory")
    parser.add_argument("--prompt_file", type=str,
                        default="configs/prompt/template1_train.json")
    parser.add_argument("--num_images_per_prompt", type=int, default=24,
                        help="Images per prompt (24 → 1080 for 45 prompts)")
    parser.add_argument("--base_model", type=str,
                        default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--base_seed", type=int, default=0,
                        help="Base seed — keep identical across all runs")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--skip_lpips", action="store_true",
                        help="Skip LPIPS computation (faster, use if lpips not installed)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  Diversity Spread Evaluation")
    print(f"  Method : {args.method_name}  |  Stage : {args.stage}")
    print(f"  Checkpoint : {args.checkpoint_path}")
    print(f"{'='*70}\n")

    # Load prompts
    with open(args.prompt_file) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    # Output paths
    run_out = os.path.join(args.output_dir, args.method_name, f"stage{args.stage}")
    img_dir = os.path.join(run_out, "images")
    os.makedirs(img_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load model
    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading model...")
    pipeline = load_model(args.checkpoint_path, args.base_model, device)

    # ------------------------------------------------------------------ #
    # 2. Generate images with fixed seeds
    # ------------------------------------------------------------------ #
    print(f"\n[2/4] Generating {len(prompts) * args.num_images_per_prompt} images "
          f"({args.num_images_per_prompt} per prompt)...")
    images_by_prompt, global_images, global_prompts = generate_images(
        pipeline, prompts,
        num_per_prompt=args.num_images_per_prompt,
        device=device,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        base_seed=args.base_seed,
    )

    # Save images to disk
    idx = 0
    for prompt, imgs in images_by_prompt.items():
        for img in imgs:
            img.save(os.path.join(img_dir, f"{idx:05d}.png"))
            idx += 1
    print(f"  Saved {idx} images to {img_dir}")

    # Free GPU memory before CLIP
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------ #
    # 3. CLIP metrics (score + image-space diversity)
    # ------------------------------------------------------------------ #
    print("\n[3/4] Computing CLIP metrics...")
    clip_mean, clip_std, all_scores = compute_clip_metrics(
        global_images, global_prompts, device
    )
    print(f"  CLIP score:  mean={clip_mean:.4f}  std={clip_std:.4f}")

    clip_div_mean, clip_div_std, per_prompt_clip_div = compute_clip_image_diversity(
        images_by_prompt, device
    )
    print(f"  CLIP image diversity (volume proxy):  "
          f"mean={clip_div_mean:.4f}  std={clip_div_std:.4f}")

    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------ #
    # 4. Intra-class LPIPS
    # ------------------------------------------------------------------ #
    lpips_mean, lpips_std, per_prompt_lpips = None, None, None
    if not args.skip_lpips:
        print("\n[4/4] Computing intra-class LPIPS...")
        try:
            lpips_mean, lpips_std, per_prompt_lpips = compute_lpips_diversity(
                images_by_prompt, device
            )
            print(f"  Intra-class LPIPS:  mean={lpips_mean:.4f}  std={lpips_std:.4f}")
        except ImportError:
            print("  WARNING: lpips package not installed. Skipping. "
                  "Install with: pip install lpips")
    else:
        print("\n[4/4] LPIPS skipped (--skip_lpips)")

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    results = {
        "method": args.method_name,
        "stage": args.stage,
        "checkpoint_path": args.checkpoint_path,
        "num_prompts": len(prompts),
        "num_images_per_prompt": args.num_images_per_prompt,
        "total_images": len(global_images),
        "base_seed": args.base_seed,
        # CLIP reward distribution
        "clip_score_mean": clip_mean,
        "clip_score_std": clip_std,
        # CLIP image-space diversity (volume-preservation proxy for R1 MW2)
        "clip_image_diversity_mean": clip_div_mean,
        "clip_image_diversity_std": clip_div_std,
        "per_prompt_clip_diversity": per_prompt_clip_div,
        # Intra-class LPIPS (requested by R1 MW3)
        "intra_lpips_mean": lpips_mean,
        "intra_lpips_std": lpips_std,
        "per_prompt_lpips": per_prompt_lpips,
        # Raw scores for plotting
        "all_clip_scores": all_scores,
    }

    results_path = os.path.join(run_out, "diversity_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  SUMMARY  —  {args.method_name}  stage {args.stage}")
    print(f"{'='*70}")
    print(f"  CLIP score mean        : {clip_mean:.4f}  ± {clip_std:.4f}")
    print(f"  CLIP image diversity   : {clip_div_mean:.4f}  ± {clip_div_std:.4f}")
    if lpips_mean is not None:
        print(f"  Intra-class LPIPS      : {lpips_mean:.4f}  ± {lpips_std:.4f}")
    print(f"  Results saved to       : {results_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
