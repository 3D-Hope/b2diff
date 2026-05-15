"""
Generate 1080-sample pretrained baseline for Proposition 1 experiment
Uses CompVis/sd-v1-4 without any LoRA to establish reference noise and final latents.
Output saved in same format as 45-sample baseline for drop-in compatibility.
"""
import os
import sys
import json
import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Add project root to path
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.insert(0, project_root)
from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
from utils.utils import seed_everything

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "base_model": "CompVis/stable-diffusion-v1-4",
    "output_dir": os.environ.get(
        "BASELINE_OUTPUT_DIR",
        "outputs/baseline_sd14_samples_1080_10steps",
    ),
    
    # Generation settings (must match LoRA inference)
    "num_inference_steps": 10,
    "guidance_scale": 5.0,
    "eta": 1.0,
    "batch_size": 32,  # Larger batch size for 1080 samples
    "seed": 42,
    "num_samples": 1080,
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float16,
}

print("=" * 80)
print("GENERATE PRETRAINED BASELINE (1080 samples)")
print("=" * 80)
print(f"  Output dir:          {CONFIG['output_dir']}")
print(f"  Num samples:         {CONFIG['num_samples']}")
print(f"  Num inference steps: {CONFIG['num_inference_steps']}")
print(f"  Batch size:          {CONFIG['batch_size']}")
print(f"  Using device:        {CONFIG['device']}")

device = CONFIG["device"]

# Create output directory
os.makedirs(os.path.join(CONFIG["output_dir"], "images"), exist_ok=True)

# ============================================================================
# STEP 1: Generate Prompts and Initial Noise
# ============================================================================
print("\n[STEP 1] Generate Prompts and Initial Noise")

# Set seed for reproducibility
seed_everything(CONFIG["seed"])

# Generate diverse prompts (1080 samples)
base_prompts = [
    "a photograph of a dog",
    "a photograph of a cat",
    "a painting of a mountain",
    "a painting of a sunset",
    "a photograph of a person",
    "a photograph of a tree",
    "a photograph of a car",
    "a photograph of a building",
]

# Cycle through and expand to 1080 with minor variations
prompts = []
for i in range(CONFIG["num_samples"]):
    prompt_idx = i % len(base_prompts)
    prompts.append(base_prompts[prompt_idx])

print(f"✓ Generated {len(prompts)} prompts")

# Generate random initial noise (1080 x 4 x 64 x 64)
initial_noise = torch.randn(
    CONFIG["num_samples"],
    4,  # latent channels
    64,  # latent height
    64,  # latent width
    dtype=CONFIG["dtype"],
)
print(f"✓ Generated random initial noise: {initial_noise.shape}")

# ============================================================================
# STEP 2: Load Base Stable Diffusion Pipeline
# ============================================================================
print("\n[STEP 2] Load Base Stable Diffusion Pipeline")
pipeline = StableDiffusionPipeline.from_pretrained(
    CONFIG["base_model"],
    torch_dtype=CONFIG["dtype"],
)
pipeline.safety_checker = None
pipeline.vae.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
pipeline.unet.requires_grad_(False)
print(f"✓ Base model loaded: {CONFIG['base_model']}")

# ============================================================================
# STEP 3: Setup DDIM Scheduler
# ============================================================================
print("\n[STEP 3] Setup DDIM Scheduler")
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
print(f"✓ DDIM scheduler configured with {CONFIG['num_inference_steps']} steps")

# ============================================================================
# STEP 4: Move Models to Device
# ============================================================================
print("\n[STEP 4] Move Models to Device")
pipeline.vae.to(device, dtype=CONFIG["dtype"])
pipeline.text_encoder.to(device, dtype=CONFIG["dtype"])
pipeline.unet.to(device, dtype=CONFIG["dtype"])
pipeline.unet.eval()
print(f"✓ All models moved to {device}")

# ============================================================================
# STEP 5: Generate Images Using Random Noise (No LoRA)
# ============================================================================
print("\n[STEP 5] Generate Images from Pretrained Model (No LoRA)")
print(f"Generating {CONFIG['num_samples']} images in batches of {CONFIG['batch_size']}...")

all_final_latents = []
all_image_tensors = []

# Generate negative prompt embeddings once
neg_prompt_ids = pipeline.tokenizer(
    [""],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=pipeline.tokenizer.model_max_length,
).input_ids.to(device)
neg_prompt_embeds = pipeline.text_encoder(neg_prompt_ids)[0]

num_batches = (CONFIG["num_samples"] + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
print(f"  Total batches: {num_batches}")

for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
    start_idx = batch_idx * CONFIG["batch_size"]
    end_idx = min(start_idx + CONFIG["batch_size"], CONFIG["num_samples"])
    batch_prompts = prompts[start_idx:end_idx]
    batch_size = len(batch_prompts)
    
    # Get random initial noise for this batch
    batch_noise = initial_noise[start_idx:end_idx].to(device)
    
    # Encode prompts
    prompt_ids = pipeline.tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
    
    # Combine with negative prompts for CFG
    batch_neg_embeds = neg_prompt_embeds.repeat(batch_size, 1, 1)
    prompt_embeds_combined = torch.cat([batch_neg_embeds, prompt_embeds])
    
    # Use random noise as initial latents
    latents = batch_noise
    
    # Setup scheduler
    pipeline.scheduler.set_timesteps(CONFIG["num_inference_steps"], device=device)
    timesteps = pipeline.scheduler.timesteps
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, CONFIG["eta"])
    
    # Denoising loop (no LoRA applied)
    with torch.no_grad():
        for t in timesteps:
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_combined,
                return_dict=False,
            )[0]
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + CONFIG["guidance_scale"] * (
                noise_pred_text - noise_pred_uncond
            )
            
            # DDIM step
            latents, _, _ = ddim_step_with_logprob(
                pipeline.scheduler, noise_pred, t, latents, **extra_step_kwargs
            )
    
    # Store final latents
    for i in range(batch_size):
        all_final_latents.append(latents[i:i+1].cpu().clone())
    
    # Decode latents to images
    images = latents_decode(pipeline, latents, device, CONFIG["dtype"])
    
    # Store image tensors
    for i in range(batch_size):
        all_image_tensors.append(images[i:i+1].cpu().clone())
    
    # Save images as PNG
    for i in range(batch_size):
        img_idx = start_idx + i
        image_array = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        img_path = os.path.join(CONFIG["output_dir"], "images", f"{img_idx:05d}.png")
        pil_image.save(img_path)

print(f"✓ Generated {len(all_final_latents)} images")

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n[STEP 6] Save Results")

final_latents_tensor = torch.cat(all_final_latents, dim=0)
image_tensors = torch.cat(all_image_tensors, dim=0)
image_paths = [
    os.path.join(CONFIG["output_dir"], "images", f"{i:05d}.png")
    for i in range(CONFIG["num_samples"])
]

# Save in same format as 45-sample baseline
torch.save(
    {
        "initial_noise": initial_noise,
        "final_latents": final_latents_tensor,
        "image_tensor": image_tensors,
        "image_path": image_paths,
    },
    os.path.join(CONFIG["output_dir"], "pretrained_latents.pt"),
)

print(f"✓ Saved initial_noise: {initial_noise.shape}")
print(f"✓ Saved final_latents: {final_latents_tensor.shape}")
print(f"✓ Saved image_tensor: {image_tensors.shape}")
print(f"✓ Saved image_path: {len(image_paths)} paths")
print(f"✓ All data saved to {CONFIG['output_dir']}/pretrained_latents.pt")

# Save prompts
with open(os.path.join(CONFIG["output_dir"], "prompts.json"), "w") as f:
    json.dump(prompts, f, indent=2)
print(f"✓ Prompts saved to {CONFIG['output_dir']}/prompts.json")

# Clean up
del pipeline
torch.cuda.empty_cache()
print("✓ Model unloaded")

print("\n" + "=" * 80)
print("✅ BASELINE GENERATION COMPLETE!")
print(f"   Baseline saved to: {CONFIG['output_dir']}/pretrained_latents.pt")
print("=" * 80)
