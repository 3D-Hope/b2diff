"""
Inference script for LoRA checkpoint using pretrained noise
For Generative Reach experiment: compare x0_before vs x0_after
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
# Go up 3 levels: scripts/inference/inference_lora.py -> project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.insert(0, project_root)
from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
from utils.utils import seed_everything

# ============================================================================
# CONFIGURATION - Edit these for different experiments
# ============================================================================
# EXPERIMENT_NAME = "uniform_all_20"  # Name for this experiment
# EXPERIMENT_NAME = "last_10" 
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "uniform_10")


def env_or_default(name, default):
    value = os.environ.get(name)
    return value if value else default

CONFIG = {
    # Paths
    "pretrained_latents_path": env_or_default(
        "PRETRAINED_LATENTS_PATH",
        "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/baseline_sd14_samples_10steps/pretrained_latents.pt",
    ),
    # "lora_checkpoint_path": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/lora_ckpt/model/lora/vanilla_ddpo/stage0/checkpoints/checkpoint_1",
    # "lora_checkpoint_path": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/last_10/stage1/checkpoints/checkpoint_1",
    "lora_checkpoint_path": env_or_default(
        "LORA_CHECKPOINT_PATH",
        "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/uniform_10/stage1/checkpoints/checkpoint_1",
    ),
    
    "base_model": "CompVis/stable-diffusion-v1-4",
    "output_dir": env_or_default("OUTPUT_DIR", f"outputs/lora_inference_{EXPERIMENT_NAME}"),
    
    # Generation settings (must match pretrained)
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "eta": 1.0,
    "batch_size": 4,
    "seed": 42,
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float16,
}

print("=" * 80)
print(f"INFERENCE WITH LORA: {EXPERIMENT_NAME}")
print("=" * 80)
print(f"  LoRA checkpoint: {CONFIG['lora_checkpoint_path']}")
print(f"  Output dir:      {CONFIG['output_dir']}")

device = CONFIG["device"]
print(f"\nUsing device: {device}")

# Create output directory
os.makedirs(os.path.join(CONFIG["output_dir"], "images"), exist_ok=True)

# ============================================================================
# STEP 1: Load Pretrained Data
# ============================================================================
print("\n[STEP 1] Load Pretrained Latents and Prompts")
print(f"Loading from: {CONFIG['pretrained_latents_path']}")

pretrained_data = torch.load(CONFIG["pretrained_latents_path"])
initial_noise = pretrained_data['initial_noise']  # [num_samples, 4, 64, 64]
pretrained_final_latents = pretrained_data['final_latents']  # [num_samples, 4, 64, 64]
pretrained_image_tensor = pretrained_data['image_tensor']  # [num_samples, 3, 512, 512]
pretrained_image_paths = pretrained_data['image_path']  # list of paths

num_samples = initial_noise.shape[0]
print(f"✓ Loaded {num_samples} samples from pretrained data")
print(f"  Initial noise shape: {initial_noise.shape}")
print(f"  Pretrained final latents shape: {pretrained_final_latents.shape}")
print(f"  Pretrained image tensor shape: {pretrained_image_tensor.shape}")

# Load prompts from the baseline output directory (same as used for pretrained generation)
baseline_dir = os.path.dirname(CONFIG["pretrained_latents_path"])
prompts_file = os.path.join(baseline_dir, "prompts.json")
with open(prompts_file, 'r') as f:
    prompts = json.load(f)
print(f"✓ Loaded {len(prompts)} prompts from {prompts_file}")

# ============================================================================
# STEP 2: Load Base Stable Diffusion Pipeline
# ============================================================================
print("\n[STEP 2] Load Base Stable Diffusion Pipeline")
pipeline = StableDiffusionPipeline.from_pretrained(
    CONFIG["base_model"], 
    torch_dtype=CONFIG["dtype"]
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
# STEP 4: Setup and Load LoRA Weights
# ============================================================================
print("\n[STEP 4] Setup and Load LoRA Weights")
# Resolve LoRA checkpoint path (supports either directory or direct safetensors file path)
lora_path_cfg = CONFIG["lora_checkpoint_path"]
if os.path.isdir(lora_path_cfg):
    checkpoint_path = os.path.join(lora_path_cfg, "pytorch_lora_weights.safetensors")
else:
    checkpoint_path = lora_path_cfg

if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
        f"LoRA checkpoint not found. Expected file at: {checkpoint_path}. "
        "Set CONFIG['lora_checkpoint_path'] to either a checkpoint directory or a .safetensors file."
    )

# Let diffusers create/load the correct LoRA processor types/ranks from checkpoint.
# This avoids silent mismatches when checkpoint rank differs from hardcoded defaults.
pipeline.unet.load_attn_procs(checkpoint_path)

# Sanity check: report total absolute LoRA weight magnitude so runs can be compared.
lora_abs_sum = 0.0
lora_param_count = 0
for _, attn_proc in pipeline.unet.attn_processors.items():
    for param_name, param in attn_proc.named_parameters(recurse=True):
        if "lora" in param_name:
            lora_abs_sum += param.detach().abs().sum().item()
            lora_param_count += param.numel()

print(f"✓ LoRA weights loaded from: {checkpoint_path}")
print(f"✓ LoRA params: {lora_param_count:,} | abs-sum: {lora_abs_sum:.4f}")
if lora_param_count == 0:
    raise RuntimeError(
        "No LoRA parameters found after loading checkpoint. "
        "Checkpoint may be incompatible or not a UNet LoRA checkpoint."
    )

# ============================================================================
# STEP 5: Move Models to Device
# ============================================================================
print("\n[STEP 5] Move Models to Device")
device = CONFIG["device"]
dtype = CONFIG["dtype"]
pipeline.vae.to(device, dtype=dtype)
pipeline.text_encoder.to(device, dtype=dtype)
pipeline.unet.to(device, dtype=dtype)
pipeline.unet.eval()
print(f"✓ All models moved to {device} with dtype {dtype}")

# ============================================================================
# STEP 6: Generate Images Using Pretrained Noise
# ============================================================================
print("\n[STEP 6] Generate Images Using Pretrained Initial Noise")
print(f"Generating {num_samples} images...")

# Set seeds for reproducibility
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])

all_final_latents = []  # Store final x0 latents for each sample
all_image_tensors = []  # Store final image tensors for each sample

# Generate negative prompt embeddings once
neg_prompt_ids = pipeline.tokenizer(
    [""],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=pipeline.tokenizer.model_max_length,
).input_ids.to(device)
neg_prompt_embeds = pipeline.text_encoder(neg_prompt_ids)[0]

num_batches = (num_samples + CONFIG['batch_size'] - 1) // CONFIG['batch_size']
print(f"  Total batches: {num_batches}")

for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
    start_idx = batch_idx * CONFIG['batch_size']
    end_idx = min(start_idx + CONFIG['batch_size'], num_samples)
    batch_prompts = prompts[start_idx:end_idx]
    batch_size = len(batch_prompts)
    
    # Use pretrained initial noise instead of generating random noise
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
    
    # Use pretrained noise as initial latents
    latents = batch_noise
    
    # Setup scheduler
    pipeline.scheduler.set_timesteps(CONFIG['num_inference_steps'], device=device)
    timesteps = pipeline.scheduler.timesteps
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, CONFIG["eta"])
    
    # Denoising loop
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
    
    # Store final latents (x0)
    for i in range(batch_size):
        all_final_latents.append(latents[i:i+1].cpu().clone())
    
    # Decode latents to images and keep as tensors
    images = latents_decode(pipeline, latents, device, dtype)
    
    # Store image tensors before converting to PIL
    for i in range(batch_size):
        all_image_tensors.append(images[i:i+1].cpu().clone())
    
    # Convert to PIL for saving
    for i in range(batch_size):
        img_idx = start_idx + i
        image_array = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        img_path = os.path.join(CONFIG["output_dir"], "images", f"{img_idx:05d}.png")
        pil_image.save(img_path)

print(f"✓ Generated {len(all_final_latents)} images")

# ============================================================================
# STEP 4: Save Results (Same Format as Pretrained)
# ============================================================================
print("\n[STEP 4] Save Results")

# Stack all tensors
final_latents_tensor = torch.cat(all_final_latents, dim=0)  # Shape: (num_samples, 4, 64, 64)
image_tensors = torch.cat(all_image_tensors, dim=0)  # Shape: (num_samples, 3, 512, 512)
image_paths = [os.path.join(CONFIG["output_dir"], "images", f"{i:05d}.png") for i in range(num_samples)]

# Save only the 4 required fields (same as pretrained)
torch.save({
    'initial_noise': initial_noise,  # Same as pretrained
    'final_latents': final_latents_tensor,  # New (from LoRA)
    'image_tensor': image_tensors,  # New (from LoRA)
    'image_path': image_paths,  # New paths
}, os.path.join(CONFIG["output_dir"], "lora_latents.pt"))

print(f"✓ Saved initial_noise: {initial_noise.shape} (same as pretrained)")
print(f"✓ Saved final_latents: {final_latents_tensor.shape} (from LoRA)")
print(f"✓ Saved image_tensor: {image_tensors.shape} (from LoRA)")
print(f"✓ Saved image_path: {len(image_paths)} paths")
print(f"✓ All data saved to {CONFIG['output_dir']}/lora_latents.pt")

# Save prompts
with open(os.path.join(CONFIG["output_dir"], "prompts.json"), 'w') as f:
    json.dump(prompts, f, indent=2)
print(f"✓ Prompts saved to {CONFIG['output_dir']}/prompts.json")

# Clean up
del pipeline
torch.cuda.empty_cache()
print("✓ Model unloaded")

print("\n" + "=" * 80)
print("✅ INFERENCE COMPLETE!")
print("=" * 80)
