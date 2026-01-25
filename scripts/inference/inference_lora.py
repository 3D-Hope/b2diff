"""
Minimal inference script for trained LoRA checkpoint
Generates images and computes CLIP reward scores
"""
import os
import sys
import json
import pickle
import contextlib
import gc
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

# Add project root to path
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(script_path))
from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
from utils.utils import seed_everything

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Paths
    "checkpoint_path": "outputs/83_pytorch_lora_weights.safetensors",
    "prompt_file": "config/prompt/template1_train.json",
    "output_dir": "outputs/inference_results_83_ckpt",
    "base_model": "CompVis/stable-diffusion-v1-4",
    
    # Generation settings
    "batch_size": 4,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "eta": 1.0,
    "seed": 300,
    "num_samples_per_prompt": 1112,  # Generate N images per prompt (45 prompts * 1112 = 50,040 total)
    
    # Checkpointing
    "checkpoint_every": 100,  # Save progress every N images
    "resume_from_checkpoint": True,  # Continue from last checkpoint if exists
    
    # Evaluation
    "eval_batch_size": 32,  # Batch size for CLIP evaluation
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float16,
}

print("=" * 80)
print("INFERENCE WITH LORA CHECKPOINT")
print("=" * 80)

# ============================================================================
# STEP 1: Setup and Load Prompts
# ============================================================================
print("\n[STEP 1] Setup and Load Prompts")
seed_everything(CONFIG["seed"])
os.makedirs(os.path.join(CONFIG["output_dir"], "images"), exist_ok=True)
# Check for existing checkpoint
checkpoint_file = os.path.join(CONFIG["output_dir"], "checkpoint.json")
start_idx = 0
existing_scores = []
if CONFIG["resume_from_checkpoint"] and os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
    start_idx = checkpoint_data["last_image_idx"] + 1
    existing_scores = checkpoint_data.get("scores", [])
    print(f"✓ Resuming from checkpoint: starting at image {start_idx}")
with open(CONFIG["prompt_file"], 'r') as f:
    prompts = json.load(f)

# Repeat each prompt N times for multiple samples
num_samples = CONFIG["num_samples_per_prompt"]
prompts = [prompt for prompt in prompts for _ in range(num_samples)]

print(f"✓ Loaded {len(prompts) // num_samples} unique prompts")
print(f"✓ Generating {num_samples} samples per prompt = {len(prompts)} total images")
print(f"  First prompt: '{prompts[0]}'")

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
# Create LoRA attention processors
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
        hidden_size=hidden_size, 
        cross_attention_dim=cross_attention_dim
    )

pipeline.unet.set_attn_processor(lora_attn_procs)
print(f"✓ LoRA processors initialized: {len(lora_attn_procs)} layers")

# Load checkpoint weights
checkpoint_path = CONFIG["checkpoint_path"]
if os.path.isfile(checkpoint_path):
    pipeline.unet.load_attn_procs(checkpoint_path)
    print(f"✓ LoRA weights loaded from: {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
# STEP 6: Generate Images (Streaming Mode - Save Only)
# ============================================================================
print("\n[STEP 6] Generate Images (Streaming Mode)")
print(f"  Total prompts: {len(prompts)}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Starting from image: {start_idx}")

all_prompts = []
global_image_idx = 0

# Generate negative prompt embeddings once
neg_prompt_ids = pipeline.tokenizer(
    [""],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=pipeline.tokenizer.model_max_length,
).input_ids.to(device)
neg_prompt_embeds = pipeline.text_encoder(neg_prompt_ids)[0]

num_batches = (len(prompts) + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
print(f"  Total batches: {num_batches}")

for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
    start_idx = batch_idx * CONFIG["batch_size"]
    end_idx = min(start_idx + CONFIG["batch_size"], len(prompts))
    batch_prompts = prompts[start_idx:end_idx]
    batch_size = len(batch_prompts)
    
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
    
    # Initialize latents
    latents = torch.randn(
        batch_size, 
        pipeline.unet.config.in_channels,
        pipeline.unet.config.sample_size,
        pipeline.unet.config.sample_size,
        device=device,
        dtype=dtype
    )
    
    # Setup scheduler
    pipeline.scheduler.set_timesteps(CONFIG["num_inference_steps"], device=device)
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
    
    # Decode latents to images
    images = latents_decode(pipeline, latents, device, dtype)
    
    # Process each image in batch immediately (save only)
    for idx_in_batch, (image_tensor, prompt) in enumerate(zip(images, batch_prompts)):
        # Skip if already processed (resuming from checkpoint)
        if global_image_idx < start_idx:
            global_image_idx += 1
            continue
            
        # Save image immediately
        image_array = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        img_path = os.path.join(CONFIG["output_dir"], "images", f"{global_image_idx:05d}.png")
        pil_image.save(img_path)
        
        all_prompts.append(prompt)
        global_image_idx += 1
    
    # Periodic checkpoint saving
    if global_image_idx % CONFIG["checkpoint_every"] == 0:
        checkpoint_data = {
            "last_image_idx": global_image_idx - 1,
            "timestamp": str(torch.cuda.Event().query() if torch.cuda.is_available() else 0)
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
    
    # Memory cleanup every 10 batches
    if batch_idx % 10 == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

print(f"✓ Generated {len(all_prompts)} images")

# Save prompts
with open(os.path.join(CONFIG["output_dir"], "prompts.json"), 'w') as f:
    json.dump(all_prompts, f, indent=2)
print(f"✓ Prompts saved to {CONFIG['output_dir']}/prompts.json")

# Clean up generation models to free memory
del pipeline
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("✓ Generation models unloaded")

# ============================================================================
# STEP 7: Load CLIP Model for Evaluation
# ============================================================================
print("\n[STEP 7] Load CLIP Model for Evaluation")
import open_clip

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-H-14', 
    pretrained='laion2B-s32B-b79K'
)
clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
clip_model = clip_model.to(device)
clip_model.eval()
print("✓ CLIP model loaded: ViT-H-14")

# ============================================================================
# STEP 8: Compute CLIP Similarity Scores
# ============================================================================
print("\n[STEP 8] Compute CLIP Similarity Scores")
print(f"  Processing {len(all_prompts)} images...")

similarity_scores = []

eval_batch_size = CONFIG["eval_batch_size"]
num_eval_batches = (len(all_prompts) + eval_batch_size - 1) // eval_batch_size

for batch_idx in tqdm(range(num_eval_batches), desc="Evaluating images"):
    start_idx = batch_idx * eval_batch_size
    end_idx = min(start_idx + eval_batch_size, len(all_prompts))
    
    # Load and preprocess images
    batch_images = []
    batch_prompts = all_prompts[start_idx:end_idx]
    
    for idx in range(start_idx, end_idx):
        img_path = os.path.join(CONFIG["output_dir"], "images", f"{idx:05d}.png")
        img = Image.open(img_path)
        batch_images.append(clip_preprocess(img))
    
    image_input = torch.stack(batch_images).to(device)
    text_input = clip_tokenizer(batch_prompts).to(device)
    
    # Encode images and text
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        
        # Normalize for similarity computation
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity (diagonal elements = same image-text pairs)
        batch_size_actual = len(image_features_norm)
        similarity = (image_features_norm @ text_features_norm.T)[
            torch.arange(batch_size_actual), torch.arange(batch_size_actual)
        ]
        similarity_scores.extend(similarity.cpu().tolist())

# Convert to tensor
similarity_scores = torch.tensor(similarity_scores)

print(f"✓ Computed CLIP similarity scores")
print(f"  Score range: [{similarity_scores.min():.4f}, {similarity_scores.max():.4f}]")
print(f"  Mean score: {similarity_scores.mean():.4f}")
print(f"  Std score: {similarity_scores.std():.4f}")

# Compute per-prompt statistics
num_samples = CONFIG["num_samples_per_prompt"]
scores_reshaped = similarity_scores.view(-1, num_samples)
per_prompt_mean = scores_reshaped.mean(dim=1)
per_prompt_max = scores_reshaped.max(dim=1).values
per_prompt_min = scores_reshaped.min(dim=1).values
print(f"\nPer-Prompt Statistics (across {num_samples} samples):")
print(f"  Mean of means: {per_prompt_mean.mean():.4f}")
print(f"  Mean of maxes: {per_prompt_max.mean():.4f}")

# ============================================================================
# STEP 9: Save Results
# ============================================================================
print("\n[STEP 9] Save Results")

# Clean up checkpoint file on successful completion
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print("✓ Removed checkpoint file (inference completed)")

# Save raw scores (pickle)
with open(os.path.join(CONFIG["output_dir"], "clip_scores.pkl"), 'wb') as f:
    pickle.dump(similarity_scores, f)
print(f"✓ Raw scores saved to {CONFIG['output_dir']}/clip_scores.pkl")

# Save human-readable scores (JSON)
scores_dict = {
    "prompts": all_prompts,
    "clip_scores": similarity_scores.tolist(),
    "num_samples_per_prompt": CONFIG["num_samples_per_prompt"],
    "statistics": {
        "clip_similarity": {
            "mean": float(similarity_scores.mean()),
            "std": float(similarity_scores.std()),
            "min": float(similarity_scores.min()),
            "max": float(similarity_scores.max()),
        },
        "per_prompt": {
            "mean_of_means": float(per_prompt_mean.mean()),
            "mean_of_maxes": float(per_prompt_max.mean()),
            "mean_of_mins": float(per_prompt_min.mean()),
        }
    }
}
with open(os.path.join(CONFIG["output_dir"], "clip_scores.json"), 'w') as f:
    json.dump(scores_dict, f, indent=2)
print(f"✓ Scores saved to {CONFIG['output_dir']}/clip_scores.json")

# Save summary
summary = f"""
INFERENCE SUMMARY
{'=' * 80}
Checkpoint: {CONFIG['checkpoint_path']}
Base Model: {CONFIG['base_model']}
Unique Prompts: {len(all_prompts) // CONFIG['num_samples_per_prompt']}
Samples per Prompt: {CONFIG['num_samples_per_prompt']}
Total Images Generated: {len(all_prompts)}

CLIP Similarity Scores (Overall):
  Mean:  {similarity_scores.mean():.4f}
  Std:   {similarity_scores.std():.4f}
  Min:   {similarity_scores.min():.4f}
  Max:   {similarity_scores.max():.4f}

CLIP Similarity (Per-Prompt Averages):
  Mean of means: {per_prompt_mean.mean():.4f}
  Mean of maxes: {per_prompt_max.mean():.4f}
  Mean of mins:  {per_prompt_min.mean():.4f}

Top 5 Images (by CLIP score):
"""
top_5_indices = similarity_scores.topk(5).indices
for rank, idx in enumerate(top_5_indices, 1):
    summary += f"  {rank}. Image {idx:05d}.png - '{all_prompts[idx]}' - Score: {similarity_scores[idx]:.4f}\n"

summary += f"\nBottom 5 Images (by CLIP score):\n"
bottom_5_indices = similarity_scores.topk(5, largest=False).indices
for rank, idx in enumerate(bottom_5_indices, 1):
    summary += f"  {rank}. Image {idx:05d}.png - '{all_prompts[idx]}' - Score: {similarity_scores[idx]:.4f}\n"

summary += f"\n{'=' * 80}\n"

with open(os.path.join(CONFIG["output_dir"], "summary.txt"), 'w') as f:
    f.write(summary)

print(summary)
print(f"✓ Summary saved to {CONFIG['output_dir']}/summary.txt")

print("\n" + "=" * 80)
print("✅ INFERENCE COMPLETE!")
print("=" * 80)
