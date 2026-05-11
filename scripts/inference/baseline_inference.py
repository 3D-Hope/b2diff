"""
Generate images from baseline SD v1-4 and compute CLIP similarity scores
"""
import os
import json
import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import open_clip

# Configuration
CONFIG = {
    "prompt_file": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/configs/prompt/template1_train.json",
    "output_dir": "outputs/baseline_sd14_samples_10steps",
    "base_model": "CompVis/stable-diffusion-v1-4",
    "batch_size": 8,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "seed": 42,
    "num_samples": 45,  # Total number of images to generate
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print("=" * 80)
print("BASELINE SD v1-4 INFERENCE + CLIP SCORING")
print("=" * 80)

device = CONFIG["device"]
print(f"\nUsing device: {device}")

# Create output directory
os.makedirs(os.path.join(CONFIG["output_dir"], "images"), exist_ok=True)

# Load prompts
print(f"\nLoading prompts from {CONFIG['prompt_file']}...")
with open(CONFIG["prompt_file"], 'r') as f:
    prompts = json.load(f)
print(f"✓ Loaded {len(prompts)} unique prompts")

# ============================================================================
# PHASE 1: Generate Images
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1: IMAGE GENERATION")
print("=" * 80)

print("\nLoading Stable Diffusion v1-4 baseline model...")
pipeline = StableDiffusionPipeline.from_pretrained(
    CONFIG["base_model"],
    torch_dtype=torch.float16,
    safety_checker=None
)
pipeline = pipeline.to(device)
pipeline.set_progress_bar_config(disable=True)
print("✓ Model loaded successfully!")

# Generate images
print(f"\nGenerating {CONFIG['num_samples']} images...")
num_batches = (CONFIG['num_samples'] + CONFIG['batch_size'] - 1) // CONFIG['batch_size']
all_prompts = []
all_initial_noise = []  # Store initial noise for each sample
all_final_latents = []  # Store final x0 latents for each sample
all_image_tensors = []  # Store final image tensors for each sample
all_seeds = []  # Store seed used for each sample

# Set all random seeds for reproducibility
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])

for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
    current_batch_size = min(CONFIG['batch_size'], CONFIG['num_samples'] - batch_idx * CONFIG['batch_size'])
    
    # Cycle through prompts
    batch_prompts = [prompts[(batch_idx * CONFIG['batch_size'] + i) % len(prompts)] 
                    for i in range(current_batch_size)]
    
    # Generate images with explicit generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(CONFIG["seed"] + batch_idx)
    
    with torch.no_grad():
        # Use custom generation to capture latents
        # Encode prompts using text encoder directly
        text_input = pipeline.tokenizer(
            batch_prompts,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device))[0]
        
        # Encode unconditional prompts for classifier-free guidance
        uncond_input = pipeline.tokenizer(
            [""] * len(batch_prompts),
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(device))[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare scheduler
        pipeline.scheduler.set_timesteps(CONFIG['num_inference_steps'], device=device)
        timesteps = pipeline.scheduler.timesteps
        
        # Generate initial noise
        batch_size = len(batch_prompts)
        noise_shape = (batch_size, 4, 64, 64)
        noise = torch.randn(noise_shape, generator=generator, device=device, dtype=torch.float16)
        
        # Store initial noise for each sample in the batch
        for i in range(batch_size):
            all_initial_noise.append(noise[i:i+1].cpu().clone())
        
        # Denoising loop
        latents = noise
        for t in timesteps:
            # Expand latents if doing classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + CONFIG['guidance_scale'] * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Store final latents (x0)
        for i in range(batch_size):
            all_final_latents.append(latents[i:i+1].cpu().clone())
        
        # Decode latents to images and keep as tensors
        latents_for_decode = latents.clone()
        latents_for_decode = 1 / 0.18215 * latents_for_decode
        decoded_images = pipeline.vae.decode(latents_for_decode).sample
        decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
        
        # Store image tensors before converting to PIL
        for i in range(batch_size):
            all_image_tensors.append(decoded_images[i:i+1].cpu().clone())
        
        # Convert to PIL for saving
        images = decoded_images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        images = [Image.fromarray(img) for img in images]
    
    # Save images and metadata
    for i, img in enumerate(images):
        img_idx = batch_idx * CONFIG['batch_size'] + i
        img_path = os.path.join(CONFIG["output_dir"], "images", f"{img_idx:05d}.png")
        img.save(img_path)
        all_prompts.append(batch_prompts[i])
        all_seeds.append(CONFIG["seed"] + batch_idx)

print(f"✓ Generated {len(all_prompts)} images")

# Save prompts
with open(os.path.join(CONFIG["output_dir"], "prompts.json"), 'w') as f:
    json.dump(all_prompts, f, indent=2)

# Save only the required fields for Generative Reach experiment
print("\nSaving initial noise, final latents, image tensors, and image paths...")

# Stack all tensors
initial_noise_tensor = torch.cat(all_initial_noise, dim=0)  # Shape: (num_samples, 4, 64, 64)
final_latents_tensor = torch.cat(all_final_latents, dim=0)  # Shape: (num_samples, 4, 64, 64)
image_tensors = torch.cat(all_image_tensors, dim=0)  # Shape: (num_samples, 3, 512, 512)
image_paths = [os.path.join(CONFIG["output_dir"], "images", f"{i:05d}.png") for i in range(len(all_prompts))]

# Save only the 4 required fields
torch.save({
    'initial_noise': initial_noise_tensor,
    'final_latents': final_latents_tensor,
    'image_tensor': image_tensors,
    'image_path': image_paths,
}, os.path.join(CONFIG["output_dir"], "pretrained_latents.pt"))

print(f"✓ Saved initial_noise: {initial_noise_tensor.shape}")
print(f"✓ Saved final_latents: {final_latents_tensor.shape}")
print(f"✓ Saved image_tensor: {image_tensors.shape}")
print(f"✓ Saved image_path: {len(image_paths)} paths")
print(f"✓ All data saved to pretrained_latents.pt")

# Clean up generation model
del pipeline
torch.cuda.empty_cache()
print("✓ Generation model unloaded")

# ============================================================================
# PHASE 2: Compute CLIP Similarity Scores
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2: CLIP SIMILARITY SCORING")
print("=" * 80)

print("\nLoading CLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-H-14', 
    pretrained='laion2B-s32B-b79K'
)
clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
clip_model = clip_model.to(device)
clip_model.eval()
print("✓ CLIP model loaded: ViT-H-14")

print(f"\nComputing CLIP similarity scores for {len(all_prompts)} images...")
similarity_scores = []

eval_batch_size = 32
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
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity (diagonal = matching pairs)
        batch_size_actual = len(image_features)
        similarity = (image_features @ text_features.T)[
            torch.arange(batch_size_actual), torch.arange(batch_size_actual)
        ]
        similarity_scores.extend(similarity.cpu().tolist())

similarity_scores = torch.tensor(similarity_scores)

# ============================================================================
# Results
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nBaseline SD v1-4 CLIP Similarity Scores:")
print(f"  Number of images: {len(similarity_scores)}")
print(f"  Mean reward:      {similarity_scores.mean():.4f}")
print(f"  Std:              {similarity_scores.std():.4f}")
print(f"  Min:              {similarity_scores.min():.4f}")
print(f"  Max:              {similarity_scores.max():.4f}")

# Save results
results = {
    "model": CONFIG["base_model"],
    "num_images": len(similarity_scores),
    "clip_scores": similarity_scores.tolist(),
    "statistics": {
        "mean": float(similarity_scores.mean()),
        "std": float(similarity_scores.std()),
        "min": float(similarity_scores.min()),
        "max": float(similarity_scores.max()),
    }
}

with open(os.path.join(CONFIG["output_dir"], "baseline_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {CONFIG['output_dir']}/baseline_results.json")
print("\n" + "=" * 80)
print("✅ BASELINE INFERENCE COMPLETE!")
print("=" * 80)
