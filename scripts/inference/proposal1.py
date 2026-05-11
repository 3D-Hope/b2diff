"""
Generative Reach Experiment Analysis for Proposition 1
Measures the impact of updates at different timesteps on final output
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import open_clip
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION - Add your methods here
# ============================================================================
METHODS = {
    "baseline": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/baseline_sd14_samples_10steps/pretrained_latents.pt",
    # Add your strategy outputs here:
    "uniform_10": "outputs/lora_inference_uniform_10/lora_latents.pt",
    "late_10": "outputs/lora_inference_last_10/lora_latents.pt",
    # "uniform_all_20": "outputs/lora_inference_uniform_all_20/lora_latents.pt",
}

CONFIG = {
    "base_model": "CompVis/stable-diffusion-v1-4",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "outputs/proposal1_analysis",
}

print("=" * 80)
print("GENERATIVE REACH EXPERIMENT ANALYSIS - PROPOSITION 1")
print("=" * 80)

device = CONFIG["device"]
print(f"\nUsing device: {device}")

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ============================================================================
# STEP 1: Load All Method Data
# ============================================================================
print("\n[STEP 1] Loading Method Data")
print(f"Analyzing {len(METHODS)} methods: {list(METHODS.keys())}")

data = {}
for method_name, pt_path in METHODS.items():
    print(f"\nLoading {method_name} from {pt_path}...")
    if not os.path.exists(pt_path):
        print(f"  ⚠ File not found, skipping...")
        continue
    
    method_data = torch.load(pt_path, map_location='cpu')
    data[method_name] = {
        'initial_noise': method_data['initial_noise'],
        'final_latents': method_data['final_latents'],
        'image_tensor': method_data['image_tensor'],
        'image_path': method_data['image_path'],
    }
    print(f"  ✓ Loaded: initial_noise {method_data['initial_noise'].shape}, "
          f"final_latents {method_data['final_latents'].shape}, "
          f"image_tensor {method_data['image_tensor'].shape}")

if len(data) == 0:
    print("❌ No valid data loaded. Please check your METHOD paths.")
    exit(1)

if "baseline" not in data:
    print("❌ Baseline data not found. Required for comparison.")
    exit(1)

baseline = data["baseline"]
num_samples = baseline['final_latents'].shape[0]
print(f"\n✓ Total samples: {num_samples}")

# ============================================================================
# STEP 2: Compute Perturbation Magnitudes
# ============================================================================
print("\n[STEP 2] Computing Perturbation Magnitudes")

results = {}
baseline_latents = baseline['final_latents'].to(device)
baseline_images = baseline['image_tensor'].to(device)

for method_name, method_data in data.items():
    if method_name == "baseline":
        continue
    
    print(f"\nComputing distances for {method_name} vs baseline...")
    
    method_latents = method_data['final_latents'].to(device)
    method_images = method_data['image_tensor'].to(device)
    
    # L2 distance between latents (latent space perturbation)
    latent_diff = method_latents - baseline_latents
    latent_l2 = torch.norm(latent_diff, p=2, dim=(1, 2, 3))  # Shape: (num_samples,)
    
    # L2 distance between images (pixel space perturbation)
    image_diff = method_images - baseline_images
    image_l2 = torch.norm(image_diff, p=2, dim=(1, 2, 3))  # Shape: (num_samples,)
    
    # Store results
    results[method_name] = {
        'latent_l2': latent_l2.cpu().numpy(),
        'image_l2': image_l2.cpu().numpy(),
    }
    
    print(f"  Latent L2: mean={latent_l2.mean():.4f}, std={latent_l2.std():.4f}")
    print(f"  Image L2: mean={image_l2.mean():.4f}, std={image_l2.std():.4f}")

# ============================================================================
# STEP 3: Compute CLIP Scores (Reward Movement)
# ============================================================================
print("\n[STEP 3] Computing CLIP Scores for Reward Movement")

print("Loading CLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-H-14', 
    pretrained='laion2B-s32B-b79K'
)
clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
clip_model = clip_model.to(device)
clip_model.eval()
print("✓ CLIP model loaded")

# Load prompts (from baseline directory)
baseline_dir = os.path.dirname(METHODS["baseline"])
prompts_file = os.path.join(baseline_dir, "prompts.json")
if os.path.exists(prompts_file):
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    prompts = prompts[:num_samples]
    print(f"✓ Loaded {len(prompts)} prompts")
else:
    print("⚠ Prompts file not found, using placeholder prompts")
    prompts = [""] * num_samples

# Compute CLIP scores for each method
clip_scores = {}
eval_batch_size = 32

for method_name, method_data in data.items():
    print(f"\nComputing CLIP scores for {method_name}...")
    
    method_scores = []
    num_eval_batches = (num_samples + eval_batch_size - 1) // eval_batch_size
    
    for batch_idx in tqdm(range(num_eval_batches), desc=f"  {method_name}"):
        start_idx = batch_idx * eval_batch_size
        end_idx = min(start_idx + eval_batch_size, num_samples)
        
        batch_prompts = prompts[start_idx:end_idx]
        batch_images = []
        
        for idx in range(start_idx, end_idx):
            img_path = method_data['image_path'][idx]
            if os.path.exists(img_path):
                img = Image.open(img_path)
                batch_images.append(clip_preprocess(img))
            else:
                # If image doesn't exist, use tensor directly
                img_tensor = method_data['image_tensor'][idx]
                img_array = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                batch_images.append(clip_preprocess(img))
        
        if len(batch_images) == 0:
            continue
        
        image_input = torch.stack(batch_images).to(device)
        text_input = clip_tokenizer(batch_prompts).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            batch_size_actual = len(image_features)
            similarity = (image_features @ text_features.T)[
                torch.arange(batch_size_actual), torch.arange(batch_size_actual)
            ]
            method_scores.extend(similarity.cpu().tolist())
    
    clip_scores[method_name] = np.array(method_scores)
    print(f"  CLIP score: mean={clip_scores[method_name].mean():.4f}, std={clip_scores[method_name].std():.4f}")

# Compute reward movement (ΔR) relative to baseline
baseline_clip = clip_scores["baseline"]
for method_name in results.keys():
    if method_name in clip_scores:
        delta_r = clip_scores[method_name] - baseline_clip
        results[method_name]['delta_r'] = delta_r
        print(f"  {method_name} ΔR: mean={delta_r.mean():.4f}, std={delta_r.std():.4f}")

# ============================================================================
# STEP 4: Display Results Table
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS TABLE")
print("=" * 80)

table_data = []
for method_name in sorted(results.keys()):
    r = results[method_name]
    row = [
        method_name,
        f"{r['latent_l2'].mean():.4f} ± {r['latent_l2'].std():.4f}",
        f"{r['image_l2'].mean():.4f} ± {r['image_l2'].std():.4f}",
    ]
    if 'delta_r' in r:
        row.append(f"{r['delta_r'].mean():.4f} ± {r['delta_r'].std():.4f}")
    else:
        row.append("N/A")
    table_data.append(row)

headers = ["Method", "Latent L2 (||δx₀||)", "Image L2 (||δx₀||)", "Reward ΔR"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# ============================================================================
# STEP 5: Visualizations
# ============================================================================
print("\n[STEP 5] Creating Visualizations")

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Generative Reach Experiment - Proposition 1', fontsize=16, fontweight='bold')

# Plot 1: Latent L2 distances
ax1 = axes[0, 0]
for method_name in sorted(results.keys()):
    r = results[method_name]
    ax1.hist(r['latent_l2'], bins=30, alpha=0.7, label=method_name)
ax1.set_xlabel('Latent L2 Distance (||δx₀||)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Latent Space Perturbation Magnitude', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Image L2 distances
ax2 = axes[0, 1]
for method_name in sorted(results.keys()):
    r = results[method_name]
    ax2.hist(r['image_l2'], bins=30, alpha=0.7, label=method_name)
ax2.set_xlabel('Image L2 Distance (||δx₀||)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Pixel Space Perturbation Magnitude', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Reward movement (if available)
ax3 = axes[1, 0]
delta_r_available = any('delta_r' in r for r in results.values())
if delta_r_available:
    for method_name in sorted(results.keys()):
        if 'delta_r' in results[method_name]:
            r = results[method_name]
            ax3.hist(r['delta_r'], bins=30, alpha=0.7, label=method_name)
    ax3.set_xlabel('Reward Movement (ΔR)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('CLIP Score Change After Update', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'CLIP scores not computed', ha='center', va='center', 
             transform=ax3.transAxes, fontsize=14)
    ax3.set_title('Reward Movement', fontsize=14, fontweight='bold')

# Plot 4: Bar chart comparison
ax4 = axes[1, 1]
methods_sorted = sorted(results.keys())
latent_means = [results[m]['latent_l2'].mean() for m in methods_sorted]
image_means = [results[m]['image_l2'].mean() for m in methods_sorted]

x = np.arange(len(methods_sorted))
width = 0.35

bars1 = ax4.bar(x - width/2, latent_means, width, label='Latent L2', alpha=0.8)
bars2 = ax4.bar(x + width/2, image_means, width, label='Image L2', alpha=0.8)

ax4.set_xlabel('Method', fontsize=12)
ax4.set_ylabel('Mean L2 Distance', fontsize=12)
ax4.set_title('Mean Perturbation Magnitude Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(methods_sorted, rotation=45, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "proposal1_analysis.png"), dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to {CONFIG['output_dir']}/proposal1_analysis.png")

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n[STEP 6] Saving Results")

# Save numerical results
results_export = {
    "baseline_clip_score": float(baseline_clip.mean()),
    "methods": {}
}

for method_name in results.keys():
    r = results[method_name]
    results_export["methods"][method_name] = {
        "latent_l2_mean": float(r['latent_l2'].mean()),
        "latent_l2_std": float(r['latent_l2'].std()),
        "image_l2_mean": float(r['image_l2'].mean()),
        "image_l2_std": float(r['image_l2'].std()),
    }
    if 'delta_r' in r:
        results_export["methods"][method_name]["delta_r_mean"] = float(r['delta_r'].mean())
        results_export["methods"][method_name]["delta_r_std"] = float(r['delta_r'].std())

with open(os.path.join(CONFIG["output_dir"], "proposal1_results.json"), 'w') as f:
    json.dump(results_export, f, indent=2)
print(f"✓ Results saved to {CONFIG['output_dir']}/proposal1_results.json")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nBaseline CLIP Score: {baseline_clip.mean():.4f}")
print(f"\nMethod Comparison:")
for method_name in sorted(results.keys()):
    r = results[method_name]
    print(f"\n  {method_name}:")
    print(f"    Latent L2:  {r['latent_l2'].mean():.4f} ± {r['latent_l2'].std():.4f}")
    print(f"    Image L2:   {r['image_l2'].mean():.4f} ± {r['image_l2'].std():.4f}")
    if 'delta_r' in r:
        print(f"    Reward ΔR:  {r['delta_r'].mean():.4f} ± {r['delta_r'].std():.4f}")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
