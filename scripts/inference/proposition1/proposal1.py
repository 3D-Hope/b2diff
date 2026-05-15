"""
Generative Reach Experiment Analysis for Proposition 1
Measures the impact of updates at different timesteps on final output
"""
import os
import json
import csv
import re
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
BASELINE_PATH = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/baseline_sd14_samples_10steps/pretrained_latents.pt"
STAGED_OUTPUT_ROOT = "outputs/proposal1_batch"


def discover_method_paths():
    methods = {"baseline": BASELINE_PATH}
    discovered = {}

    if os.path.exists(STAGED_OUTPUT_ROOT):
        for current_root, _, files in os.walk(STAGED_OUTPUT_ROOT):
            if "lora_latents.pt" not in files:
                continue

            pt_path = os.path.join(current_root, "lora_latents.pt")
            relative_path = os.path.relpath(current_root, STAGED_OUTPUT_ROOT)
            match = re.search(r"(late_10|uniform_10|uniform_all_20).*stage(\d+)", relative_path)
            if match:
                method_name = f"{match.group(1)}_stage{match.group(2)}"
                discovered[method_name] = pt_path

    if discovered:
        methods.update(dict(sorted(discovered.items())))
    else:
        methods.update(
            {
                "uniform_10": "outputs/lora_inference_uniform_10/lora_latents.pt",
                "late_10": "outputs/lora_inference_last_10/lora_latents.pt",
            }
        )

    return methods


METHODS = discover_method_paths()

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


def method_group(method_name):
    if "_stage" in method_name:
        return method_name.split("_stage", 1)[0]
    return method_name


def method_stage(method_name):
    if "_stage" in method_name:
        return int(method_name.split("_stage", 1)[1])
    return None


def summarize(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def paired_effect_size(a, b):
    diff = a - b
    std = diff.std()
    return float(diff.mean()), float(std), float(diff.mean() / std) if std > 0 else float("inf")

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

non_baseline_methods = [name for name in data.keys() if name != "baseline"]

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


summary_rows = []
for method_name in sorted(results.keys()):
    r = results[method_name]
    summary_rows.append(
        {
            "method": method_name,
            "run_group": method_group(method_name),
            "stage": method_stage(method_name),
            "latent_l2_mean": float(r['latent_l2'].mean()),
            "latent_l2_std": float(r['latent_l2'].std()),
            "image_l2_mean": float(r['image_l2'].mean()),
            "image_l2_std": float(r['image_l2'].std()),
            "delta_r_mean": float(r['delta_r'].mean()) if 'delta_r' in r else None,
            "delta_r_std": float(r['delta_r'].std()) if 'delta_r' in r else None,
        }
    )

summary_csv_path = os.path.join(CONFIG["output_dir"], "proposal1_summary.csv")
with open(summary_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"✓ Summary CSV saved to {summary_csv_path}")

sample_rows = []
for method_name, r in results.items():
    for idx in range(len(r['latent_l2'])):
        sample_rows.append(
            {
                "method": method_name,
                "run_group": method_group(method_name),
                "stage": method_stage(method_name),
                "sample_idx": idx,
                "latent_l2": float(r['latent_l2'][idx]),
                "image_l2": float(r['image_l2'][idx]),
                "delta_r": float(r['delta_r'][idx]) if 'delta_r' in r else None,
            }
        )

samples_csv_path = os.path.join(CONFIG["output_dir"], "proposal1_samples.csv")
with open(samples_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
    writer.writeheader()
    writer.writerows(sample_rows)
print(f"✓ Per-sample CSV saved to {samples_csv_path}")

pairwise_rows = []
groups = sorted({method_group(name) for name in results.keys()})
stages = sorted({method_stage(name) for name in results.keys() if method_stage(name) is not None})
for stage in stages:
    stage_methods = {method_group(name): name for name in results.keys() if method_stage(name) == stage}
    if len(stage_methods) < 2:
        continue
    if "uniform_10" in stage_methods and "late_10" in stage_methods:
        uniform_name = stage_methods["uniform_10"]
        late_name = stage_methods["late_10"]
        uniform_latent = results[uniform_name]['latent_l2']
        late_latent = results[late_name]['latent_l2']
        uniform_delta = results[uniform_name].get('delta_r')
        late_delta = results[late_name].get('delta_r')

        latent_mean_diff, latent_std_diff, latent_d = paired_effect_size(uniform_latent, late_latent)
        pairwise_rows.append(
            {
                "stage": stage,
                "metric": "latent_l2",
                "compare": "uniform_10 - late_10",
                "mean_diff": latent_mean_diff,
                "std_diff": latent_std_diff,
                "cohens_d": latent_d,
            }
        )

        if uniform_delta is not None and late_delta is not None:
            reward_mean_diff, reward_std_diff, reward_d = paired_effect_size(uniform_delta, late_delta)
            pairwise_rows.append(
                {
                    "stage": stage,
                    "metric": "reward_delta_r",
                    "compare": "uniform_10 - late_10",
                    "mean_diff": reward_mean_diff,
                    "std_diff": reward_std_diff,
                    "cohens_d": reward_d,
                }
            )

pairwise_csv_path = os.path.join(CONFIG["output_dir"], "proposal1_pairwise_comparisons.csv")
if pairwise_rows:
    with open(pairwise_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pairwise_rows)
    print(f"✓ Pairwise CSV saved to {pairwise_csv_path}")

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

if pairwise_rows:
    print("\n" + "=" * 80)
    print("PAIRWISE RUN COMPARISON")
    print("=" * 80)
    pairwise_table = []
    for row in pairwise_rows:
        pairwise_table.append(
            [
                f"stage{row['stage']}",
                row["metric"],
                f"{row['mean_diff']:.4f}",
                f"{row['std_diff']:.4f}",
                f"{row['cohens_d']:.4f}",
            ]
        )
    print(tabulate(pairwise_table, headers=["Stage", "Metric", "Mean Diff", "Std Diff", "Cohen's d"], tablefmt="grid"))

# ============================================================================
# STEP 5: Visualizations
# ============================================================================
print("\n[STEP 5] Creating Visualizations")

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Generative Reach Experiment - Proposition 1', fontsize=16, fontweight='bold')

# Plot 1: Latent L2 distances by run group
ax1 = axes[0, 0]
for run_name in sorted({method_group(name) for name in results.keys()}):
    group_latents = np.concatenate([results[name]['latent_l2'] for name in results if method_group(name) == run_name])
    ax1.hist(group_latents, bins=30, alpha=0.6, label=run_name)
ax1.set_xlabel('Latent L2 Distance (||δx₀||)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Latent Space Perturbation Magnitude by Run', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Reward movement by run group
ax2 = axes[0, 1]
for run_name in sorted({method_group(name) for name in results.keys()}):
    group_delta = np.concatenate([results[name]['delta_r'] for name in results if method_group(name) == run_name and 'delta_r' in results[name]]) if any('delta_r' in results[name] and method_group(name) == run_name for name in results) else None
    if group_delta is not None and len(group_delta) > 0:
        ax2.hist(group_delta, bins=30, alpha=0.6, label=run_name)
ax2.set_xlabel('Reward Movement (ΔR)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Reward Movement by Run', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Mean latent L2 vs stage
ax3 = axes[1, 0]
for run_name in sorted({method_group(name) for name in results.keys()}):
    stage_points = sorted(
        (method_stage(name), results[name]['latent_l2'].mean())
        for name in results.keys()
        if method_group(name) == run_name and method_stage(name) is not None
    )
    if stage_points:
        stages_x, values_y = zip(*stage_points)
        ax3.plot(stages_x, values_y, marker='o', linewidth=2, label=run_name)
ax3.set_xlabel('Stage', fontsize=12)
ax3.set_ylabel('Mean Latent L2', fontsize=12)
ax3.set_title('Stage-wise Latent Perturbation', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Mean reward delta vs stage
ax4 = axes[1, 1]
for run_name in sorted({method_group(name) for name in results.keys()}):
    stage_points = sorted(
        (method_stage(name), results[name]['delta_r'].mean())
        for name in results.keys()
        if method_group(name) == run_name and 'delta_r' in results[name] and method_stage(name) is not None
    )
    if stage_points:
        stages_x, values_y = zip(*stage_points)
        ax4.plot(stages_x, values_y, marker='o', linewidth=2, label=run_name)
ax4.axhline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.7)
ax4.set_xlabel('Stage', fontsize=12)
ax4.set_ylabel('Mean Reward ΔR', fontsize=12)
ax4.set_title('Stage-wise Reward Movement', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "proposal1_analysis.png"), dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to {CONFIG['output_dir']}/proposal1_analysis.png")

if pairwise_rows:
    plt.figure(figsize=(10, 6))
    heatmap_data = []
    heatmap_labels = []
    for stage in stages:
        row = [item for item in pairwise_rows if item["stage"] == stage and item["metric"] == "latent_l2"]
        if row:
            heatmap_data.append([row[0]["mean_diff"], row[0]["cohens_d"]])
            heatmap_labels.append(f"stage{stage}")
    if heatmap_data:
        sns.heatmap(
            np.array(heatmap_data),
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            yticklabels=heatmap_labels,
            xticklabels=["mean diff", "Cohen's d"],
        )
        plt.title("Uniform_10 vs Late_10: Latent Difference by Stage")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], "proposal1_pairwise_heatmap.png"), dpi=300, bbox_inches='tight')
        print(f"✓ Pairwise heatmap saved to {CONFIG['output_dir']}/proposal1_pairwise_heatmap.png")

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

if pairwise_rows:
    print("\nPairwise interpretation:")
    for row in pairwise_rows:
        if row["metric"] == "latent_l2":
            direction = "higher" if row["mean_diff"] > 0 else "lower"
            print(
                f"  stage{row['stage']}: uniform_10 is {direction} than late_10 by "
                f"{abs(row['mean_diff']):.4f} latent-L2 units (Cohen's d={row['cohens_d']:.4f})"
            )

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
