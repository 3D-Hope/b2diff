"""
Analyze perturbation geometry: compare how uniform vs late strategies 
spread changes across latent dimensions.

Computes:
- Per-dimension variance (active dimensions)
- Effective dimensionality (ED)
- Anisotropy ratio
- Per-channel contribution
- PCA cumulative variance explained
- Participation ratio

Outputs: CSVs + publication-quality plots
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.insert(0, project_root)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "stage": 0,  # Stage number to analyze
    "methods": {
        "baseline": "outputs/baseline_sd14_samples_1080_10steps/pretrained_latents.pt",
        "uniform_10": "outputs/proposition1_stage0_1080/uniform_10/lora_latents.pt",
        "last_10": "outputs/proposition1_stage0_1080/last_10/lora_latents.pt",
    },
    "output_dir": "outputs/proposition1_geometry_analysis",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,  # Use float32 for analysis stability
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

print("=" * 80)
print("PERTURBATION GEOMETRY ANALYSIS")
print("=" * 80)
print(f"Output dir: {CONFIG['output_dir']}")
print(f"Device: {CONFIG['device']}")
print()

# ============================================================================
# UTILITIES
# ============================================================================

def load_latents(path):
    """Load final_latents from .pt file"""
    data = torch.load(path, map_location=CONFIG["device"])
    return data['final_latents'].to(CONFIG["dtype"])


def compute_perturbation(baseline_latents, lora_latents):
    """Compute delta_x0 = lora - baseline"""
    return (lora_latents - baseline_latents).to(CONFIG["dtype"])


def per_dimension_variance(delta_x0):
    """
    Compute per-dimension variance across samples.
    delta_x0: (num_samples, 4, 64, 64)
    Returns: (16384,) tensor of per-dimension stds
    """
    N = delta_x0.shape[0]
    delta_flat = delta_x0.reshape(N, -1)  # (N, 16384)
    dim_stds = delta_flat.std(dim=0)  # (16384,)
    return dim_stds


def effective_dimensionality(delta_x0):
    """
    Compute effective dimensionality (ED) using singular values.
    ED = (sum sigma_i)^2 / sum sigma_i^2
    ED = d means all dimensions equally active
    ED << d means concentrated in few dims
    """
    N = delta_x0.shape[0]
    delta_flat = delta_x0.reshape(N, -1)
    
    try:
        U, S, Vt = torch.linalg.svd(delta_flat, full_matrices=False)
    except:
        # Fallback for numerical issues
        C = delta_flat.T @ delta_flat / N
        S = torch.linalg.eigvalsh(C)
        S = torch.sqrt(S.clamp(min=0))
    
    S_norm = S / S.sum()
    ED = (S_norm.sum() ** 2) / ((S_norm ** 2).sum())
    return ED.item()


def anisotropy_ratio(delta_x0):
    """
    Compute anisotropy ratio = max_eigenval / mean_eigenval
    Close to 1 = isotropic (uniform)
    >> 1 = anisotropic (concentrated)
    """
    N = delta_x0.shape[0]
    delta_flat = delta_x0.reshape(N, -1)
    C = (delta_flat.T @ delta_flat) / N
    eigenvals = torch.linalg.eigvalsh(C)
    eigenvals = eigenvals[eigenvals > 0]  # Remove numerical noise
    
    if len(eigenvals) == 0:
        return 1.0
    
    ratio = eigenvals[-1] / (eigenvals.mean() + 1e-8)
    return ratio.item()


def per_channel_analysis(delta_x0):
    """
    Compute per-channel contribution to total perturbation.
    delta_x0: (N, 4, 64, 64)
    Returns: dict with per-channel metrics
    """
    N = delta_x0.shape[0]
    channel_norms = []
    channel_stds = []
    
    for ch in range(4):
        ch_data = delta_x0[:, ch, :, :]  # (N, 64, 64)
        ch_norms = ch_data.reshape(N, -1).norm(dim=1)  # (N,)
        channel_norms.append(ch_norms.mean().item())
        channel_stds.append(ch_norms.std().item())
    
    total_norm = sum(channel_norms)
    channel_fracs = [n / (total_norm + 1e-8) for n in channel_norms]
    
    return {
        "channel_means": channel_norms,
        "channel_stds": channel_stds,
        "channel_fracs": channel_fracs,
    }


def pca_analysis(delta_x0, num_components=None):
    """
    Run PCA, return cumulative variance explained.
    Returns: dict with variance metrics
    """
    N = delta_x0.shape[0]
    delta_flat = delta_x0.reshape(N, -1)
    
    U, S, Vt = torch.linalg.svd(delta_flat, full_matrices=False)
    var_explained = (S ** 2) / (S ** 2).sum()
    cumsum_var = torch.cumsum(var_explained, dim=0)
    
    # Find thresholds
    thresholds = [0.5, 0.9, 0.95, 0.99]
    num_comps = {}
    for thresh in thresholds:
        idx = (cumsum_var >= thresh).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            num_comps[f"num_pc_{int(thresh*100)}"] = idx[0].item()
        else:
            num_comps[f"num_pc_{int(thresh*100)}"] = len(S)
    
    return {
        "var_explained": var_explained.cpu().numpy(),
        "cumsum_var": cumsum_var.cpu().numpy(),
        "singular_values": S.cpu().numpy(),
        **num_comps,
    }


def participation_ratio(delta_x0):
    """
    PR = (sum sigma_i)^2 / sum sigma_i^4
    Simpler alternative to ED
    """
    N = delta_x0.shape[0]
    delta_flat = delta_x0.reshape(N, -1)
    
    U, S, Vt = torch.linalg.svd(delta_flat, full_matrices=False)
    S_norm = S / S.sum()
    
    PR = (S_norm.sum() ** 2) / ((S_norm ** 4).sum() + 1e-8)
    return PR.item()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("[STEP 1] Load Latents")
print()

baseline_latents = load_latents(CONFIG["methods"]["baseline"])
print(f"✓ Baseline loaded: {baseline_latents.shape}")

results = {}

for method_name in ["uniform_10", "last_10"]:
    method_path = CONFIG["methods"][method_name]
    lora_latents = load_latents(method_path)
    print(f"✓ {method_name} loaded: {lora_latents.shape}")
    
    delta_x0 = compute_perturbation(baseline_latents, lora_latents)
    
    # Compute all metrics
    print(f"\n[STEP 2.{method_name}] Computing Metrics")
    
    # 1. Per-dimension variance
    dim_stds = per_dimension_variance(delta_x0)
    dim_stds_np = dim_stds.cpu().numpy()
    sorted_dim_stds = np.sort(dim_stds_np)[::-1]  # Descending
    
    # 2. Effective dimensionality
    ed = effective_dimensionality(delta_x0)
    
    # 3. Anisotropy ratio
    aniso = anisotropy_ratio(delta_x0)
    
    # 4. Per-channel
    ch_analysis = per_channel_analysis(delta_x0)
    
    # 5. PCA
    pca_res = pca_analysis(delta_x0)
    
    # 6. Participation ratio
    pr = participation_ratio(delta_x0)
    
    # Count active dimensions (std > median)
    median_std = np.median(dim_stds_np)
    active_dims = np.sum(dim_stds_np > median_std)
    active_dims_frac = active_dims / len(dim_stds_np)
    
    results[method_name] = {
        "delta_x0": delta_x0,
        "dim_stds": dim_stds_np,
        "sorted_dim_stds": sorted_dim_stds,
        "median_std": median_std,
        "active_dims": active_dims,
        "active_dims_frac": active_dims_frac,
        "ED": ed,
        "anisotropy_ratio": aniso,
        "participation_ratio": pr,
        "per_channel": ch_analysis,
        "pca": pca_res,
    }
    
    print(f"  ✓ Effective Dimensionality: {ed:.2f} / 16384")
    print(f"  ✓ Anisotropy Ratio: {aniso:.4f}")
    print(f"  ✓ Participation Ratio: {pr:.2f}")
    print(f"  ✓ Active Dims (> median): {active_dims} ({active_dims_frac*100:.1f}%)")
    print(f"  ✓ Per-channel means: {[f'{x:.4f}' for x in ch_analysis['channel_means']]}")
    print(f"  ✓ PCs for 90% var: {pca_res['num_pc_90']}")

print()

# ============================================================================
# SAVE SUMMARY TO CSV
# ============================================================================
print("[STEP 3] Save Summary Metrics")

summary_data = {
    "method": [],
    "effective_dimensionality": [],
    "anisotropy_ratio": [],
    "participation_ratio": [],
    "active_dims": [],
    "active_dims_frac": [],
    "median_std": [],
    "num_pc_50": [],
    "num_pc_90": [],
    "num_pc_95": [],
    "num_pc_99": [],
    "ch0_mean": [],
    "ch1_mean": [],
    "ch2_mean": [],
    "ch3_mean": [],
    "ch0_frac": [],
    "ch1_frac": [],
    "ch2_frac": [],
    "ch3_frac": [],
}

for method_name, res in results.items():
    summary_data["method"].append(method_name)
    summary_data["effective_dimensionality"].append(res["ED"])
    summary_data["anisotropy_ratio"].append(res["anisotropy_ratio"])
    summary_data["participation_ratio"].append(res["participation_ratio"])
    summary_data["active_dims"].append(res["active_dims"])
    summary_data["active_dims_frac"].append(res["active_dims_frac"])
    summary_data["median_std"].append(res["median_std"])
    summary_data["num_pc_50"].append(res["pca"]["num_pc_50"])
    summary_data["num_pc_90"].append(res["pca"]["num_pc_90"])
    summary_data["num_pc_95"].append(res["pca"]["num_pc_95"])
    summary_data["num_pc_99"].append(res["pca"]["num_pc_99"])
    
    for ch in range(4):
        summary_data[f"ch{ch}_mean"].append(res["per_channel"]["channel_means"][ch])
        summary_data[f"ch{ch}_frac"].append(res["per_channel"]["channel_fracs"][ch])

summary_df = pd.DataFrame(summary_data)
summary_csv = os.path.join(CONFIG["output_dir"], "geometry_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"✓ Summary saved to {summary_csv}")
print()
print(summary_df.to_string(index=False))
print()

# ============================================================================
# SAVE PCA VARIANCE DETAILS
# ============================================================================
print("[STEP 4] Save PCA Variance Details")

pca_data = {
    "method": [],
    "component": [],
    "variance_explained": [],
    "cumsum_variance": [],
}

for method_name, res in results.items():
    for i, (var, cumsum) in enumerate(
        zip(res["pca"]["var_explained"], res["pca"]["cumsum_var"])
    ):
        pca_data["method"].append(method_name)
        pca_data["component"].append(i)
        pca_data["variance_explained"].append(var)
        pca_data["cumsum_variance"].append(cumsum)

pca_df = pd.DataFrame(pca_data)
pca_csv = os.path.join(CONFIG["output_dir"], "pca_variance.csv")
pca_df.to_csv(pca_csv, index=False)
print(f"✓ PCA details saved to {pca_csv}")
print()

# ============================================================================
# SAVE PER-DIMENSION VARIANCE
# ============================================================================
print("[STEP 5] Save Per-Dimension Variance")

# Save sorted dimension stds for both methods
for method_name, res in results.items():
    dims_csv = os.path.join(
        CONFIG["output_dir"], f"dimension_variance_{method_name}.csv"
    )
    dim_df = pd.DataFrame(
        {
            "dimension_rank": np.arange(len(res["sorted_dim_stds"])),
            "std": res["sorted_dim_stds"],
            "cumsum_std": np.cumsum(res["sorted_dim_stds"]),
        }
    )
    dim_df.to_csv(dims_csv, index=False)
    print(f"✓ Dimension variance ({method_name}): {dims_csv}")

print()

# ============================================================================
# CREATE PLOTS
# ============================================================================
print("[STEP 6] Generate Plots")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Per-Dimension Variance (Top Dimensions)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for method_name, res in results.items():
    sorted_stds = res["sorted_dim_stds"][:500]  # Top 500 dims
    ax.plot(
        range(len(sorted_stds)),
        sorted_stds,
        marker="o",
        markersize=2,
        linewidth=2,
        label=method_name,
    )

ax.set_xlabel("Dimension Rank (sorted by variance)", fontsize=12)
ax.set_ylabel("Standard Deviation", fontsize=12)
ax.set_title("Per-Dimension Variance: Top 500 Dimensions", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "01_dimension_variance.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Cumulative Variance Explained (PCA)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for method_name, res in results.items():
    cumsum_var = res["pca"]["cumsum_var"][:500]  # First 500 components
    ax.plot(
        range(len(cumsum_var)),
        cumsum_var,
        marker="o",
        markersize=3,
        linewidth=2.5,
        label=method_name,
    )

# Add horizontal lines for thresholds
for thresh in [0.5, 0.9, 0.95]:
    ax.axhline(thresh, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(10, thresh + 0.01, f"{int(thresh*100)}%", fontsize=10, color="gray")

ax.set_xlabel("Principal Component", fontsize=12)
ax.set_ylabel("Cumulative Variance Explained", fontsize=12)
ax.set_title("Cumulative Variance Explained (PCA): First 500 Components", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.02])
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "02_cumulative_variance_pca.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Per-Channel Contribution
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

methods = list(results.keys())
x = np.arange(len(methods))
width = 0.35

# Subplot 1: Absolute channel means
channel_means_data = [
    [results[m]["per_channel"]["channel_means"][ch] for m in methods] for ch in range(4)
]
for ch in range(4):
    ax1.bar(x + ch * width / 4, channel_means_data[ch], width / 4, label=f"Channel {ch}")

ax1.set_xlabel("Method", fontsize=12)
ax1.set_ylabel("Mean Perturbation Norm", fontsize=12)
ax1.set_title("Per-Channel Contribution (Absolute)", fontsize=13, fontweight="bold")
ax1.set_xticks(x + width / 2)
ax1.set_xticklabels(methods)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis="y")

# Subplot 2: Channel fractions (stacked bar)
channel_fracs_data = np.array(
    [
        [results[m]["per_channel"]["channel_fracs"][ch] for ch in range(4)]
        for m in methods
    ]
)
bottom = np.zeros(len(methods))
for ch in range(4):
    ax2.bar(
        methods,
        channel_fracs_data[:, ch],
        bottom=bottom,
        label=f"Channel {ch}",
        alpha=0.8,
    )
    bottom += channel_fracs_data[:, ch]

ax2.set_ylabel("Fraction of Total Perturbation", fontsize=12)
ax2.set_title("Per-Channel Contribution (Fractions)", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.set_ylim([0, 1.0])
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "03_per_channel_analysis.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Summary Metrics Comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

metrics = [
    ("effective_dimensionality", "Effective Dimensionality", axes[0, 0]),
    ("anisotropy_ratio", "Anisotropy Ratio (log scale)", axes[0, 1]),
    ("participation_ratio", "Participation Ratio", axes[1, 0]),
    ("active_dims_frac", "Active Dims Fraction", axes[1, 1]),
]

for metric_key, title, ax in metrics:
    values = [summary_df.loc[summary_df["method"] == m, metric_key].values[0] for m in methods]
    bars = ax.bar(methods, values, color=["#1f77b4", "#ff7f0e"], alpha=0.7, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Log scale for anisotropy
    if metric_key == "anisotropy_ratio":
        ax.set_yscale("log")

plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "04_summary_metrics.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Number of Components for Variance Thresholds
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

thresholds = [50, 90, 95, 99]
x = np.arange(len(thresholds))
width = 0.35

for i, method in enumerate(methods):
    values = [summary_df.loc[summary_df["method"] == method, f"num_pc_{t}"].values[0] for t in thresholds]
    ax.bar(x + i * width, values, width, label=method, alpha=0.8, edgecolor="black", linewidth=1.5)

ax.set_xlabel("Variance Threshold (%)", fontsize=12)
ax.set_ylabel("Number of Principal Components", fontsize=12)
ax.set_title("PCs Required to Reach Variance Thresholds", fontsize=14, fontweight="bold")
ax.set_xticks(x + width / 2)
ax.set_xticklabels([f"{t}%" for t in thresholds])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "05_pc_thresholds.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

print()
print("=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
print()
print("Output files:")
print(f"  Summary metrics:     {summary_csv}")
print(f"  PCA variance:        {pca_csv}")
print(f"  Plots (5):           {CONFIG['output_dir']}/[01-05]_*.png")
print()
print("Key Findings:")
print(f"  Uniform ED:          {summary_df.loc[summary_df['method'] == 'uniform_10', 'effective_dimensionality'].values[0]:.2f}")
print(f"  Last ED:             {summary_df.loc[summary_df['method'] == 'last_10', 'effective_dimensionality'].values[0]:.2f}")
print()
print(f"  Uniform Anisotropy:  {summary_df.loc[summary_df['method'] == 'uniform_10', 'anisotropy_ratio'].values[0]:.4f}")
print(f"  Last Anisotropy:     {summary_df.loc[summary_df['method'] == 'last_10', 'anisotropy_ratio'].values[0]:.4f}")
print()
print(f"  Uniform PC90:        {summary_df.loc[summary_df['method'] == 'uniform_10', 'num_pc_90'].values[0]:.0f}")
print(f"  Last PC90:           {summary_df.loc[summary_df['method'] == 'last_10', 'num_pc_90'].values[0]:.0f}")
print()
