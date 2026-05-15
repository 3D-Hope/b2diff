"""
Cumulative L2 norm across stages - shows compounding divergence.
Stage on x-axis, cumulative sum of latent L2 on y-axis.

Stage 0: late=118.06, uniform=118.69 (gap=0.59)
Stage 1: late=118.06+118.13=236.19, uniform=118.69+119.0=237.69 (gap increases!)
Stage 2: late=236.19+118.13=354.32, uniform=237.69+119.06=356.75 (gap widens!)
...
Shows total perturbation accumulation over training.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "results_json": "outputs/proposal1_analysis/proposal1_results.json",
    "output_dir": "outputs/proposition1_compounding_analysis",
    "metric": "latent_l2_mean",  # Only latents
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

print("=" * 80)
print("CUMULATIVE PERTURBATION ACROSS STAGES")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================
print("[STEP 1] Load and Parse Data")

with open(CONFIG["results_json"], "r") as f:
    results = json.load(f)

# Extract and organize data by stage
data = {}
for method_name, metrics in results["methods"].items():
    parts = method_name.rsplit("_", 1)
    if len(parts) == 2:
        run_name = parts[0]
        try:
            stage_num = int(parts[1].replace("stage", ""))
        except:
            continue
    else:
        continue
    
    if run_name not in data:
        data[run_name] = {}
    
    latent_l2 = metrics.get(CONFIG["metric"])
    if latent_l2 is not None and not np.isinf(latent_l2):
        data[run_name][stage_num] = latent_l2

print(f"✓ Found runs: {list(data.keys())}")
for run_name, stages in data.items():
    print(f"  {run_name}: stages {sorted(stages.keys())}")
print()

# Get common stages
late_stages = sorted(data.get("late_10", {}).keys())
uniform_stages = sorted(data.get("uniform_10", {}).keys())
common_stages = sorted(set(late_stages) & set(uniform_stages))

print(f"Common stages: {common_stages}")
print()

# ============================================================================
# COMPUTE CUMULATIVE SUMS
# ============================================================================
print("[STEP 2] Compute Cumulative L2 Sums")

cumsum_data = {
    "stage": [],
    "late_10_cumsum": [],
    "uniform_10_cumsum": [],
    "gap": [],
    "gap_percent": [],
}

late_cumsum = 0.0
uniform_cumsum = 0.0

for stage in common_stages:
    late_val = data["late_10"][stage]
    uniform_val = data["uniform_10"][stage]
    
    late_cumsum += late_val
    uniform_cumsum += uniform_val
    gap = uniform_cumsum - late_cumsum
    gap_pct = (gap / late_cumsum) * 100 if late_cumsum > 0 else 0
    
    cumsum_data["stage"].append(stage)
    cumsum_data["late_10_cumsum"].append(late_cumsum)
    cumsum_data["uniform_10_cumsum"].append(uniform_cumsum)
    cumsum_data["gap"].append(gap)
    cumsum_data["gap_percent"].append(gap_pct)
    
    print(f"Stage {stage}:")
    print(f"  late_10 (this stage):   {late_val:.4f}")
    print(f"  uniform_10 (this stage): {uniform_val:.4f}")
    print(f"  late_10 cumsum:         {late_cumsum:.4f}")
    print(f"  uniform_10 cumsum:      {uniform_cumsum:.4f}")
    print(f"  Gap (uniform - late):   {gap:.4f} ({gap_pct:.2f}%)")
    print()

# ============================================================================
# SAVE TO CSV
# ============================================================================
print("[STEP 3] Save Results")

df = pd.DataFrame(cumsum_data)
csv_path = os.path.join(CONFIG["output_dir"], "cumulative_l2_across_stages.csv")
df.to_csv(csv_path, index=False)
print(f"✓ Saved: {csv_path}")
print()
print(df.to_string(index=False))
print()

# ============================================================================
# CREATE PLOTS
# ============================================================================
print("[STEP 4] Generate Plots")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Main Cumulative Trajectories
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(
    df["stage"],
    df["late_10_cumsum"],
    marker="o",
    markersize=10,
    linewidth=3.5,
    color="#d62728",
    label="late_10 (cumulative)",
)
ax.plot(
    df["stage"],
    df["uniform_10_cumsum"],
    marker="s",
    markersize=10,
    linewidth=3.5,
    color="#1f77b4",
    label="uniform_10 (cumulative)",
)

# Shade divergence region
ax.fill_between(
    df["stage"],
    df["late_10_cumsum"],
    df["uniform_10_cumsum"],
    alpha=0.25,
    color="gray",
    label="Cumulative Divergence (uniform advantage)",
)

# Add value labels on points
for stage, late, uniform in zip(df["stage"], df["late_10_cumsum"], df["uniform_10_cumsum"]):
    ax.text(stage, late - 5, f"{late:.1f}", ha="center", va="top", fontsize=9, color="#d62728", fontweight="bold")
    ax.text(stage, uniform + 5, f"{uniform:.1f}", ha="center", va="bottom", fontsize=9, color="#1f77b4", fontweight="bold")

ax.set_xlabel("Training Stage", fontsize=13, fontweight="bold")
ax.set_ylabel("Cumulative Latent L2 Norm", fontsize=13, fontweight="bold")
ax.set_title("Cumulative Perturbation Across Training Stages\n(Shows Compounding Divergence)", fontsize=15, fontweight="bold")
ax.legend(fontsize=12, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xticks(df["stage"])
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "01_cumulative_trajectories.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Gap Growth Over Stages
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(df["stage"], df["gap"], color="#ff7f0e", alpha=0.7, edgecolor="black", linewidth=1.5)

# Add value labels
for bar, val in zip(bars, df["gap"]):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_xlabel("Training Stage", fontsize=12, fontweight="bold")
ax.set_ylabel("Cumulative Gap (uniform - late)", fontsize=12, fontweight="bold")
ax.set_title("Cumulative Gap Growth: Uniform Advantage Compounds", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.set_xticks(df["stage"])
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "02_gap_growth.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Gap as Percentage
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    df["stage"],
    df["gap_percent"],
    marker="D",
    markersize=9,
    linewidth=3,
    color="#2ca02c",
    label="Gap as % of late_10 cumsum",
)

# Add value labels
for stage, gap_pct in zip(df["stage"], df["gap_percent"]):
    ax.text(
        stage,
        gap_pct + 0.03,
        f"{gap_pct:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax.set_xlabel("Training Stage", fontsize=12, fontweight="bold")
ax.set_ylabel("Gap as % of late_10 Cumulative Sum", fontsize=12, fontweight="bold")
ax.set_title("Relative Gap Growth: Percentage Advantage Over Stages", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xticks(df["stage"])
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "03_gap_percentage.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Stage-wise vs Cumulative Comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Per-stage L2 values
stage_gaps = [data["uniform_10"][s] - data["late_10"][s] for s in common_stages]
ax1.bar(df["stage"], stage_gaps, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=1.5)
for stage, gap in zip(df["stage"], stage_gaps):
    ax1.text(stage, gap + 0.01, f"{gap:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_xlabel("Training Stage", fontsize=11, fontweight="bold")
ax1.set_ylabel("Per-Stage Gap (uniform - late)", fontsize=11, fontweight="bold")
ax1.set_title("Per-Stage Perturbation Difference", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_xticks(df["stage"])

# Right: Cumulative gap
ax2.bar(df["stage"], df["gap"], color="#ff7f0e", alpha=0.7, edgecolor="black", linewidth=1.5)
for stage, gap in zip(df["stage"], df["gap"]):
    ax2.text(stage, gap + 0.1, f"{gap:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_xlabel("Training Stage", fontsize=11, fontweight="bold")
ax2.set_ylabel("Cumulative Gap", fontsize=11, fontweight="bold")
ax2.set_title("Cumulative Divergence (Compounds Over Stages)", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_xticks(df["stage"])

plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "04_per_stage_vs_cumulative.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Summary Panel
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# Main plot: cumulative trajectories (large)
ax_main = fig.add_subplot(gs[0:2, :])
ax_main.plot(df["stage"], df["late_10_cumsum"], marker="o", markersize=10, linewidth=3.5, 
             color="#d62728", label="late_10")
ax_main.plot(df["stage"], df["uniform_10_cumsum"], marker="s", markersize=10, linewidth=3.5, 
             color="#1f77b4", label="uniform_10")
ax_main.fill_between(df["stage"], df["late_10_cumsum"], df["uniform_10_cumsum"], 
                     alpha=0.25, color="gray")
ax_main.set_ylabel("Cumulative Latent L2", fontsize=12, fontweight="bold")
ax_main.set_title("Cumulative Perturbation: Uniform Advantage Compounds Over Stages", 
                  fontsize=14, fontweight="bold")
ax_main.legend(fontsize=11, loc="upper left")
ax_main.grid(True, alpha=0.3)
ax_main.set_xticks(df["stage"])

# Bottom left: Gap growth
ax_bl = fig.add_subplot(gs[2, 0])
ax_bl.bar(df["stage"], df["gap"], color="#ff7f0e", alpha=0.7, edgecolor="black", linewidth=1.5)
ax_bl.set_xlabel("Stage", fontsize=10, fontweight="bold")
ax_bl.set_ylabel("Cumulative Gap", fontsize=10, fontweight="bold")
ax_bl.set_title("Gap Growth", fontsize=11, fontweight="bold")
ax_bl.grid(True, alpha=0.3, axis="y")
ax_bl.set_xticks(df["stage"])

# Bottom right: Summary text
ax_br = fig.add_subplot(gs[2, 1])
ax_br.axis("off")

final_gap = df["gap"].iloc[-1]
final_gap_pct = df["gap_percent"].iloc[-1]
total_late = df["late_10_cumsum"].iloc[-1]
total_uniform = df["uniform_10_cumsum"].iloc[-1]

summary_text = f"""
CUMULATIVE COMPOUNDING EFFECT

Final Stage: {common_stages[-1]}

Late-10 Total:    {total_late:.2f}
Uniform-10 Total: {total_uniform:.2f}
Cumulative Gap:   {final_gap:.2f} ({final_gap_pct:.2f}%)

Interpretation:
Over {len(common_stages)} training stages, uniform strategy 
accumulates {final_gap:.2f} units more latent 
perturbation than late strategy.

Each stage adds ~{(final_gap/len(common_stages)):.2f} units,
compounding to total advantage of {final_gap_pct:.2f}%.
"""

ax_br.text(
    0.1, 0.5, summary_text,
    fontsize=11, verticalalignment="center",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
)

plt.suptitle("Cumulative Analysis: How Consistent Gains Compound", fontsize=16, fontweight="bold", y=0.995)
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "05_summary_panel.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

print()
print("=" * 80)
print("✅ CUMULATIVE ANALYSIS COMPLETE!")
print("=" * 80)
print()
print("📊 Output Files:")
print(f"  CSV:   {csv_path}")
print(f"  Plots: {CONFIG['output_dir']}/[01-05]_*.png")
print()
print(f"💡 Key Finding:")
print(f"   Final cumulative gap: {df['gap'].iloc[-1]:.2f} units ({df['gap_percent'].iloc[-1]:.2f}%)")
print(f"   Over {len(common_stages)} stages: {df['late_10_cumsum'].iloc[-1]:.2f} → {df['uniform_10_cumsum'].iloc[-1]:.2f}")
print()
