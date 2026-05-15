"""
Analyze compounding Delta % effect across training stages.
Shows how consistent uniform strategy perturbation adds up over time.

From proposal1_results.json, calculate:
- Per-stage Delta % = ((uniform - late) / late) * 100
- Cumulative multiplier effect
- Total divergence accumulation
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
    "metric": "latent_l2_mean",  # Only latents, as requested
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

print("=" * 80)
print("COMPOUNDING DELTA % ANALYSIS (Latent L2 Only)")
print("=" * 80)
print(f"Input:  {CONFIG['results_json']}")
print(f"Output: {CONFIG['output_dir']}")
print()

# ============================================================================
# LOAD DATA
# ============================================================================
print("[STEP 1] Load proposal1_results.json")

with open(CONFIG["results_json"], "r") as f:
    results = json.load(f)

print(f"✓ Loaded {len(results['methods'])} method results")
print()

# ============================================================================
# EXTRACT AND ORGANIZE DATA
# ============================================================================
print("[STEP 2] Extract latent_l2_mean for all stages")

# Parse method names and group by run
data = {}
for method_name, metrics in results["methods"].items():
    # Extract run name and stage from method_name (e.g., "late_10_stage0" -> "late_10", 0)
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

# ============================================================================
# CALCULATE DELTA %
# ============================================================================
print("[STEP 3] Calculate per-stage Delta %")

# Get stage numbers (intersection of both runs)
late_stages = set(data.get("late_10", {}).keys())
uniform_stages = set(data.get("uniform_10", {}).keys())
common_stages = sorted(late_stages & uniform_stages)

print(f"Common stages: {common_stages}")
print()

analysis_data = {
    "stage": [],
    "late_10": [],
    "uniform_10": [],
    "delta_absolute": [],
    "delta_percent": [],
    "cumulative_multiplier": [],
    "cumulative_divergence": [],
}

cumulative_mult = 1.0
cumulative_div = 0.0

for stage in common_stages:
    late_val = data["late_10"][stage]
    uniform_val = data["uniform_10"][stage]
    
    delta_abs = uniform_val - late_val
    delta_pct = (delta_abs / late_val) * 100
    
    # Cumulative multiplier: (1 + delta_pct/100)^stage
    multiplier_this_stage = 1.0 + (delta_pct / 100.0)
    cumulative_mult *= multiplier_this_stage
    
    # Cumulative divergence: absolute sum of deltas
    cumulative_div += delta_abs
    
    analysis_data["stage"].append(stage)
    analysis_data["late_10"].append(late_val)
    analysis_data["uniform_10"].append(uniform_val)
    analysis_data["delta_absolute"].append(delta_abs)
    analysis_data["delta_percent"].append(delta_pct)
    analysis_data["cumulative_multiplier"].append(cumulative_mult)
    analysis_data["cumulative_divergence"].append(cumulative_div)
    
    print(f"Stage {stage}:")
    print(f"  late_10:              {late_val:.4f}")
    print(f"  uniform_10:           {uniform_val:.4f}")
    print(f"  Δ absolute:           {delta_abs:.4f}")
    print(f"  Δ %:                  {delta_pct:.3f}%")
    print(f"  Cumulative mult:      {cumulative_mult:.6f}x")
    print(f"  Cumulative div sum:   {cumulative_div:.4f}")
    print()

# ============================================================================
# SAVE TO CSV
# ============================================================================
print("[STEP 4] Save Results to CSV")

df = pd.DataFrame(analysis_data)
csv_path = os.path.join(CONFIG["output_dir"], "compounding_delta_analysis.csv")
df.to_csv(csv_path, index=False)
print(f"✓ Saved: {csv_path}")
print()
print(df.to_string(index=False))
print()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("[STEP 5] Summary Statistics")

avg_delta_pct = df["delta_percent"].mean()
std_delta_pct = df["delta_percent"].std()
total_cumulative_divergence = df["cumulative_divergence"].iloc[-1]
final_multiplier = df["cumulative_multiplier"].iloc[-1]

print(f"Average Delta %:              {avg_delta_pct:.3f}%")
print(f"Std Dev Delta %:              {std_delta_pct:.3f}%")
print(f"Total Cumulative Divergence:  {total_cumulative_divergence:.4f}")
print(f"Final Cumulative Multiplier:  {final_multiplier:.6f}x")
print(f"Equivalent to:                {(final_multiplier - 1) * 100:.2f}% total effect")
print()

# ============================================================================
# CREATE PLOTS
# ============================================================================
print("[STEP 6] Generate Plots")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Per-Stage Delta %
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(
    df["stage"],
    df["delta_percent"],
    color="#1f77b4",
    alpha=0.7,
    edgecolor="black",
    linewidth=1.5,
)

# Add value labels on bars
for bar, val in zip(bars, df["delta_percent"]):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{val:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax.axhline(avg_delta_pct, color="red", linestyle="--", linewidth=2, label=f"Average: {avg_delta_pct:.2f}%")

ax.set_xlabel("Training Stage", fontsize=12)
ax.set_ylabel("Δ % (uniform vs late)", fontsize=12)
ax.set_title("Per-Stage Delta %: Uniform Strategy Gains Over Late Strategy", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "01_per_stage_delta_percent.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Cumulative Divergence (Absolute)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    df["stage"],
    df["cumulative_divergence"],
    marker="o",
    markersize=8,
    linewidth=3,
    color="#ff7f0e",
    label="Cumulative Divergence",
)

# Fill under curve
ax.fill_between(
    df["stage"],
    df["cumulative_divergence"],
    alpha=0.3,
    color="#ff7f0e",
)

# Add value labels on points
for stage, cum_div in zip(df["stage"], df["cumulative_divergence"]):
    ax.text(
        stage,
        cum_div + 0.01,
        f"{cum_div:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax.set_xlabel("Training Stage", fontsize=12)
ax.set_ylabel("Cumulative Absolute Divergence", fontsize=12)
ax.set_title("Compounding Effect: Total Divergence Accumulation Over Stages", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "02_cumulative_divergence.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Cumulative Multiplier Effect
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    df["stage"],
    df["cumulative_multiplier"],
    marker="s",
    markersize=8,
    linewidth=3,
    color="#2ca02c",
    label="Cumulative Multiplier",
)

# Fill under curve
ax.fill_between(
    df["stage"],
    1.0,
    df["cumulative_multiplier"],
    alpha=0.3,
    color="#2ca02c",
)

# Add value labels
for stage, mult in zip(df["stage"], df["cumulative_multiplier"]):
    ax.text(
        stage,
        mult + 0.0001,
        f"{mult:.5f}x",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax.axhline(1.0, color="black", linestyle="-", linewidth=1, alpha=0.5)
ax.set_xlabel("Training Stage", fontsize=12)
ax.set_ylabel("Cumulative Multiplier (1.0 = no effect)", fontsize=12)
ax.set_title("Compounding Multiplier: Cascade Effect of Consistent Gains", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "03_cumulative_multiplier.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Latent L2 Trajectories
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    df["stage"],
    df["late_10"],
    marker="o",
    markersize=8,
    linewidth=2.5,
    color="#d62728",
    label="late_10",
)
ax.plot(
    df["stage"],
    df["uniform_10"],
    marker="s",
    markersize=8,
    linewidth=2.5,
    color="#1f77b4",
    label="uniform_10",
)

# Shade difference region
ax.fill_between(
    df["stage"],
    df["late_10"],
    df["uniform_10"],
    alpha=0.2,
    color="gray",
    label="Gap (uniform advantage)",
)

ax.set_xlabel("Training Stage", fontsize=12)
ax.set_ylabel("Latent L2 Norm", fontsize=12)
ax.set_title("Latent L2 Trajectories: Persistent Uniform Advantage", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "04_latent_l2_trajectories.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Summary Metrics Panel
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Delta % distribution
axes[0, 0].hist(df["delta_percent"], bins=10, color="#1f77b4", alpha=0.7, edgecolor="black")
axes[0, 0].axvline(avg_delta_pct, color="red", linestyle="--", linewidth=2, label=f"Mean: {avg_delta_pct:.2f}%")
axes[0, 0].set_xlabel("Delta %", fontsize=11)
axes[0, 0].set_ylabel("Frequency", fontsize=11)
axes[0, 0].set_title("Distribution of Per-Stage Delta %", fontsize=12, fontweight="bold")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis="y")

# Subplot 2: Cumulative multiplier growth
axes[0, 1].bar(
    range(len(df["stage"])),
    df["cumulative_multiplier"] - 1.0,
    color="#2ca02c",
    alpha=0.7,
    edgecolor="black",
)
axes[0, 1].set_xticks(range(len(df["stage"])))
axes[0, 1].set_xticklabels(df["stage"])
axes[0, 1].set_xlabel("Stage", fontsize=11)
axes[0, 1].set_ylabel("Multiplier - 1 (effect magnitude)", fontsize=11)
axes[0, 1].set_title("Cumulative Effect Growth", fontsize=12, fontweight="bold")
axes[0, 1].grid(True, alpha=0.3, axis="y")

# Subplot 3: Per-stage absolute delta
axes[1, 0].bar(
    df["stage"],
    df["delta_absolute"],
    color="#ff7f0e",
    alpha=0.7,
    edgecolor="black",
)
axes[1, 0].set_xlabel("Stage", fontsize=11)
axes[1, 0].set_ylabel("Δ Absolute", fontsize=11)
axes[1, 0].set_title("Per-Stage Absolute Divergence", fontsize=12, fontweight="bold")
axes[1, 0].grid(True, alpha=0.3, axis="y")

# Subplot 4: Summary text
axes[1, 1].axis("off")
summary_text = f"""
KEY FINDINGS (Latent L2 Only)

Average per-stage Delta %:
  {avg_delta_pct:.3f}% ± {std_delta_pct:.3f}%

Total Accumulation:
  Cumulative Divergence: {total_cumulative_divergence:.4f}
  
Compounding Effect:
  Final Multiplier: {final_multiplier:.6f}x
  Total Effect: {(final_multiplier - 1) * 100:.2f}%
  
Interpretation:
  Uniform strategy consistently adds ~{avg_delta_pct:.2f}% 
  more perturbation at each stage.
  
  Over {len(common_stages)} stages, this compounds to 
  a {(final_multiplier - 1) * 100:.2f}% cumulative advantage,
  demonstrating the cascade effect of 
  consistent incremental gains.
"""
axes[1, 1].text(
    0.1,
    0.5,
    summary_text,
    fontsize=11,
    verticalalignment="center",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "05_summary_metrics_panel.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Saved: {plot_path}")

print()
print("=" * 80)
print("✅ COMPOUNDING ANALYSIS COMPLETE!")
print("=" * 80)
print()
print("📊 Output Files:")
print(f"  CSV:   {csv_path}")
print(f"  Plots: {CONFIG['output_dir']}/[01-05]_*.png")
print()
print("💡 Key Insight:")
print(f"   Uniform strategy gains ~{avg_delta_pct:.2f}% per stage")
print(f"   Over {len(common_stages)} stages → {(final_multiplier - 1) * 100:.2f}% total compounding effect")
print()
