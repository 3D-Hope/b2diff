"""
Create a compact, no-axis, tight-margin plot of cumulative divergence (gap growth)
from the compounding analysis CSV.
Saves to outputs/proposition1_compounding_analysis/02_gap_growth.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "outputs/proposition1_compounding_analysis/compounding_delta_analysis.csv"
OUT_PATH = "outputs/proposition1_compounding_analysis/02_gap_growth.png"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Input CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
# Expect columns: stage, cumulative_divergence
x = df['stage'].to_numpy()
y = df['cumulative_divergence'].to_numpy()

# Labels to print above bars: just the absolute delta (no percentages)
has_abs = 'delta_absolute' in df.columns
abs_vals = df['delta_absolute'].to_numpy() if has_abs else None

# Create compact figure with orange bars on white background
fig, ax = plt.subplots(figsize=(3.0, 1.2), dpi=300)

color = '#ff7f0e'
bars = ax.bar(x, y, color=color, edgecolor='#cc6600')

# Add small numeric labels above each bar (absolute values only, NO percentages)
for i, bar in enumerate(bars):
    h = bar.get_height()
    if has_abs:
        label = f"{abs_vals[i]:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(y) * 0.01,
            label,
            ha='center', va='bottom', fontsize=6, color='#333333'
        )

# Remove axes, ticks, spines
ax.set_axis_off()
for spine in ax.spines.values():
    spine.set_visible(False)

# Tight margins and white background
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.patch.set_facecolor('white')

# Save with no extra padding
fig.savefig(OUT_PATH, bbox_inches='tight', pad_inches=0, dpi=300)
plt.close(fig)
print(f"Saved compact gap growth plot to: {OUT_PATH}")
