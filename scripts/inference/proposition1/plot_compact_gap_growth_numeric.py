"""
Create a compact, no-axis bar chart for compounding delta analysis showing absolute delta numbers.
Reads CSV outputs/proposition1_compounding_analysis/compounding_delta_analysis.csv
Writes outputs/proposition1_compounding_analysis/02_gap_growth.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "outputs/proposition1_compounding_analysis/compounding_delta_analysis.csv"
OUT_PNG = "outputs/proposition1_compounding_analysis/02_gap_growth.png"

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

if not os.path.exists(CSV_PATH):
    raise SystemExit(f"Missing CSV: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

stages = df['stage'].tolist()
deltas = df['delta_absolute'].tolist()

# Plot: white background, orange bars, no axes, minimal margins
plt.figure(figsize=(3.2, 1.2), dpi=300)
ax = plt.gca()

bars = ax.bar(stages, deltas, color='#ff7f0e', edgecolor='none')

# Add small numeric labels above bars (absolute values with 3 decimals)
for bar, val in zip(bars, deltas):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + (max(deltas) * 0.01),
        f"{val:.3f}",
        ha='center', va='bottom', fontsize=6, color='black'
    )

# Remove axes, ticks, spines
ax.set_axis_off()

# Tight layout with minimal padding
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Saved compact numeric gap growth plot to: {OUT_PNG}")
