#!/usr/bin/env python3
"""
plot_collision_vs_reward_eccv.py

Plots Object Collision Rate (%) vs Mean Reward with ECCV paper aesthetics:
- Spline smoothing
- Local standard deviation bands
- specific color scheme and fonts
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

# ── Global style: clean white with warm-grey axes background ─────────────
plt.rcParams.update({
    'font.family':      'serif',
    'axes.facecolor':   '#ffffff',
    'figure.facecolor': '#ffffff',
    'axes.edgecolor':   '#bbbbbb',
    'xtick.color':      '#555555',
    'ytick.color':      '#555555',
    'text.color':       '#222222',
    'font.size':        12,
})

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RESULTS_DIR = (
    "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch"
    "/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion"
    "/output/full_predicted_results"
)
DEFAULT_OUT_DIR = (
    "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch"
    "/3d_b2diff/b2diff/logs/metric_plots"
)

# ── Run selection & styling ────────────────────────────────────────────────────
RUN_CONFIG = {
    "4_particles_incremental_branch_fk_tv_bed": dict(
        label="Ours",
        color='#3a86ff',   # Blue
    ),
    "b2_tv_bed": dict(
        label=r"$B^2$-DiffuRL",
        color='#ff006e',   # Pink
    ),
    "ddpo_tv_bed": dict(
        label="DDPO",
        color='#06d6a0',   # Green
    ),
}

# ── Static baseline reference points ──────────────────────────────────────────
STATIC_REFS = [
    dict(
        label="FK Steering Inference",
        reward=-4.077,
        values={"col_obj": 48.54},
        color="#e07b00", # Orange
        marker="p",
    ),
    dict(
        label="SD", # Pretrained
        reward=-5.2489,
        values={"col_obj": 53.8696},
        color="#888888", # Grey
        marker="D",
    ),
]

# ── Metrics & labels ──────────────────────────────────────────────────────────
X_COL = "mean_tv_bed_reward"
Y_COL = "col_obj"
REWARD_CUTOFF = -1

# ── Data loading ──────────────────────────────────────────────────────────────

def load_runs(results_dir):
    runs: dict[str, pd.DataFrame] = {}
    for run_name in RUN_CONFIG:
        csv_path = results_dir / run_name / "metrics_table.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df = df[pd.to_numeric(df[X_COL], errors="coerce").notna()].copy()
        df[X_COL] = df[X_COL].astype(float)
        df = df[df[X_COL] <= REWARD_CUTOFF].copy()
        df = df.sort_values(X_COL).reset_index(drop=True)
        if len(df) == 0:
            continue
        runs[run_name] = df
    return runs

def smooth_data(data, sigma=2.0):
    if len(data) == 0: return data
    return gaussian_filter1d(data, sigma=sigma, mode='nearest')

def compute_local_stds(ys, local_window=3):
    n = len(ys)
    local_stds = np.zeros(n)
    for i in range(n):
        lo = max(0, i - local_window // 2)
        hi = min(n, i + local_window // 2 + 1)
        local_stds[i] = np.std(ys[lo:hi]) if (hi - lo) > 1 else 0.0
    return local_stds

def plot_eccv_collision(runs, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    plotted_any = False
    
    # Pre-compute global min/max for x-axis normalization
    all_x = []
    for run_name, df in runs.items():
        if Y_COL in df.columns:
            y = pd.to_numeric(df[Y_COL], errors="coerce")
            x_vals = df[X_COL][y.notna()].values
            all_x.extend(x_vals)
    for ref in STATIC_REFS:
        if Y_COL in ref["values"]:
            all_x.append(ref["reward"])
            
    min_x, max_x = min(all_x), max(all_x)
    x_range = (max_x - min_x) if (max_x - min_x) > 1e-8 else 1.0

    for run_name, df in runs.items():
        if Y_COL not in df.columns:
            continue
        cfg = RUN_CONFIG[run_name]
        
        y = pd.to_numeric(df[Y_COL], errors="coerce")
        mask = y.notna()
        # Normalize the x_vals
        x_vals = (df[X_COL][mask].values - min_x) / x_range
        y_vals = y[mask].values
        
        if len(x_vals) == 0:
            continue
            
        plotted_any = True
        
        # Sort values
        sort_idx = np.argsort(x_vals)
        xs = x_vals[sort_idx]
        ys = y_vals[sort_idx]
        
        y_stds = compute_local_stds(ys, local_window=7)
        ys = smooth_data(ys, sigma=4.0)
        y_stds = smooth_data(y_stds, sigma=4.0)

        # Make it sparser for smoother curves, but explicitly include endpoints
        num_points_to_keep = min(len(xs), 12)  # Keep about 12 points
        if len(xs) > num_points_to_keep:
            indices = np.unique(np.linspace(0, len(xs) - 1, num_points_to_keep).astype(int))
            xs = xs[indices]
            ys = ys[indices]
            y_stds = y_stds[indices]
        
        color = cfg['color']
        label = cfg['label']
        
        if len(xs) > 3:
            t = np.linspace(0, 1, len(xs))
            t_smooth = np.linspace(0, 1, 300)
            
            # Use k=2 or k=3
            try:
                x_sm = make_interp_spline(t, xs, k=3)(t_smooth)
                y_sm = make_interp_spline(t, ys, k=3)(t_smooth)
                ystd_sm = make_interp_spline(t, y_stds, k=3)(t_smooth)
            except ValueError:
                x_sm = make_interp_spline(t, xs, k=2)(t_smooth)
                y_sm = make_interp_spline(t, ys, k=2)(t_smooth)
                ystd_sm = np.interp(t_smooth, t, y_stds)
                
            ax.fill_between(x_sm, y_sm - ystd_sm, y_sm + ystd_sm,
                            color=color, alpha=0.20, linewidth=0)
            ax.plot(x_sm, y_sm, color=color, alpha=0.93, linewidth=3.5,
                    label=label, solid_capstyle='round', solid_joinstyle='round')
        else:
            ax.fill_between(xs, ys - y_stds, ys + y_stds,
                            color=color, alpha=0.20, linewidth=0)
            ax.plot(xs, ys, color=color, alpha=0.93, linewidth=3.5, label=label)

    if plotted_any:
        # ── Axes background + grid ────────────────────────────────────────────
        ax.set_facecolor('#ffffff')
        ax.set_axisbelow(True)
        ax.grid(True, linestyle=':', linewidth=0.9, color='#d0d0d0', alpha=1.0)

        # Spines: only left + bottom, soft grey
        for sp in ['right', 'top']:
            ax.spines[sp].set_visible(False)
        for sp in ['left', 'bottom']:
            ax.spines[sp].set_linewidth(1.1)
            ax.spines[sp].set_color('#aaaaaa')

        # Ticks
        ax.tick_params(axis='both', which='major',
                       labelsize=13, width=1.0, length=5,
                       color='#aaaaaa', labelcolor='#333333', pad=4)

        # Axis labels
        ax.set_xlabel("Spatial Reward $\\rightarrow$", fontsize=17, fontweight='bold',
                      color='#222222', labelpad=12)
        ax.set_ylabel("Collision Rate (%) $\\rightarrow$", fontsize=17, fontweight='bold',
                      color='#222222', labelpad=12)

        # ── Baseline single points ────────────────────────────────────────────
        for ref in STATIC_REFS:
            if Y_COL not in ref["values"]:
                continue
            
            # Normalize baseline x-coord
            norm_x = (ref["reward"] - min_x) / x_range
            
            ax.scatter(norm_x, ref["values"][Y_COL],
                       color=ref["color"], marker=ref["marker"],
                       s=180, zorder=9, edgecolors='white', linewidths=1.5,
                       label=ref["label"])

        # Legend
        leg = ax.legend(frameon=True, fontsize=14.5, loc='best',
                        fancybox=True, framealpha=0.93, shadow=False,
                        edgecolor='#cccccc', borderpad=0.8, handlelength=2.0,
                        labelspacing=0.5, ncol=1)
        leg.get_frame().set_linewidth(0.8)

        fig.tight_layout()

        out_path = out_dir / "object_collision_vs_reward.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        fig.savefig(out_dir / "object_collision_vs_reward.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"  Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot valid ECCV-style Object Collision vs Reward.")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out_dir",     default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nScanning: {results_dir}")
    runs = load_runs(results_dir)
    if not runs:
        print("No runs found — nothing to plot.")
        return

    plot_eccv_collision(runs, out_dir)
    print("\nDone.\n")

if __name__ == "__main__":
    main()
