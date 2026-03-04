 #!/usr/bin/env python3
"""
plot_metrics.py

Plots training-curve metrics vs mean_tv_bed_reward for selected runs.
Includes static reference points for FK Steering and Pretrained baselines.

Usage:
    python plot_metrics.py [--results_dir PATH] [--out_dir PATH] [--show]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

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
# Only these three runs are included; order controls legend order.
RUN_CONFIG = {
    "4_particles_incremental_branch_fk_tv_bed": dict(
        label="Ours",
        color="#00BCD4",   # cyan
        linestyle="-",
        linewidth=2.2,
        marker="o",
        markersize=5,
    ),
    "b2_tv_bed": dict(
        label="B2Diff",
        color="#E91E8C",   # hot pink
        linestyle="-",
        linewidth=2.2,
        marker="o",
        markersize=5,
    ),
    "ddpo_tv_bed": dict(
        label="DDPO",
        color="#FF9800",   # orange
        linestyle="--",
        linewidth=2.2,
        marker="o",
        markersize=5,
    ),
}

# ── Static baseline reference points ──────────────────────────────────────────
# These appear as single scatter markers in every plot.
STATIC_REFS = [
    dict(
        label="FK Steering Inference",
        reward=-4.077,
        values={
            "col_obj":                       48.54,
            "col_scene":                     79.54,
            "scenes_with_multiple_tv_stands": 30.0,
            "scenes_with_multiple_beds":      10.0,
        },
        color="#FFC107",   # gold / amber
        marker="+",
        markersize=14,
        markeredgewidth=2.5,
    ),
    dict(
        label="SD (Pretrained)",
        reward=-5.2489,
        values={
            "col_obj":                       53.8696,
            "col_scene":                     84.0,
            "scenes_with_multiple_tv_stands":  5.0,
            "scenes_with_multiple_beds":       2.0,
        },
        color="#757575",   # grey
        marker="D",
        markersize=8,
        markeredgewidth=1.5,
    ),
]

# ── Metrics & labels ──────────────────────────────────────────────────────────
X_COL = "mean_tv_bed_reward"
REWARD_CUTOFF = -1   # only keep stages where reward ≤ this value

METRIC_COLS = [
    "col_obj",
    "col_scene",
    "scenes_with_multiple_tv_stands",
    "scenes_with_multiple_beds",
]

METRIC_LABELS = {
    "col_obj":                        "Object Collision Rate (%)",
    "col_scene":                      "Scene Collision Rate (%)",
    "scenes_with_multiple_tv_stands": "Scenes w/ Multiple TV Stands (%)",
    "scenes_with_multiple_beds":      "Scenes w/ Multiple Beds (%)",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_runs(results_dir: Path) -> dict[str, pd.DataFrame]:
    runs: dict[str, pd.DataFrame] = {}
    for run_name in RUN_CONFIG:
        csv_path = results_dir / run_name / "metrics_table.csv"
        if not csv_path.exists():
            print(f"  [WARN] CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        # Keep only numeric reward rows
        df = df[pd.to_numeric(df[X_COL], errors="coerce").notna()].copy()
        df[X_COL] = df[X_COL].astype(float)
        # Apply reward cutoff: keep stages up to REWARD_CUTOFF
        df = df[df[X_COL] <= REWARD_CUTOFF].copy()
        # Sort by reward (ascending) for clean line plots
        df = df.sort_values(X_COL).reset_index(drop=True)
        if len(df) == 0:
            print(f"  [WARN] {run_name}: no rows after cutoff — skipped")
            continue
        runs[run_name] = df
        print(f"  Loaded {run_name:52s}  ({len(df)} stages,  "
              f"reward [{df[X_COL].min():.3f} … {df[X_COL].max():.3f}])")
    return runs


# ── Single metric plot ────────────────────────────────────────────────────────

def plot_metric(metric: str, runs: dict[str, pd.DataFrame],
                args, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # ── training curves ──
    for run_name, df in runs.items():
        cfg = RUN_CONFIG[run_name]
        if metric not in df.columns:
            print(f"  [WARN] {run_name}: '{metric}' missing — skipped")
            continue
        y = pd.to_numeric(df[metric], errors="coerce")
        mask = y.notna()
        x_vals = df[X_COL][mask].values
        y_vals = y[mask].values
        if len(x_vals) == 0:
            continue
        ax.plot(
            x_vals, y_vals,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            marker=cfg["marker"],
            markersize=cfg["markersize"],
            markevery=max(1, args.marker_every),
            alpha=args.alpha,
            label=cfg["label"],
            zorder=3,
        )

    # ── static reference points ──
    for ref in STATIC_REFS:
        if metric not in ref["values"]:
            continue
        ax.scatter(
            ref["reward"], ref["values"][metric],
            color=ref["color"],
            marker=ref["marker"],
            s=ref["markersize"] ** 2,
            linewidths=ref["markeredgewidth"],
            zorder=5,
            label=ref["label"],
        )

    ax.set_xlabel("Mean TV-Bed Reward", fontsize=13)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=13)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs Mean TV-Bed Reward",
                 fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.85,
              edgecolor="#cccccc", borderpad=0.8)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.tick_params(labelsize=11)
    fig.tight_layout()

    out_path = out_dir / f"{metric}_vs_reward.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)


# ── Summary 2×2 grid ─────────────────────────────────────────────────────────

def plot_summary_grid(runs: dict[str, pd.DataFrame], args, out_dir: Path) -> None:
    n_cols = 2
    n_rows = (len(METRIC_COLS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(13, 4.8 * n_rows),
                             squeeze=False)

    for idx, metric in enumerate(METRIC_COLS):
        ax = axes[idx // n_cols][idx % n_cols]

        for run_name, df in runs.items():
            cfg = RUN_CONFIG[run_name]
            if metric not in df.columns:
                continue
            y = pd.to_numeric(df[metric], errors="coerce")
            mask = y.notna()
            x_vals = df[X_COL][mask].values
            y_vals = y[mask].values
            if len(x_vals) == 0:
                continue
            ax.plot(x_vals, y_vals,
                    color=cfg["color"],
                    linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"] - 0.4,
                    marker=cfg["marker"],
                    markersize=cfg["markersize"] - 1,
                    markevery=max(1, args.marker_every),
                    alpha=args.alpha,
                    label=cfg["label"],
                    zorder=3)

        for ref in STATIC_REFS:
            if metric not in ref["values"]:
                continue
            ax.scatter(ref["reward"], ref["values"][metric],
                       color=ref["color"],
                       marker=ref["marker"],
                       s=(ref["markersize"] - 2) ** 2,
                       linewidths=ref["markeredgewidth"],
                       label=ref["label"],
                       zorder=5)

        ax.set_xlabel("Mean TV-Bed Reward", fontsize=11)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_title(METRIC_LABELS.get(metric, metric),
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.85,
                  edgecolor="#cccccc")
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.tick_params(labelsize=10)

    # Hide any spare subplot cells
    for idx in range(len(METRIC_COLS), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Metrics vs Mean TV-Bed Reward", fontsize=15,
                 fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = out_dir / "all_metrics_summary.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n  Summary grid saved: {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot per-metric training curves.")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out_dir",     default=DEFAULT_OUT_DIR)
    parser.add_argument("--show",        action="store_true")
    parser.add_argument("--alpha",       type=float, default=0.85)
    parser.add_argument("--marker_every", type=int,  default=5)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nScanning: {results_dir}")
    runs = load_runs(results_dir)
    if not runs:
        print("No runs found — nothing to plot.")
        return

    print(f"\nDrawing {len(METRIC_COLS)} individual plots …")
    for metric in METRIC_COLS:
        plot_metric(metric, runs, args, out_dir)

    print("\nDrawing summary grid …")
    plot_summary_grid(runs, args, out_dir)

    print(f"\nDone — all plots written to: {out_dir}\n")


if __name__ == "__main__":
    main()
