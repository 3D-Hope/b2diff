 #!/usr/bin/env python3
"""
plot_metrics.py

Plots training-curve metrics for selected runs.

Modes
-----
  collision  (--mode collision)
      X-axis = Object Collision Rate (col_obj).
      High collision = start of training (left), low = end (right).
      Reveals reward hacking: does avg_num_obj drop as col_obj falls?

  tv_bed     (--mode tv_bed)
      X-axis = Mean TV-Bed Reward (mean_tv_bed_reward).

Usage:
    python plot_metrics.py [--mode collision|tv_bed] [--results_dir PATH]
                           [--out_dir PATH] [--show]
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

# ── Shared run styling (same colours/markers for both modes) ──────────────────
_STYLE = {
    "ours":  dict(color="#00BCD4", linestyle="-",  linewidth=2.2, marker="o", markersize=5),
    "b2":    dict(color="#E91E8C", linestyle="-",  linewidth=2.2, marker="o", markersize=5),
    "ddpo":  dict(color="#FF9800", linestyle="--", linewidth=2.2, marker="o", markersize=5),
}

# ── Per-mode configuration ────────────────────────────────────────────────────
MODE_CONFIG = {
    # ── collision reward ──────────────────────────────────────────────────────
    "collision": dict(
        x_col="col_obj",
        x_label="Object Collision Rate (%)",
        # Sort descending so training flows left (high collision) → right (low)
        x_ascending=False,
        invert_xaxis=True,   # high collision on left = start of training
        reward_cutoff=100,
        title_suffix="Minimize Collision Reward",
        run_config={
            "ours_3d_collision": dict(label="Ours", **_STYLE["ours"]),
            "b2_3d_collision":   dict(label="B2Diff", **_STYLE["b2"]),
            "ddpo_3d_collision": dict(label="DDPO",  **_STYLE["ddpo"]),
        },
        # x-coordinate = col_obj value for each baseline
        static_refs=[
            dict(
                label="FK Steering Inference",
                x=48.54,
                values={
                    "col_scene":                     79.54,
                    "scenes_with_multiple_tv_stands": 30.0,
                    "scenes_with_multiple_beds":      10.0,
                    "avg_num_obj":                    5.16,
                },
                color="#FFC107", marker="+",  markersize=14, markeredgewidth=2.5,
            ),
            dict(
                label="SD (Pretrained)",
                x=53.8696,
                values={
                    "col_scene":                     84.0,
                    "scenes_with_multiple_tv_stands":  5.0,
                    "scenes_with_multiple_beds":       2.0,
                    "avg_num_obj":                    5.16,
                },
                color="#757575", marker="D",  markersize=8,  markeredgewidth=1.5,
            ),
        ],
        metric_cols=[
            "col_scene",
            "scenes_with_multiple_tv_stands",
            "scenes_with_multiple_beds",
            "avg_num_obj",
        ],
    ),

    # ── TV-bed reward ─────────────────────────────────────────────────────────
    "tv_bed": dict(
        x_col="mean_tv_bed_reward",
        x_label="Mean TV-Bed Reward",
        x_ascending=True,
        invert_xaxis=False,
        reward_cutoff=1,
        title_suffix="TV-Bed Reward",
        run_config={
            "ours_3d_tv_bed": dict(label="Ours",   **_STYLE["ours"]),
            "b2_3d_tv_bed":   dict(label="B2Diff", **_STYLE["b2"]),
            "ddpo_3d_tv_bed": dict(label="DDPO",   **_STYLE["ddpo"]),
        },
        # x-coordinate = mean_tv_bed_reward value for each baseline
        static_refs=[
            dict(
                label="FK Steering Inference",
                x=-4.077,
                values={
                    "col_obj":                       48.54,
                    "col_scene":                     79.54,
                    "scenes_with_multiple_tv_stands": 30.0,
                    "scenes_with_multiple_beds":      10.0,
                    "avg_num_obj":                    5.16,
                },
                color="#FFC107", marker="+",  markersize=14, markeredgewidth=2.5,
            ),
            dict(
                label="SD (Pretrained)",
                x=-5.2489,
                values={
                    "col_obj":                       53.8696,
                    "col_scene":                     84.0,
                    "scenes_with_multiple_tv_stands":  5.0,
                    "scenes_with_multiple_beds":       2.0,
                    "avg_num_obj":                    5.16,
                },
                color="#757575", marker="D",  markersize=8,  markeredgewidth=1.5,
            ),
        ],
        metric_cols=[
            "col_obj",
            "col_scene",
            "scenes_with_multiple_tv_stands",
            "scenes_with_multiple_beds",
            "avg_num_obj",
        ],
    ),
}

# ── Metric display labels (shared) ───────────────────────────────────────────
METRIC_LABELS = {
    "col_obj":                        "Object Collision Rate (%)",
    "col_scene":                      "Scene Collision Rate (%)",
    "scenes_with_multiple_tv_stands": "Scenes w/ Multiple TV Stands (%)",
    "scenes_with_multiple_beds":      "Scenes w/ Multiple Beds (%)",
    "avg_num_obj":                    "Avg. Number of Objects",
    "mean_tv_bed_reward":             "Mean TV-Bed Reward",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_runs(results_dir: Path, cfg: dict) -> dict[str, pd.DataFrame]:
    x_col      = cfg["x_col"]
    ascending  = cfg["x_ascending"]
    cutoff     = cfg["reward_cutoff"]
    run_config = cfg["run_config"]

    runs: dict[str, pd.DataFrame] = {}
    for run_name in run_config:
        csv_path = results_dir / run_name / "metrics_table.csv"
        if not csv_path.exists():
            print(f"  [WARN] CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df = df[pd.to_numeric(df[x_col], errors="coerce").notna()].copy()
        df[x_col] = df[x_col].astype(float)
        df = df[df[x_col] <= cutoff].copy()
        # Sort by stage so each run's natural training trajectory is preserved,
        # then use x_col for the x-axis (may be non-monotone but reflects reality)
        df = df.sort_values("stage").drop_duplicates(subset=["stage"]).reset_index(drop=True)
        if not ascending:
            # For collision mode: reverse so x goes high → low (start → end of training)
            df = df.iloc[::-1].reset_index(drop=True)
        if len(df) == 0:
            print(f"  [WARN] {run_name}: no rows after filtering — skipped")
            continue
        runs[run_name] = df
        print(f"  Loaded {run_name:52s}  ({len(df)} stages,  "
              f"{x_col} [{df[x_col].min():.3f} … {df[x_col].max():.3f}])")
    return runs


# ── Single metric plot ────────────────────────────────────────────────────────

def plot_metric(metric: str, runs: dict[str, pd.DataFrame],
                mode_cfg: dict, args, out_dir: Path) -> None:
    x_col       = mode_cfg["x_col"]
    x_label     = mode_cfg["x_label"]
    title_sfx   = mode_cfg["title_suffix"]
    run_config  = mode_cfg["run_config"]
    static_refs = mode_cfg["static_refs"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # ── training curves ──
    for run_name, df in runs.items():
        rc = run_config[run_name]
        if metric not in df.columns:
            print(f"  [WARN] {run_name}: '{metric}' missing — skipped")
            continue
        y = pd.to_numeric(df[metric], errors="coerce")
        mask = y.notna()
        x_vals = df[x_col][mask].values
        y_vals = y[mask].values
        if len(x_vals) == 0:
            continue
        ax.plot(
            x_vals, y_vals,
            color=rc["color"], linestyle=rc["linestyle"],
            linewidth=rc["linewidth"], marker=rc["marker"],
            markersize=rc["markersize"],
            markevery=max(1, args.marker_every),
            alpha=args.alpha, label=rc["label"], zorder=3,
        )

    # ── static reference points ──
    for ref in static_refs:
        if metric not in ref["values"]:
            continue
        ax.scatter(
            ref["x"], ref["values"][metric],
            color=ref["color"], marker=ref["marker"],
            s=ref["markersize"] ** 2, linewidths=ref["markeredgewidth"],
            zorder=5, label=ref["label"],
        )

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=13)
    ax.set_title(
        f"{METRIC_LABELS.get(metric, metric)} vs {x_label}\n({title_sfx})",
        fontsize=13, fontweight="bold", pad=10,
    )
    if mode_cfg.get("invert_xaxis"):
        ax.invert_xaxis()
    ax.legend(fontsize=10, loc="best", framealpha=0.85,
              edgecolor="#cccccc", borderpad=0.8)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.tick_params(labelsize=11)
    fig.tight_layout()

    out_path = out_dir / f"{metric}_vs_{x_col}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)


# ── Summary grid ─────────────────────────────────────────────────────────────

def plot_summary_grid(runs: dict[str, pd.DataFrame],
                      mode_cfg: dict, args, out_dir: Path) -> None:
    x_col       = mode_cfg["x_col"]
    x_label     = mode_cfg["x_label"]
    title_sfx   = mode_cfg["title_suffix"]
    run_config  = mode_cfg["run_config"]
    static_refs = mode_cfg["static_refs"]
    metric_cols = mode_cfg["metric_cols"]

    n_cols = 2
    n_rows = (len(metric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(13, 4.8 * n_rows), squeeze=False)

    for idx, metric in enumerate(metric_cols):
        ax = axes[idx // n_cols][idx % n_cols]

        for run_name, df in runs.items():
            rc = run_config[run_name]
            if metric not in df.columns:
                continue
            y = pd.to_numeric(df[metric], errors="coerce")
            mask = y.notna()
            x_vals = df[x_col][mask].values
            y_vals = y[mask].values
            if len(x_vals) == 0:
                continue
            ax.plot(x_vals, y_vals,
                    color=rc["color"], linestyle=rc["linestyle"],
                    linewidth=rc["linewidth"] - 0.4,
                    marker=rc["marker"], markersize=rc["markersize"] - 1,
                    markevery=max(1, args.marker_every),
                    alpha=args.alpha, label=rc["label"], zorder=3)

        for ref in static_refs:
            if metric not in ref["values"]:
                continue
            ax.scatter(ref["x"], ref["values"][metric],
                       color=ref["color"], marker=ref["marker"],
                       s=(ref["markersize"] - 2) ** 2,
                       linewidths=ref["markeredgewidth"],
                       label=ref["label"], zorder=5)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12, fontweight="bold")
        if mode_cfg.get("invert_xaxis"):
            ax.invert_xaxis()
        ax.legend(fontsize=8.5, loc="best", framealpha=0.85, edgecolor="#cccccc")
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.tick_params(labelsize=10)

    for idx in range(len(metric_cols), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(f"Metrics vs {x_label} ({title_sfx})",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = out_dir / f"all_metrics_summary_{x_col}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n  Summary grid saved: {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot per-metric training curves.")
    parser.add_argument("--mode",         default="collision",
                        choices=["collision", "tv_bed"],
                        help="Which reward axis to use (default: collision)")
    parser.add_argument("--results_dir",  default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out_dir",      default=DEFAULT_OUT_DIR)
    parser.add_argument("--show",         action="store_true")
    parser.add_argument("--alpha",        type=float, default=0.85)
    parser.add_argument("--marker_every", type=int,   default=5)
    args = parser.parse_args()

    mode_cfg    = MODE_CONFIG[args.mode]
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMode: {args.mode}  |  X-axis: {mode_cfg['x_col']}")
    print(f"Scanning: {results_dir}")
    runs = load_runs(results_dir, mode_cfg)
    if not runs:
        print("No runs found — nothing to plot.")
        return

    metric_cols = mode_cfg["metric_cols"]
    print(f"\nDrawing {len(metric_cols)} individual plots …")
    for metric in metric_cols:
        plot_metric(metric, runs, mode_cfg, args, out_dir)

    print("\nDrawing summary grid …")
    plot_summary_grid(runs, mode_cfg, args, out_dir)

    print(f"\nDone — all plots written to: {out_dir}\n")


if __name__ == "__main__":
    main()
