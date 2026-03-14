#!/usr/bin/env python3
"""
viz_per_stage_metrics.py

Plot all evaluation metrics vs stage from a metrics_table.csv file.
Creates one image with multiple subplots (one subplot per metric).

Example:
  python viz_per_stage_metrics.py \
    --csv 3d_layout_generation/MiDiffusion/output/full_predicted_results/accessibility_universal_only/metrics_table.csv \
    --out 3d_layout_generation/MiDiffusion/output/full_predicted_results/accessibility_universal_only/per_stage_metrics.png
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize metric evolution across stages in one multi-subplot image."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to metrics_table.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="per_stage_metrics.png",
        help="Output image path",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Optional metric columns to exclude",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output figure DPI",
    )
    return parser


def main():
    args = build_parser().parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "stage" not in df.columns:
        raise ValueError("CSV must contain a 'stage' column.")

    # Stage should be numeric and sorted
    df["stage"] = pd.to_numeric(df["stage"], errors="coerce")
    df = df.dropna(subset=["stage"]).copy()
    df["stage"] = df["stage"].astype(int)
    df = df.sort_values("stage").reset_index(drop=True)

    # Candidate metric columns: everything except stage
    metric_cols = [c for c in df.columns if c != "stage" and c not in set(args.exclude)]

    # Convert to numeric where possible
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only metrics that have at least one numeric value
    metric_cols = [c for c in metric_cols if df[c].notna().any()]
    if not metric_cols:
        raise ValueError("No numeric metric columns found to plot.")

    # Dynamic subplot grid
    n = len(metric_cols)
    ncols = 3 if n >= 3 else n
    nrows = int(math.ceil(n / ncols))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3.8 * nrows))
    axes = np.array(axes).reshape(-1)

    x = df["stage"].values

    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        y = df[metric].values
        valid = ~np.isnan(y)

        if valid.sum() == 0:
            ax.set_visible(False)
            continue

        ax.plot(
            x[valid],
            y[valid],
            marker="o",
            markersize=3.2,
            linewidth=1.8,
            color="#1f77b4",
            alpha=0.95,
        )

        # Mark first and last valid points for quick trend reading
        first_idx = np.where(valid)[0][0]
        last_idx = np.where(valid)[0][-1]
        ax.scatter(x[first_idx], y[first_idx], color="#2ca02c", s=24, label="start", zorder=3)
        ax.scatter(x[last_idx], y[last_idx], color="#d62728", s=24, label="end", zorder=3)

        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("stage")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Only show legend on first subplot to reduce clutter
        if i == 0:
            ax.legend(frameon=True, fontsize=8, loc="best")

    # Hide any unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Evaluation Metrics vs Stage", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()