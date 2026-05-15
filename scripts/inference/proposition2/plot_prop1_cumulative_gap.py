#!/usr/bin/env python
"""
Cumulative-gap bar chart for Proposition 1.

For each training stage K we compute the per-sample paired gap

    g_{i,K}  =  ||x_0^{uniform,K}_i - x_0^{pre}_i||_2
              - ||x_0^{late,K}_i    - x_0^{pre}_i||_2

(positive ⇒ uniform drifts further from pretrained than late does, which is
the direction Prop 1 / Remark 1 predicts because uniform_10 touches early
high-noise timesteps and late_10 does not).

We then cumulate across stages:

    cum_K(i)  =  Σ_{k=1..K}  g_{i,k}

and plot mean ± SEM of cum_K vs K as a bar chart. Each stage's bar is
taller than the previous one whenever the per-stage gap is positive,
making the compounding visible.

Usage:
    python scripts/diagnostics/plot_prop1_cumulative_gap.py \
        --input outputs/proposal1_analysis/proposal1_samples.csv \
        --output rebuttal/figures/prop1_cumulative_gap.pdf \
        --group_a uniform_10 --group_b late_10 \
        --metric latent_l2

`--metric` can also be `image_l2` or `delta_r`.
"""

import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/proposal1_analysis/proposal1_samples.csv")
    ap.add_argument("--output", default="rebuttal/figures/prop1_cumulative_gap.pdf")
    ap.add_argument("--group_a", default="uniform_10",
                    help="Strategy whose drift should be LARGER per Prop 1")
    ap.add_argument("--group_b", default="late_10",
                    help="Strategy whose drift should be SMALLER per Prop 1")
    ap.add_argument("--metric", default="latent_l2",
                    choices=["latent_l2", "image_l2", "delta_r"])
    ap.add_argument("--max_stage", type=int, default=None,
                    help="Truncate to stages <= this (optional)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df[df.run_group.isin([args.group_a, args.group_b])].copy()
    df = df.dropna(subset=["stage"])
    df["stage"] = df["stage"].astype(int)
    if args.max_stage is not None:
        df = df[df.stage <= args.max_stage]

    # Pivot to one row per (sample_idx, stage), one column per run_group
    piv = (df.pivot_table(index=["sample_idx", "stage"],
                          columns="run_group",
                          values=args.metric)
             .reset_index())

    # Per-sample, per-stage paired gap (group_a - group_b)
    piv["gap"] = piv[args.group_a] - piv[args.group_b]

    # Cumulative gap per sample, ordered by stage
    piv = piv.sort_values(["sample_idx", "stage"])
    piv["cum_gap"] = (piv.groupby("sample_idx")["gap"]
                          .cumsum())

    agg = (piv.groupby("stage")["cum_gap"]
              .agg(["mean", "sem", "count"])
              .reset_index())

    # Color bars: green if predicted direction (group_a > group_b, gap > 0),
    # red otherwise.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4),
                                    gridspec_kw={"width_ratios": [1.3, 1.0]})

    # ----- Bar chart: cumulative gap -----
    x = agg.stage.values
    y = agg["mean"].values
    e = agg["sem"].values
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in y]
    ax1.bar(x, y, yerr=e, color=colors, edgecolor="black",
            linewidth=0.4, capsize=2.5, error_kw={"linewidth": 0.8})
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_xlabel("Training stage")
    metric_label = {
        "latent_l2": r"Cumulative drift gap  $\sum_{k\le K} (\Vert\Delta x_0^{\,A}\Vert - \Vert\Delta x_0^{\,B}\Vert)_k$",
        "image_l2":  r"Cumulative image-L2 gap (uniform − late)",
        "delta_r":   r"Cumulative $\Delta R$ gap (uniform − late)",
    }[args.metric]
    ax1.set_ylabel(metric_label, fontsize=10)
    ax1.set_title(f"Cumulative drift gap: {args.group_a} − {args.group_b}", fontsize=11)
    if len(x) > 0:
        last_K, last_mean, last_sem = int(x[-1]), float(y[-1]), float(e[-1])
        ax1.text(0.02, 0.98,
                 f"final (stage {last_K}):\n  {last_mean:+.4f}  ±  {last_sem:.4f}\n"
                 f"  n = {int(agg['count'].iloc[-1])} samples",
                 transform=ax1.transAxes, va="top",
                 fontsize=8.5, family="monospace",
                 bbox=dict(facecolor="white", edgecolor="#bbb",
                           alpha=0.9, boxstyle="round,pad=0.3"))
    ax1.grid(axis="y", alpha=0.25)

    # ----- Line chart: per-stage gap (the increments) -----
    gap_agg = (piv.groupby("stage")["gap"]
                  .agg(["mean", "sem"]).reset_index())
    gx = gap_agg.stage.values
    gy = gap_agg["mean"].values
    ge = gap_agg["sem"].values
    ax2.errorbar(gx, gy, yerr=ge, fmt="o-", color="#1f77b4",
                  capsize=2.5, linewidth=1.4, markersize=4)
    ax2.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Training stage")
    ax2.set_ylabel(f"Per-stage gap ({args.group_a} − {args.group_b})", fontsize=10)
    ax2.set_title("Per-stage increment (positive ⇒ Prop 1 direction)", fontsize=11)
    ax2.grid(alpha=0.25)

    fig.suptitle(f"Proposition 1: generative reach gap accumulates across stages "
                 f"(metric = {args.metric})", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.replace(".pdf", ".png"), dpi=160)
    print(f"wrote {args.output}")
    print(f"wrote {args.output.replace('.pdf', '.png')}")

    # Text summary
    print("\nPer-stage cumulative gap (mean ± SEM, paired across samples):")
    for _, row in agg.iterrows():
        print(f"  stage {int(row.stage):>2}:   cum = {row['mean']:+10.4f}  ± {row['sem']:.4f}   (n={int(row['count'])})")


if __name__ == "__main__":
    main()
