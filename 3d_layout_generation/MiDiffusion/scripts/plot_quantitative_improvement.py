#!/usr/bin/env python3
"""
Create a bar plot similar to the "Quantitative Improvement Over Baseline" figure.
Data is manually defined in this script for now.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Manual values from the provided table.
    metrics = [
        r"$R_{reach}$",
        r"$R_{walkable}$",
        r"$Col_{obj}$",
        r"$Col_{scene}$",
        r"$R_{out}$",
    ]

    # 52.67% 81.67% 5.89% 85.7% 0.806 1.34 66.00%
    # 41.49% 63.61% 3.12% 92.52% 0.8272 

    # Decimal values converted to percentages: 0.806 -> 80.6, 0.8272 -> 82.72
    baseline_raw = np.array([52.67, 81.67, 5.89, 85.7, 80.6], dtype=float)
    augmented_raw = np.array([41.49, 63.61, 3.12, 92.52, 82.72], dtype=float)

    # Reorder from [Col_obj, Col_scene, R_out, R_reach, R_walkable]
    # to       [R_reach, R_walkable, Col_obj, Col_scene, R_out]
    order = np.array([3, 4, 0, 1, 2])
    baseline = baseline_raw[order]
    augmented = augmented_raw[order]

    x = np.arange(len(metrics))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax_left = plt.subplots(figsize=(11.5, 7.5), dpi=140)
    ax_right = ax_left.twinx()

    physical_idx = np.array([2, 3, 4])
    functional_idx = np.array([0, 1])

    baseline_left = np.full_like(baseline, np.nan)
    augmented_left = np.full_like(augmented, np.nan)
    baseline_right = np.full_like(baseline, np.nan)
    augmented_right = np.full_like(augmented, np.nan)

    # Left axis shows functional utility; right axis shows physical plausibility.
    baseline_left[functional_idx] = baseline[functional_idx]
    augmented_left[functional_idx] = augmented[functional_idx]
    baseline_right[physical_idx] = baseline[physical_idx]
    augmented_right[physical_idx] = augmented[physical_idx]

    bars_baseline = ax_left.bar(
        x - width / 2,
        baseline_left,
        width,
        label="MiDiffusion (3D Front)",
        color="#6f90c7",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    bars_augmented = ax_left.bar(
        x + width / 2,
        augmented_left,
        width,
        label="MiDiffusion (3D Front + iARCS aug.)",
        color="#79b081",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax_right.bar(
        x - width / 2,
        baseline_right,
        width,
        color="#6f90c7",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax_right.bar(
        x + width / 2,
        augmented_right,
        width,
        color="#79b081",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax_left.set_ylabel(r"Functional utility ($\uparrow$)", fontsize=17, fontweight="bold")
    ax_right.set_ylabel(r"Physical plausibility ($\downarrow$)", fontsize=17, fontweight="bold")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(metrics, fontsize=16, fontweight="bold")
    ax_left.set_ylim(0, 100)
    ax_right.set_ylim(0, 90)
    ax_left.tick_params(axis="y", labelsize=14)
    ax_right.tick_params(axis="y", labelsize=14)
    ax_left.grid(axis="y", linestyle="-", alpha=0.35)
    ax_right.grid(False)
    ax_left.axvline(1.5, color="#666", linestyle="--", linewidth=1.3, alpha=0.9, zorder=2)

    legend = ax_left.legend([bars_baseline, bars_augmented], ["MiDiffusion (3D Front)", "MiDiffusion (3D Front + iARCS aug.)"], loc="upper right", fontsize=13, frameon=True)
    legend.get_frame().set_alpha(0.95)

    # Compact inset with overlapping normal distributions (same mean FID).
    inset = ax_left.inset_axes([0.39, 0.82, 0.22, 0.16])
    inset.set_facecolor("#e8eef5")
    for spine in inset.spines.values():
        spine.set_color("#7f8fa3")
        spine.set_linewidth(1.0)

    mu = 1.34
    sigma = 0.18
    x_fid = np.linspace(0.75, 1.95, 300)
    y_base = np.exp(-0.5 * ((x_fid - mu) / sigma) ** 2)
    y_aug = np.exp(-0.5 * ((x_fid - mu) / sigma) ** 2)

    inset.plot(x_fid, y_base, color="#6f90c7", lw=2.8, linestyle="-", alpha=0.95, zorder=3)
    inset.plot(
        x_fid,
        y_aug,
        color="#79b081",
        lw=2.4,
        linestyle=(0, (4, 2)),
        marker="o",
        markevery=26,
        markersize=2.6,
        markerfacecolor="white",
        markeredgewidth=0.8,
        alpha=1.0,
        zorder=4,
    )
    inset.fill_between(x_fid, 0, y_base, color="#6f90c7", alpha=0.08, zorder=1)
    inset.fill_between(x_fid, 0, y_aug, color="#79b081", alpha=0.06, zorder=2)
    inset.axvline(mu, color="#333", linestyle=":", lw=1.2)

    inset.text(0.5, 0.97, "Same Quality", fontsize=12.0, fontweight="bold",
               ha="center", va="top", transform=inset.transAxes)
    inset.text(0.5, 0.84, r"FID $= 1.34$", fontsize=11.5, fontweight="bold",
               ha="center", va="top", transform=inset.transAxes)

    inset.set_xlim(0.60, 2.10)  # extra horizontal padding
    inset.set_ylim(-0.08, 1.30)  # extra vertical padding below and above curve
    inset.set_xticks([])
    inset.set_yticks([])

    fig.tight_layout()

    out_png = Path("output/quantitative_improvement_barplot.png")
    out_pdf = Path("output/quantitative_improvement_barplot.pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved plot to: {out_png}")
    print(f"Saved plot to: {out_pdf}")

    plt.show()


if __name__ == "__main__":
    main()
