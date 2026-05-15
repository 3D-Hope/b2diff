#!/usr/bin/env python
"""Cumulative log|det J| vs step idx — show how per-step gaps compound."""

import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="rebuttal/artifacts/volume_preservation/run2_stage27")
    ap.add_argument("--output", default="rebuttal/figures/volume_preservation_cumulative.pdf")
    args = ap.parse_args()

    parts = [pd.read_csv(os.path.join(args.input_dir, f"divergence_{m}.csv"))
             for m in ["ref", "iadd", "full"]]
    d = pd.concat(parts, ignore_index=True)

    # Per-trajectory cumulative sum across step_idx
    d = d.sort_values(["model", "prompt_id", "seed", "step_idx"])
    d["cum_log_det"] = (d.groupby(["model", "prompt_id", "seed"])
                          .log_det_step.cumsum())

    # Aggregate across 24 trajectories per (model, step)
    agg = (d.groupby(["model", "step_idx"])
             .cum_log_det.agg(["mean", "sem", "count"])
             .reset_index())

    colors = {"ref": "#444", "iadd": "#1f77b4", "full": "#d62728"}
    labels = {"ref": "ref  (SD v1.4, no RL)",
              "iadd": "sparse (iadd, stage 27)",
              "full": "dense (full DDPO, stage 27)"}
    order = ["ref", "iadd", "full"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                    gridspec_kw={"width_ratios": [1.0, 1.0]})

    # --- Panel A: cumulative log|det| vs step idx ---
    for m in order:
        sub = agg[agg.model == m].sort_values("step_idx")
        x = sub.step_idx.values
        y = sub["mean"].values
        e = sub["sem"].values
        ax1.plot(x, y, color=colors[m], marker="o", markersize=3,
                 linewidth=1.7, label=labels[m])
        ax1.fill_between(x, y - e, y + e, color=colors[m], alpha=0.18)

    ax1.set_xlabel("DDIM step idx  (0 = noisiest,  19 = cleanest)")
    ax1.set_ylabel(r"Cumulative  $\log|\det J_{T\to t}|$  (nats)")
    ax1.set_title("How the volume contraction compounds across the trajectory")
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax1.legend(frameon=False, loc="lower left", fontsize=9)
    ax1.grid(alpha=0.2)

    # --- Panel B: paired cumulative GAP (each method minus ref) ---
    # Compute paired diffs trajectory-by-trajectory, then average
    piv = (d.pivot_table(index=["prompt_id", "seed", "step_idx"],
                         columns="model", values="cum_log_det")
             .reset_index())

    for m, color, lab in [("iadd", colors["iadd"], "sparse − ref"),
                          ("full", colors["full"], "dense − ref")]:
        piv[f"{m}_diff"] = piv[m] - piv["ref"]
        gap = (piv.groupby("step_idx")[f"{m}_diff"]
                  .agg(["mean", "sem"]).reset_index())
        x = gap.step_idx.values
        y = gap["mean"].values
        e = gap["sem"].values
        ax2.plot(x, y, color=color, marker="o", markersize=3,
                 linewidth=1.7, label=lab)
        ax2.fill_between(x, y - e, y + e, color=color, alpha=0.18)

    ax2.axhline(0, color="#444", linewidth=0.7, linestyle="--", alpha=0.6,
                label="ref  (zero by definition)")
    ax2.set_xlabel("DDIM step idx")
    ax2.set_ylabel(r"Paired cumulative gap  $\log|\det J|^{\,\mathrm{method}} - \log|\det J|^{\,\mathrm{ref}}$")
    ax2.set_title("Gap vs pretrained — positive = more volume preserved")
    ax2.legend(frameon=False, loc="upper left", fontsize=9)
    ax2.grid(alpha=0.2)

    # Annotate final numbers
    last_step = int(agg.step_idx.max())
    finals = {m: agg[(agg.model == m) & (agg.step_idx == last_step)]["mean"].iloc[0]
              for m in order}
    txt = (f"Final cumulative log|det J_{{T→0}}|:\n"
           f"  sparse:   {finals['iadd']:>10,.0f}\n"
           f"  ref:        {finals['ref']:>10,.0f}\n"
           f"  dense:    {finals['full']:>10,.0f}\n"
           f"  sparse − dense = {finals['iadd']-finals['full']:>+,.0f}")
    ax1.text(0.02, 0.02, txt, transform=ax1.transAxes,
             fontsize=8.5, family="monospace",
             verticalalignment="bottom",
             bbox=dict(facecolor="white", edgecolor="#bbb", alpha=0.9,
                       boxstyle="round,pad=0.3"))

    fig.suptitle("Volume preservation — per-step gaps compound across 20 DDIM steps "
                 "(6 prompts × 4 seeds, K=4 Hutchinson probes, 3-term Taylor)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.replace(".pdf", ".png"), dpi=160)
    print(f"wrote {args.output}")
    print(f"wrote {args.output.replace('.pdf', '.png')}")

    # Text summary
    print("\nFinal cumulative log|det J|:")
    for m in order:
        print(f"  {m:5s} {finals[m]:>12,.2f}")
    print(f"\nFinal-step paired gap (mean ± SEM, n=24):")
    for m in ("iadd", "full"):
        last = piv[piv.step_idx == last_step][f"{m}_diff"].dropna()
        print(f"  {m:5s} - ref :   {last.mean():>+10.2f}  ±{last.sem():>6.2f}")
    diff_im_full = (piv[piv.step_idx == last_step]["iadd"]
                    - piv[piv.step_idx == last_step]["full"]).dropna()
    print(f"  iadd  - full :   {diff_im_full.mean():>+10.2f}  ±{diff_im_full.sem():>6.2f}")


if __name__ == "__main__":
    main()
