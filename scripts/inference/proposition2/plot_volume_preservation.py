#!/usr/bin/env python
"""
Make the two-panel volume-preservation figure from divergence_{model}.csv files.

  (a) div(eps_tilde_t) vs t, one curve per model, S shaded.
  (b) Cumulative sum_t delta_t * div_mean vs t (= empirical log|det J_{T->0}|).

Usage:
    python scripts/diagnostics/plot_volume_preservation.py \
        --input_dir rebuttal/artifacts/volume_preservation/run1 \
        --output rebuttal/figures/volume_preservation.pdf
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_all(input_dir):
    rows = []
    for name in ["ref", "iadd", "full"]:
        p = os.path.join(input_dir, f"divergence_{name}.csv")
        if not os.path.exists(p):
            print(f"[warn] missing {p}")
            continue
        df = pd.read_csv(p)
        rows.append(df)
    if not rows:
        raise SystemExit("no input files")
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="rebuttal/artifacts/volume_preservation/run1")
    ap.add_argument("--output", default="rebuttal/figures/volume_preservation.pdf")
    args = ap.parse_args()

    cfg_path = os.path.join(args.input_dir, "run_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    active_S = set(cfg.get("active_S", []))
    T = cfg.get("num_inference_steps", 20)

    df = load_all(args.input_dir)

    # Mean across (prompt, seed) per (model, step_idx)
    agg = (df.groupby(["model", "step_idx"])
             .agg(div_mean=("div_mean", "mean"),
                  div_sem=("div_mean", "sem"),
                  delta_t=("delta_t", "first"))
             .reset_index())

    colors = {"ref": "#444", "iadd": "#1f77b4", "full": "#d62728"}
    order = ["ref", "iadd", "full"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Panel (a) ----
    for m in order:
        sub = agg[agg.model == m].sort_values("step_idx")
        if sub.empty:
            continue
        x = sub.step_idx.values
        y = sub.div_mean.values
        e = sub.div_sem.values
        ax1.plot(x, y, color=colors[m], label=m, marker="o", markersize=3, linewidth=1.5)
        ax1.fill_between(x, y - e, y + e, color=colors[m], alpha=0.15)

    if active_S:
        for s in active_S:
            ax1.axvline(s, color="gray", alpha=0.18, linewidth=0.8)
    ax1.set_xlabel("DDIM step index  (t)")
    ax1.set_ylabel(r"$\mathrm{div}(\tilde{\epsilon}_t)$  (Hutchinson, mean over prompts/seeds)")
    ax1.set_title(r"(a) Per-step divergence")
    ax1.legend(frameon=False)

    # ---- Panel (b): cumulative log|det J| ----
    for m in order:
        sub = agg[agg.model == m].sort_values("step_idx")
        if sub.empty:
            continue
        cum = np.cumsum(sub.delta_t.values * sub.div_mean.values)
        ax2.plot(sub.step_idx.values, cum, color=colors[m], label=m,
                 marker="o", markersize=3, linewidth=1.5)
    if active_S:
        for s in active_S:
            ax2.axvline(s, color="gray", alpha=0.18, linewidth=0.8)
    ax2.set_xlabel("DDIM step index  (t)")
    ax2.set_ylabel(r"$\sum_{s \leq t} \Delta s \cdot \mathrm{div}(\tilde{\epsilon}_s)$")
    ax2.set_title(r"(b) Cumulative log$|\det J_{T\to t}|$")
    ax2.legend(frameon=False)

    fig.suptitle(f"Volume preservation along DDIM trajectory  "
                 f"(T={T}, |S|={len(active_S)})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")

    # Also dump terminal cumulative values
    print("\nFinal cumulative log|det J_{T->0}|  (mean over prompts/seeds):")
    for m in order:
        sub = agg[agg.model == m].sort_values("step_idx")
        if sub.empty:
            continue
        cum = float(np.sum(sub.delta_t.values * sub.div_mean.values))
        # split on / off S
        in_S = sub.step_idx.isin(active_S)
        cum_in = float(np.sum((sub.delta_t.values * sub.div_mean.values)[in_S.values]))
        cum_out = cum - cum_in
        print(f"  {m:5s}  total = {cum: .4f}   on-S = {cum_in: .4f}   off-S = {cum_out: .4f}")


if __name__ == "__main__":
    main()
