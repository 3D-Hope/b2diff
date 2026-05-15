"""
Plot Proposition 1 stage curves from proposal1_results.json.

Creates matching chart sets for latent L2 and image L2:
1. Absolute mean curves with std bands when available.
2. Centered delta charts to make small gaps visible.
3. Zoomed mean-only charts for close inspection.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/proposal1_analysis/proposal1_results.json")
DEFAULT_SAMPLE_INPUT = Path("outputs/proposal1_analysis/proposition1_samples.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/proposal1_analysis")
TARGET_CENTER = 118.0
IMAGE_CENTER = 234.0
STAGE_MIN = 0
STAGE_MAX = 5
Y_AXIS_MIN = 105.0
Y_AXIS_MAX = 135.0
ZOOM_Y_MIN = 115.0
ZOOM_Y_MAX = 120.0
IMAGE_Y_MIN = 230.0
IMAGE_Y_MAX = 238.0
IMAGE_ZOOM_Y_MIN = 232.5
IMAGE_ZOOM_Y_MAX = 236.5

RUNS = {
    "late_10": {
        "label": "late strategy",
        "color": "#ff69b4",
    },
    "uniform_10": {
        "label": "uniform strategy",
        "color": "#1f77b4",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot stage-wise latent-L2 results for Proposition 1")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to proposal1_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write plots and CSV",
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=DEFAULT_SAMPLE_INPUT,
        help="Optional per-sample CSV for more accurate metric bands",
    )
    parser.add_argument(
        "--center",
        type=float,
        default=TARGET_CENTER,
        help="Value to subtract in the delta view to make small differences visible",
    )
    return parser.parse_args()


def load_metric_rows_from_samples(samples_path: Path, metric_name: str):
    if not samples_path.exists():
        return []

    sample_frame = pd.read_csv(samples_path)
    sample_frame = sample_frame[sample_frame["run_name"].isin(RUNS.keys())]
    sample_frame = sample_frame[(sample_frame["stage"] >= STAGE_MIN) & (sample_frame["stage"] <= STAGE_MAX)]

    rows = []
    for (run_name, stage), group in sample_frame.groupby(["run_name", "stage"], sort=True):
        values = pd.to_numeric(group[metric_name], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "run_name": run_name,
                "stage": int(stage),
                "label": RUNS[run_name]["label"],
                "color": RUNS[run_name]["color"],
                f"{metric_name}_mean": float(values.mean()),
                f"{metric_name}_std": float(values.std()),
                f"{metric_name}_count": int(values.size),
            }
        )

    return rows


def load_metric_stage_rows(results_path: Path, samples_path: Path, metric_name: str):
    with open(results_path, "r") as f:
        payload = json.load(f)

    sample_rows = load_metric_rows_from_samples(samples_path, metric_name)
    if sample_rows:
        frame = pd.DataFrame(sample_rows)
        return payload, frame.sort_values(["run_name", "stage"]).reset_index(drop=True)

    rows = []
    for method_name, stats in payload["methods"].items():
        if "_stage" not in method_name:
            continue

        run_name, stage_text = method_name.rsplit("_stage", 1)
        if run_name not in RUNS:
            continue

        mean_key = f"{metric_name}_mean"
        std_key = f"{metric_name}_std"
        mean_value = float(stats[mean_key])
        std_value = float(stats[std_key]) if np.isfinite(stats[std_key]) else np.nan

        rows.append(
            {
                "run_name": run_name,
                "stage": int(stage_text),
                "label": RUNS[run_name]["label"],
                "color": RUNS[run_name]["color"],
                f"{metric_name}_mean": mean_value,
                f"{metric_name}_std": std_value,
                f"{metric_name}_count": None,
            }
        )

    if not rows:
        raise RuntimeError("No staged late_10/uniform_10 rows found in proposal1_results.json")

    frame = pd.DataFrame(rows)
    frame = frame[(frame["stage"] >= STAGE_MIN) & (frame["stage"] <= STAGE_MAX)]
    return payload, frame.sort_values(["run_name", "stage"]).reset_index(drop=True)


def add_line_with_band(ax, frame: pd.DataFrame, mean_col: str, std_col: str, label: str, color: str):
    x = frame["stage"].to_numpy()
    y = frame[mean_col].to_numpy()
    y_std = frame[std_col].to_numpy() if std_col in frame.columns else np.array([])

    ax.plot(x, y, marker="o", linewidth=2.5, color=color, label=label)
    finite = np.isfinite(y) & np.isfinite(y_std)
    if finite.any():
        ax.fill_between(x[finite], y[finite] - y_std[finite], y[finite] + y_std[finite], color=color, alpha=0.18, linewidth=0)


def plot_metric_set(frame: pd.DataFrame, metric_name: str, output_dir: Path, center: float, abs_range: tuple[float, float], zoom_range: tuple[float, float]):
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"
    title_prefix = "Latent L2" if metric_name == "latent_l2" else "Image L2"
    file_prefix = "latent_l2" if metric_name == "latent_l2" else "image_l2"

    # Absolute chart.
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name in ["late_10", "uniform_10"]:
        run_frame = frame[frame["run_name"] == run_name]
        if run_frame.empty:
            continue
        add_line_with_band(ax, run_frame, mean_col, std_col, run_frame.iloc[0]["label"], run_frame.iloc[0]["color"])

    ax.set_title(f"Proposition 1: stage-wise {title_prefix}")
    ax.set_xlabel("Stage")
    ax.set_ylabel(f"{title_prefix} norm")
    ax.set_xticks(list(range(STAGE_MIN, STAGE_MAX + 1)))
    ax.set_xlim(STAGE_MIN - 0.25, STAGE_MAX + 0.25)
    ax.set_ylim(abs_range[0], abs_range[1])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    abs_path = output_dir / f"proposition1_{file_prefix}_absolute.png"
    fig.savefig(abs_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Centered chart.
    fig, ax = plt.subplots(figsize=(10, 6))
    centered_values = []
    for run_name in ["late_10", "uniform_10"]:
        run_frame = frame[frame["run_name"] == run_name].copy()
        if run_frame.empty:
            continue
        run_frame["centered_mean"] = run_frame[mean_col] - center
        centered_values.extend(run_frame["centered_mean"].tolist())
        x = run_frame["stage"].to_numpy()
        y = run_frame["centered_mean"].to_numpy()
        y_std = run_frame[std_col].to_numpy() if std_col in run_frame.columns else np.array([])
        color = run_frame.iloc[0]["color"]
        label = run_frame.iloc[0]["label"]
        ax.plot(x, y, marker="o", linewidth=2.5, color=color, label=label)
        finite = np.isfinite(y) & np.isfinite(y_std)
        if finite.any():
            ax.fill_between(x[finite], y[finite] - y_std[finite], y[finite] + y_std[finite], color=color, alpha=0.18, linewidth=0)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title(f"Proposition 1: {title_prefix} deviation from {center:.1f}")
    ax.set_xlabel("Stage")
    ax.set_ylabel(f"{title_prefix} - {center:.1f}")
    ax.set_xticks(list(range(STAGE_MIN, STAGE_MAX + 1)))
    ax.set_xlim(STAGE_MIN - 0.25, STAGE_MAX + 0.25)
    if centered_values:
        pad = max(0.25, (max(centered_values) - min(centered_values)) * 0.15)
        ax.set_ylim(min(centered_values) - pad, max(centered_values) + pad)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    centered_path = output_dir / f"proposition1_{file_prefix}_centered.png"
    fig.savefig(centered_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Zoomed chart without std bands.
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name in ["late_10", "uniform_10"]:
        run_frame = frame[frame["run_name"] == run_name]
        if run_frame.empty:
            continue
        x = run_frame["stage"].to_numpy()
        y = run_frame[mean_col].to_numpy()
        color = run_frame.iloc[0]["color"]
        label = run_frame.iloc[0]["label"]
        ax.plot(x, y, marker="o", linewidth=2.5, color=color, label=label)

    ax.set_title(f"Proposition 1: zoomed {title_prefix} norm without std band")
    ax.set_xlabel("Stage")
    ax.set_ylabel(f"{title_prefix} norm")
    ax.set_xticks(list(range(STAGE_MIN, STAGE_MAX + 1)))
    ax.set_xlim(STAGE_MIN - 0.25, STAGE_MAX + 0.25)
    ax.set_ylim(zoom_range[0], zoom_range[1])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    zoom_path = output_dir / f"proposition1_{file_prefix}_zoom_no_std.png"
    fig.savefig(zoom_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return abs_path, centered_path, zoom_path


def main():
    args = parse_args()
    payload, latent_df = load_metric_stage_rows(args.input, args.samples_csv, "latent_l2")
    _, image_df = load_metric_stage_rows(args.input, args.samples_csv, "image_l2")

    latent_csv_path = args.output_dir / "proposition1_stage_latent_l2.csv"
    image_csv_path = args.output_dir / "proposition1_stage_image_l2.csv"
    latent_df.to_csv(latent_csv_path, index=False)
    image_df.to_csv(image_csv_path, index=False)

    latent_abs_path, latent_centered_path, latent_zoom_path = plot_metric_set(
        latent_df,
        "latent_l2",
        args.output_dir,
        args.center,
        (Y_AXIS_MIN, Y_AXIS_MAX),
        (ZOOM_Y_MIN, ZOOM_Y_MAX),
    )

    image_abs_path, image_centered_path, image_zoom_path = plot_metric_set(
        image_df,
        "image_l2",
        args.output_dir,
        IMAGE_CENTER,
        (IMAGE_Y_MIN, IMAGE_Y_MAX),
        (IMAGE_ZOOM_Y_MIN, IMAGE_ZOOM_Y_MAX),
    )

    # Terminal summary for quick inspection.
    print("\nProposition 1 latent-L2 summary")
    print("=" * 80)
    print(f"Baseline center used for delta view: {args.center:.1f}")
    print(f"Results loaded from: {args.input}")
    print(f"Latent CSV written to: {latent_csv_path}")
    print(f"Image CSV written to: {image_csv_path}")
    print(f"Latent absolute plot written to: {latent_abs_path}")
    print(f"Latent centered plot written to: {latent_centered_path}")
    print(f"Latent zoomed no-std plot written to: {latent_zoom_path}")
    print(f"Image absolute plot written to: {image_abs_path}")
    print(f"Image centered plot written to: {image_centered_path}")
    print(f"Image zoomed no-std plot written to: {image_zoom_path}")
    print(f"Using stages {STAGE_MIN} to {STAGE_MAX} and latent y-axis range {Y_AXIS_MIN:.1f} to {Y_AXIS_MAX:.1f}")

    table_rows = []
    for run_name in ["late_10", "uniform_10"]:
        run_frame = latent_df[latent_df["run_name"] == run_name]
        if run_frame.empty:
            continue
        table_rows.append(
            [
                RUNS[run_name]["label"],
                f"{run_frame['latent_l2_mean'].mean():.4f}",
                f"{run_frame['latent_l2_std'].mean():.4f}",
                f"{(run_frame['latent_l2_mean'] - args.center).mean():.4f}",
            ]
        )

    print(pd.DataFrame(table_rows, columns=["Strategy", "Mean latent L2", "Mean std", f"Mean (value - {args.center:.1f})"]))

    # Also print the stage-by-stage gaps to make the small decimal differences obvious.
    pivot = latent_df.pivot(index="stage", columns="run_name", values="latent_l2_mean").sort_index()
    print("\nStage-wise gap (uniform - late):")
    for stage, row in pivot.iterrows():
        if np.isfinite(row["uniform_10"]) and np.isfinite(row["late_10"]):
            gap = row["uniform_10"] - row["late_10"]
            print(f"  stage{stage}: {gap:+.4f}")

    image_table_rows = []
    for run_name in ["late_10", "uniform_10"]:
        run_frame = image_df[image_df["run_name"] == run_name]
        if run_frame.empty:
            continue
        image_table_rows.append(
            [
                RUNS[run_name]["label"],
                f"{run_frame['image_l2_mean'].mean():.4f}",
                f"{run_frame['image_l2_std'].replace([np.inf, -np.inf], np.nan).mean():.4f}",
                f"{(run_frame['image_l2_mean'] - IMAGE_CENTER).mean():.4f}",
            ]
        )

    print("\nImage-L2 summary")
    print(pd.DataFrame(image_table_rows, columns=["Strategy", "Mean image L2", "Mean std", f"Mean (value - {IMAGE_CENTER:.1f})"]))

    image_pivot = image_df.pivot(index="stage", columns="run_name", values="image_l2_mean").sort_index()
    print("\nStage-wise image gap (uniform - late):")
    for stage, row in image_pivot.iterrows():
        if np.isfinite(row["uniform_10"]) and np.isfinite(row["late_10"]):
            gap = row["uniform_10"] - row["late_10"]
            print(f"  stage{stage}: {gap:+.4f}")


if __name__ == "__main__":
    main()