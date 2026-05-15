"""
Batch runner for Proposition 1 experiments.

It runs inference for every `stage*/checkpoints/checkpoint_1` checkpoint under
the configured run roots, saves outputs in stage-specific folders, and then
launches the comparison analysis in `proposal1.py`.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SCRIPT = ROOT / "scripts" / "inference" / "inference_lora_for_proposal.py"
ANALYSIS_SCRIPT = ROOT / "scripts" / "inference" / "proposal1.py"

BASE_PRETRAINED = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/baseline_sd14_samples_10steps/pretrained_latents.pt"
RUN_ROOTS = {
    "late_10": ROOT / "outputs" / "last_10",
    "uniform_10": ROOT / "outputs" / "uniform_10",
}
OUTPUT_ROOT = ROOT / "outputs" / "proposal1_batch"


def find_stage_checkpoints(run_root: Path):
    checkpoints = []
    for stage_dir in sorted(run_root.glob("stage*/checkpoints/checkpoint_1")):
        match = re.search(r"stage(\d+)", str(stage_dir))
        if match:
            checkpoints.append((int(match.group(1)), stage_dir))
    return sorted(checkpoints, key=lambda item: item[0])


def run_inference(run_name: str, stage_number: int, checkpoint_dir: Path):
    output_dir = OUTPUT_ROOT / run_name / f"stage{stage_number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["EXPERIMENT_NAME"] = f"{run_name}_stage{stage_number}"
    env["LORA_CHECKPOINT_PATH"] = str(checkpoint_dir)
    env["OUTPUT_DIR"] = str(output_dir)
    env["PRETRAINED_LATENTS_PATH"] = BASE_PRETRAINED

    print("=" * 80)
    print(f"Running {run_name} stage {stage_number}")
    print(f"  checkpoint: {checkpoint_dir}")
    print(f"  output:     {output_dir}")
    print("=" * 80)

    subprocess.run([sys.executable, str(INFERENCE_SCRIPT)], check=True, env=env)
    return output_dir


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for run_name, run_root in RUN_ROOTS.items():
        if not run_root.exists():
            print(f"⚠ Skipping missing run root: {run_root}")
            continue

        stage_checkpoints = find_stage_checkpoints(run_root)
        if not stage_checkpoints:
            print(f"⚠ No stage*/checkpoints/checkpoint_1 found under {run_root}")
            continue

        for stage_number, checkpoint_dir in stage_checkpoints:
            try:
                output_dir = run_inference(run_name, stage_number, checkpoint_dir)
                manifest_rows.append(
                    {
                        "run_name": run_name,
                        "stage": stage_number,
                        "checkpoint_dir": str(checkpoint_dir),
                        "output_dir": str(output_dir),
                        "lora_latents": str(output_dir / "lora_latents.pt"),
                    }
                )
            except subprocess.CalledProcessError as exc:
                manifest_rows.append(
                    {
                        "run_name": run_name,
                        "stage": stage_number,
                        "checkpoint_dir": str(checkpoint_dir),
                        "output_dir": str(OUTPUT_ROOT / run_name / f"stage{stage_number}"),
                        "lora_latents": "",
                        "error": f"exit_code={exc.returncode}",
                    }
                )
                raise

    manifest_path = OUTPUT_ROOT / "batch_manifest.csv"
    fieldnames = ["run_name", "stage", "checkpoint_dir", "output_dir", "lora_latents", "error"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"✓ Batch manifest saved to {manifest_path}")

    print("\nLaunching proposal1 analysis...")
    subprocess.run([sys.executable, str(ANALYSIS_SCRIPT)], check=True)
    print("✓ Analysis complete")


if __name__ == "__main__":
    main()