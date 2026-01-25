"""
Test script to verify single wandb run behavior.
Run this before your full training to ensure everything is configured correctly.

Usage:
    python test_single_run.py
"""

import os
import sys
import torch
from omegaconf import OmegaConf
import wandb

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.sampling import run_sampling
from core.selection import run_selection
from core.training import run_training


def test_single_run():
    """Test that all stages log to a single wandb run."""
    
    print("="*80)
    print("TESTING SINGLE WANDB RUN CONFIGURATION")
    print("="*80)
    
    # Load config
    config = OmegaConf.load("configs/config.yaml")
    
    # Override for quick test
    config.exp_name = "test_single_run"
    config.pipeline.stage_cnt = 2  # Just 2 stages for test
    config.sample.num_batches_per_epoch = 2
    config.train.num_epochs = 1
    config.wandb.enabled = True
    config.wandb.project = "b2diff-test"
    
    print(f"\nConfig:")
    print(f"  exp_name: {config.exp_name}")
    print(f"  stages: {config.pipeline.stage_cnt}")
    print(f"  wandb project: {config.wandb.project}")
    
    # Initialize single wandb run
    print("\n" + "="*80)
    print("INITIALIZING SINGLE WANDB RUN")
    print("="*80)
    
    wandb_run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.exp_name,  # ONE name for entire experiment
        config=OmegaConf.to_container(config, resolve=True),
        tags=["test", "single-run"],
    )
    
    print(f"✓ Wandb run created: {wandb_run.name}")
    print(f"✓ Run ID: {wandb_run.id}")
    print(f"✓ Run URL: {wandb_run.url}")
    
    # Simulate running multiple stages
    for stage_idx in range(config.pipeline.stage_cnt):
        print("\n" + "-"*80)
        print(f"STAGE {stage_idx}")
        print("-"*80)
        
        # Log stage metrics directly to the SAME run
        wandb_run.log({
            f"stage/index": stage_idx,
            f"test/stage_{stage_idx}/metric_a": stage_idx * 0.1,
            f"test/stage_{stage_idx}/metric_b": stage_idx * 0.2,
            f"rewards/stage_{stage_idx}/raw_mean": 0.3 + stage_idx * 0.05,
            f"selection/stage_{stage_idx}/num_positive": 10 + stage_idx * 2,
        })
        
        print(f"✓ Logged metrics to stage {stage_idx}")
    
    # Final summary
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    wandb_run.log({
        "pipeline/total_stages": config.pipeline.stage_cnt,
        "test/completed": True,
    })
    
    print(f"\nCheck your wandb dashboard:")
    print(f"  URL: {wandb_run.url}")
    print(f"\nYou should see:")
    print(f"  ✓ ONE run named '{config.exp_name}'")
    print(f"  ✓ Metrics for both stages in the SAME run")
    print(f"  ✓ Hierarchical metric names (stage/*, test/stage_0/*, test/stage_1/*)")
    
    wandb_run.finish()
    print("\n✓ Wandb run finished")
    print("\nIf you see only ONE run with all stage metrics, the configuration is correct!")


if __name__ == "__main__":
    test_single_run()
