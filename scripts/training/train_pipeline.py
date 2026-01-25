"""
Main training pipeline orchestrator for B2Diff.
This replaces the bash script training loop with a Python implementation using Hydra.

Features:
- Hydra-based configuration management
- Proper wandb logging at pipeline level
- Debugger-friendly structure
- Exception handling and progress tracking
- Calls sample/select/train as functions instead of subprocess calls

Usage:
    python train_pipeline.py
    python train_pipeline.py pipeline.continue_from_stage=10
    python train_pipeline.py --config-name config exp_name=my_experiment
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Add project root to path (2 levels up: scripts/training/ -> scripts/ -> root/)
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.insert(0, project_root)

# Import core modules
from core.sampling import run_sampling
from core.selection import run_selection
from core.training import run_training

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Main training pipeline orchestrator."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize the training pipeline.
        
        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.start_time = time.time()
        self.stage_times = []
        self.wandb_run = None
        self.pipeline = None  # Will hold the model pipeline
        self.trainable_layers = None  # Will hold trainable parameters
        # Note: Accelerator is NOT created here - will be created per-stage in training.py
        # because gradient_accumulation_steps depends on split_step which varies by stage
        
        # Track metrics for plotting CLIP score vs reward queries (paper figure)
        self.metrics_history = []  # Store metrics from each stage
        
        # Setup directories
        self.setup_directories()
        
        # Initialize wandb at pipeline level (single run for all stages)
        if config.wandb.enabled:
            self.init_wandb()
        
        # Load model ONCE for all stages
        self.load_model()
        
        # If resuming from a stage > 0, load the checkpoint from previous stage
        if config.pipeline.continue_from_stage > 0:
            self.load_resume_checkpoint(config.pipeline.continue_from_stage)
    
    def setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.save_path, self.config.exp_name), exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info(f"Save path: {self.config.save_path}")
        logger.info(f"Experiment name: {self.config.exp_name}")
    
    def init_wandb(self):
        """Initialize wandb for pipeline-level tracking (single run for all stages)."""
        self.wandb_run = wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=f"{self.config.exp_name}",
            config=OmegaConf.to_container(self.config, resolve=True),
            tags=["pipeline", self.config.exp_name],
            reinit=False,  # Use same run throughout
        )
        logger.info("Wandb initialized for pipeline tracking (single run for all stages)")
    
    def load_model(self):
        """Load model once at the start - reused for all stages."""
        import torch
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        from diffusers.loaders import AttnProcsLayers
        from diffusers.models.attention_processor import LoRAAttnProcessor
        
        logger.info("Loading model (ONCE for all stages)...")
        torch.cuda.set_device(self.config.dev_id)
        
        # Load pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.pretrained.model,
            torch_dtype=torch.float16
        )
        
        # Freeze base models
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.unet.requires_grad_(not self.config.use_lora)
        self.pipeline.safety_checker = None
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        
        # Setup inference dtype
        inference_dtype = torch.float32
        if self.config.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.config.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        
        # Move to device
        device = torch.device(f"cuda:{self.config.dev_id}")
        self.pipeline.vae.to(device, dtype=inference_dtype)
        self.pipeline.text_encoder.to(device, dtype=inference_dtype)
        
        if self.config.use_lora:
            self.pipeline.unet.to(device, dtype=inference_dtype)
            
            # Setup LoRA layers
            lora_attn_procs = {}
            for name in self.pipeline.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else self.pipeline.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.pipeline.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.pipeline.unet.config.block_out_channels[block_id]
                
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim
                )
            
            self.pipeline.unet.set_attn_processor(lora_attn_procs)
            self.trainable_layers = AttnProcsLayers(self.pipeline.unet.attn_processors)
        else:
            self.trainable_layers = self.pipeline.unet
        
        logger.info("✓ Model loaded and ready for all stages")
    
    def load_resume_checkpoint(self, resume_stage_idx: int):
        """Load checkpoint from previous stage when resuming.
        
        Args:
            resume_stage_idx: The stage we want to resume from
        """
        prev_stage = resume_stage_idx - 1
        checkpoint_num = self.config.train.num_epochs // self.config.train.save_interval
        checkpoint_path = os.path.join(
            self.config.save_path,
            self.config.exp_name,
            f"stage{prev_stage}",
            "checkpoints",
            f"checkpoint_{checkpoint_num}"
        )
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Resume checkpoint not found: {checkpoint_path}")
            logger.warning(f"Starting from scratch instead of resuming from stage {resume_stage_idx}")
            return
        
        logger.info(f"Loading checkpoint from stage {prev_stage}: {checkpoint_path}")
        
        if self.config.use_lora:
            # Load LoRA weights following the original pattern
            from diffusers import UNet2DConditionModel
            from diffusers.loaders import AttnProcsLayers
            
            tmp_unet = UNet2DConditionModel.from_pretrained(
                self.config.pretrained.model,
                revision=self.config.pretrained.revision,
                subfolder="unet"
            )
            tmp_unet.load_attn_procs(checkpoint_path)
            self.trainable_layers.load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
            logger.info(f"✓ Resumed LoRA weights from stage {prev_stage}")
        else:
            # Load full UNet weights
            from diffusers import UNet2DConditionModel
            loaded_unet = UNet2DConditionModel.from_pretrained(
                checkpoint_path,
                subfolder="unet"
            )
            self.pipeline.unet.register_to_config(**loaded_unet.config)
            self.pipeline.unet.load_state_dict(loaded_unet.state_dict())
            del loaded_unet
            logger.info(f"✓ Resumed UNet weights from stage {prev_stage}")
    
    def calculate_split_step(self, stage_idx: int) -> int:
        """
        Calculate the split step for a given stage.
        
        Args:
            stage_idx: Current stage index
            
        Returns:
            cur_split_step: Calculated split step value
        """
        interval = self.config.pipeline.split_step_right - self.config.pipeline.split_step_left + 1
        level = (stage_idx * interval) // self.config.pipeline.stage_cnt
        cur_split_step = level + self.config.pipeline.split_step_left
        return cur_split_step
    
    def create_stage_config(self, stage_idx: int) -> DictConfig:
        """
        Create configuration for a specific stage.
        
        Args:
            stage_idx: Current stage index
            
        Returns:
            stage_config: Configuration for this stage
        """
        # Create a copy of the config
        stage_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        
        # Update stage-specific values (NOT run_name - we use one run for all)
        stage_config.split_step = self.calculate_split_step(stage_idx)
        stage_config.seed = self.config.seed + stage_idx
        
        return stage_config
    
    def run_stage(self, stage_idx: int):
        """
        Run a complete stage (sample -> select -> train).
        
        Args:
            stage_idx: Current stage index
        """
        stage_start_time = time.time()
        
        logger.info("=" * 80)
        logger.info(f"STAGE {stage_idx} / {self.config.pipeline.stage_cnt - 1}")
        logger.info("=" * 80)
        
        # Create stage-specific config
        stage_config = self.create_stage_config(stage_idx)
        
        # Log stage info
        logger.info(f"Split step: {stage_config.split_step}")
        logger.info(f"Seed: {stage_config.seed}")
        
        try:
            # Step 1: Sampling
            logger.info(f"[{stage_idx}] Running sampling...")
            save_dir = run_sampling(
                stage_config, stage_idx, logger, 
                wandb_run=self.wandb_run,
                pipeline=self.pipeline,
                trainable_layers=self.trainable_layers
            )
            logger.info(f"[{stage_idx}] Sampling completed")
            
            # Step 2: Selection
            logger.info(f"[{stage_idx}] Running selection...")
            save_dir, metrics = run_selection(
                stage_config, stage_idx, logger, 
                wandb_run=self.wandb_run
            )
            logger.info(f"[{stage_idx}] Selection completed")
            
            # Store metrics for this stage (for plotting)
            stage_metrics = {
                'stage': stage_idx,
                'mean_reward': metrics.get('rewards/mean_reward', 0.0),
                'std_reward': metrics.get('rewards/std_reward', 0.0),
                'cumulative_queries': metrics.get('rewards/cumulative_queries', 0),
                'num_queries': metrics.get('rewards/num_queries', 0),
            }
            self.metrics_history.append(stage_metrics)
            
            # Step 3: Training
            logger.info(f"[{stage_idx}] Running training...")
            save_dir = run_training(
                stage_config, stage_idx, logger, 
                wandb_run=self.wandb_run,
                pipeline=self.pipeline,
                trainable_layers=self.trainable_layers
            )
            logger.info(f"[{stage_idx}] Training completed")
            
            # Calculate stage time
            stage_time = time.time() - stage_start_time
            self.stage_times.append(stage_time)
            
            # Log stage metrics to wandb (including selection metrics)
            if self.config.wandb.enabled and self.wandb_run:
                log_dict = {
                    "stage/index": stage_idx,
                    "stage/time_seconds": stage_time,
                    "stage/time_minutes": stage_time / 60,
                    "stage/split_step": stage_config.split_step,
                    "pipeline/elapsed_hours": (time.time() - self.start_time) / 3600,
                }
                # Add selection metrics if available
                if metrics:
                    log_dict.update(metrics)
                self.wandb_run.log(log_dict)
            
            logger.info(f"[{stage_idx}] Stage completed in {stage_time:.2f}s ({stage_time/60:.2f}m)")
            
            # Generate incremental plot to show live progress in wandb
            # Update every 5 stages to avoid too much overhead, or always update if <20 stages
            if (self.config.wandb.enabled and self.wandb_run and 
                (stage_idx % 5 == 0 or stage_idx < 20 or stage_idx == self.config.pipeline.stage_cnt - 1)):
                logger.info(f"[{stage_idx}] Generating incremental plot...")
                self.plot_clip_vs_queries_incremental(stage_idx)
            
            # Sleep between stages
            if stage_idx < self.config.pipeline.stage_cnt - 1:
                time.sleep(self.config.pipeline.sleep_between_stages)
                
        except Exception as e:
            logger.error(f"Error in stage {stage_idx}: {str(e)}", exc_info=True)
            
            if self.config.wandb.enabled and self.wandb_run:
                self.wandb_run.log({
                    "stage/index": stage_idx,
                    "stage/error": str(e),
                })
            
            raise
    
    def run(self):
        """Run the complete training pipeline."""
        logger.info("Starting B2Diff Training Pipeline")
        logger.info(f"Experiment: {self.config.exp_name}")
        logger.info(f"Total stages: {self.config.pipeline.stage_cnt}")
        logger.info(f"Continue from stage: {self.config.pipeline.continue_from_stage}")
        if self.config.pipeline.continue_from_stage > 0:
            logger.info(f"✓ Resumed from stage {self.config.pipeline.continue_from_stage - 1} checkpoint")
        logger.info(f"Split step range: [{self.config.pipeline.split_step_left}, {self.config.pipeline.split_step_right}]")
        
        # Run stages
        for stage_idx in range(self.config.pipeline.continue_from_stage, self.config.pipeline.stage_cnt):
            self.run_stage(stage_idx)
        
        # Calculate final statistics
        total_time = time.time() - self.start_time
        total_hours = total_time / 3600
        avg_stage_time = sum(self.stage_times) / len(self.stage_times) if self.stage_times else 0
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time:.2f}s ({total_hours:.4f} hours)")
        logger.info(f"Average stage time: {avg_stage_time:.2f}s ({avg_stage_time/60:.2f}m)")
        logger.info(f"Stages completed: {len(self.stage_times)}")
        
        # Save metrics and generate plot (reproducing paper's figure)
        logger.info("Saving metrics and generating plots...")
        self.save_metrics_history()
        self.plot_clip_vs_queries()
        
        # Save timing summary
        self.save_timing_summary(total_time, total_hours)
        
        # Log final metrics to wandb
        if self.config.wandb.enabled and self.wandb_run:
            self.wandb_run.log({
                "pipeline/total_time_seconds": total_time,
                "pipeline/total_time_hours": total_hours,
                "pipeline/avg_stage_time_seconds": avg_stage_time,
                "pipeline/stages_completed": len(self.stage_times),
            })
            self.wandb_run.finish()
    
    def save_metrics_history(self):
        """Save metrics history to JSON file for later analysis."""
        import json
        
        metrics_file = os.path.join("logs", f"{self.config.exp_name}_metrics_history.json")
        
        with open(metrics_file, 'w') as f:
            json.dump({
                'exp_name': self.config.exp_name,
                'stages': self.metrics_history
            }, f, indent=2)
        
        logger.info(f"Metrics history saved to {metrics_file}")
        return metrics_file
    
    def plot_clip_vs_queries(self):
        """Generate CLIP score vs reward queries plot (reproducing paper's figure)."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("Matplotlib not available - skipping plot generation")
            return None
        
        if len(self.metrics_history) == 0:
            logger.warning("No metrics to plot")
            return None
        
        # Extract data
        stages = [m['stage'] for m in self.metrics_history]
        mean_rewards = [m['mean_reward'] for m in self.metrics_history]
        std_rewards = [m['std_reward'] for m in self.metrics_history]
        cumulative_queries = [m['cumulative_queries'] for m in self.metrics_history]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to numpy for easier manipulation
        queries = np.array(cumulative_queries)
        rewards = np.array(mean_rewards)
        stds = np.array(std_rewards)
        
        # Plot mean with confidence band (matching paper's style)
        ax.plot(queries, rewards, label='Ours (B²-DiffuRL)', 
                color='#ff7f0e', linewidth=2, marker='o', markersize=3)
        ax.fill_between(queries, rewards - stds, rewards + stds, 
                        alpha=0.2, color='#ff7f0e')
        
        ax.set_xlabel('Reward Queries', fontsize=12)
        ax.set_ylabel('CLIP Scores', fontsize=12)
        ax.set_title(f'CLIP Score vs Reward Queries - {self.config.exp_name}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = os.path.join("logs", f"{self.config.exp_name}_clip_vs_queries.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"CLIP vs queries plot saved to {plot_file}")
        
        # Log to wandb if enabled
        if self.wandb_run:
            self.wandb_run.log({"final_plot/clip_vs_queries": wandb.Image(plot_file)})
        
        return plot_file
    
    def plot_clip_vs_queries_incremental(self, stage_idx: int):
        """Generate incremental plot during training to show progress in real-time.
        
        This creates a plot with current data and logs it to wandb so you can
        see the CLIP score curve building up as training progresses.
        
        Args:
            stage_idx: Current stage index
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None
        
        if len(self.metrics_history) == 0:
            return None
        
        # Extract data up to current stage
        stages = [m['stage'] for m in self.metrics_history]
        mean_rewards = [m['mean_reward'] for m in self.metrics_history]
        std_rewards = [m['std_reward'] for m in self.metrics_history]
        cumulative_queries = [m['cumulative_queries'] for m in self.metrics_history]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        queries = np.array(cumulative_queries)
        rewards = np.array(mean_rewards)
        stds = np.array(std_rewards)
        
        # Plot with progress indicator
        ax.plot(queries, rewards, label=f'Ours (Stage {stage_idx}/{self.config.pipeline.stage_cnt-1})', 
                color='#ff7f0e', linewidth=2, marker='o', markersize=3)
        ax.fill_between(queries, rewards - stds, rewards + stds, 
                        alpha=0.2, color='#ff7f0e')
        
        # Add annotation for latest point
        if len(rewards) > 0:
            ax.annotate(f'{rewards[-1]:.4f}', 
                       xy=(queries[-1], rewards[-1]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Reward Queries', fontsize=12)
        ax.set_ylabel('CLIP Scores', fontsize=12)
        ax.set_title(f'CLIP Score vs Reward Queries [Training Progress]', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save incremental plot
        plot_file = os.path.join("logs", f"{self.config.exp_name}_clip_vs_queries_progress.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to wandb with step counter so it shows progression
        if self.wandb_run:
            self.wandb_run.log({
                "progress/clip_vs_queries": wandb.Image(plot_file),
                "progress/current_stage": stage_idx,
                "progress/latest_clip_score": float(rewards[-1]) if len(rewards) > 0 else 0.0,
            })
        
        return plot_file
    
    def save_timing_summary(self, total_time: float, total_hours: float):
        """
        Save timing summary to file.
        
        Args:
            total_time: Total elapsed time in seconds
            total_hours: Total elapsed time in hours
        """
        timing_log = f"logs/{self.config.exp_name}_timing.txt"
        
        with open(timing_log, 'w') as f:
            f.write(f"Experiment: {self.config.exp_name}\n")
            f.write(f"Start time: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total stages: {len(self.stage_times)}\n")
            f.write(f"Wall time: {total_time:.2f}s\n")
            f.write(f"Total hours: {total_hours:.4f}\n")
            f.write(f"Average stage time: {sum(self.stage_times) / len(self.stage_times):.2f}s\n")
            f.write("\nPer-stage times:\n")
            for idx, stage_time in enumerate(self.stage_times):
                stage_num = self.config.pipeline.continue_from_stage + idx
                f.write(f"  Stage {stage_num}: {stage_time:.2f}s ({stage_time/60:.2f}m)\n")
        
        logger.info(f"Timing summary saved to {timing_log}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig):
    """
    Main entry point for the training pipeline.
    
    Args:
        config: Hydra configuration
    """
    # Print configuration
    logger.info("Configuration:")
    logger.info(config)
    # Create and run pipeline
    pipeline = TrainingPipeline(config)
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        if config.wandb.enabled and pipeline.wandb_run:
            pipeline.wandb_run.finish()
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        if config.wandb.enabled and pipeline.wandb_run:
            pipeline.wandb_run.finish()
        raise


if __name__ == "__main__":
    main()
