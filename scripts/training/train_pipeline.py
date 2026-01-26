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
        
        # Metrics aggregation for progression curves across ALL stages
        self.metrics_history = {
            'stages': [],  # Stage indices
            'num_selected': [],  # Number of selected samples per stage
            'num_positive': [],  # Number of positive samples per stage
            'num_negative': [],  # Number of negative samples per stage
            'num_generated': [],  # Total generated samples per stage
            'num_rejected': [],  # Rejected samples per stage
            'mean_reward': [],  # Mean reward per stage
            'std_reward': [],  # Std reward per stage
            'cumulative_queries': [],  # Cumulative reward queries
            'stage_duration': [],  # Time taken per stage
        }
        
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
            
            # Aggregate metrics for progression curves
            # metrics dict contains: num_selected, num_positive, num_negative, num_generated, 
            # num_rejected, mean_reward, std_reward, num_queries, cumulative_queries
            self.metrics_history['stages'].append(stage_idx)
            self.metrics_history['num_selected'].append(metrics.get('num_selected', 0))
            self.metrics_history['num_positive'].append(metrics.get('num_positive', 0))
            self.metrics_history['num_negative'].append(metrics.get('num_negative', 0))
            self.metrics_history['num_generated'].append(metrics.get('num_generated', 0))
            self.metrics_history['num_rejected'].append(metrics.get('num_rejected', 0))
            self.metrics_history['mean_reward'].append(metrics.get('mean_reward', 0.0))
            self.metrics_history['std_reward'].append(metrics.get('std_reward', 0.0))
            self.metrics_history['cumulative_queries'].append(metrics.get('cumulative_queries', 0))
            self.metrics_history['stage_duration'].append(stage_time)
            
            # Log aggregated metrics with stage as x-axis for clean progression curves
            if self.config.wandb.enabled and self.wandb_run:
                log_dict = {
                    # Progression curves (clean - shows evolution across stages)
                    "progression/num_selected": metrics.get('num_selected', 0),
                    "progression/num_positive": metrics.get('num_positive', 0),
                    "progression/num_negative": metrics.get('num_negative', 0),
                    "progression/num_generated": metrics.get('num_generated', 0),
                    "progression/num_rejected": metrics.get('num_rejected', 0),
                    "progression/selection_rate": (metrics.get('num_selected', 0) / max(metrics.get('num_generated', 1), 1)) * 100,
                    "progression/mean_reward": metrics.get('mean_reward', 0.0),
                    "progression/std_reward": metrics.get('std_reward', 0.0),
                    "progression/cumulative_queries": metrics.get('cumulative_queries', 0),
                    "progression/stage_duration_seconds": stage_time,
                    "progression/stage_duration_minutes": stage_time / 60,
                    # Pipeline level info
                    "pipeline/stage": stage_idx,
                    "pipeline/elapsed_hours": (time.time() - self.start_time) / 3600,
                }
                self.wandb_run.log(log_dict)
            
            logger.info(f"[{stage_idx}] Stage completed in {stage_time:.2f}s ({stage_time/60:.2f}m)")
            logger.info(f"  → Generated: {metrics.get('num_generated', 0)}, Selected: {metrics.get('num_selected', 0)}, Rejected: {metrics.get('num_rejected', 0)}")
            logger.info(f"  → Positive: {metrics.get('num_positive', 0)}, Negative: {metrics.get('num_negative', 0)}")
            logger.info(f"  → Mean reward: {metrics.get('mean_reward', 0.0):.4f} ± {metrics.get('std_reward', 0.0):.4f}")
            
            # Sleep between stages
            if stage_idx < self.config.pipeline.stage_cnt - 1:
                time.sleep(self.config.pipeline.sleep_between_stages)
                
        except Exception as e:
            logger.error(f"Error in stage {stage_idx}: {str(e)}", exc_info=True)
            
            if self.config.wandb.enabled and self.wandb_run:
                self.wandb_run.log({
                    "pipeline/stage": stage_idx,
                    "error/stage_failed": True,
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
        
        # Save metrics and generate summary tables
        logger.info("Saving metrics and generating summary...")
        self.save_metrics_history()
        self.create_summary_table()
        
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
        """Save aggregated metrics history to JSON file for analysis."""
        import json
        
        metrics_file = os.path.join("logs", f"{self.config.exp_name}_progression_metrics.json")
        
        # Prepare data in a format that's easy to plot
        metrics_export = {
            'exp_name': self.config.exp_name,
            'total_stages_completed': len(self.metrics_history['stages']),
            'data': {
                'stages': self.metrics_history['stages'],
                'num_selected': self.metrics_history['num_selected'],
                'num_positive': self.metrics_history['num_positive'],
                'num_negative': self.metrics_history['num_negative'],
                'num_generated': self.metrics_history['num_generated'],
                'num_rejected': self.metrics_history['num_rejected'],
                'mean_reward': self.metrics_history['mean_reward'],
                'std_reward': self.metrics_history['std_reward'],
                'cumulative_queries': self.metrics_history['cumulative_queries'],
                'stage_duration_seconds': self.metrics_history['stage_duration'],
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_export, f, indent=2)
        
        logger.info(f"✓ Metrics saved to {metrics_file}")
        return metrics_file
    
    def create_summary_table(self):
        """Create and log a summary table of all stages."""
        if len(self.metrics_history['stages']) == 0:
            logger.warning("No stages completed - skipping summary table")
            return
        
        # Prepare table data
        summary_data = []
        for i in range(len(self.metrics_history['stages'])):
            summary_data.append({
                'Stage': self.metrics_history['stages'][i],
                'Generated': self.metrics_history['num_generated'][i],
                'Selected': self.metrics_history['num_selected'][i],
                'Rejected': self.metrics_history['num_rejected'][i],
                'Positive': self.metrics_history['num_positive'][i],
                'Negative': self.metrics_history['num_negative'][i],
                'Selection %': f"{(self.metrics_history['num_selected'][i] / max(self.metrics_history['num_generated'][i], 1)) * 100:.1f}%",
                'Mean Reward': f"{self.metrics_history['mean_reward'][i]:.4f}",
                'Reward Std': f"{self.metrics_history['std_reward'][i]:.4f}",
                'Duration (m)': f"{self.metrics_history['stage_duration'][i] / 60:.2f}",
            })
        
        # Log to console
        logger.info("\n" + "=" * 150)
        logger.info("STAGE PROGRESSION SUMMARY")
        logger.info("=" * 150)
        
        # Print header
        header = " | ".join([f"{k:>12}" for k in summary_data[0].keys()])
        logger.info(header)
        logger.info("-" * len(header))
        
        # Print rows
        for row in summary_data:
            row_str = " | ".join([f"{v:>12}" for v in row.values()])
            logger.info(row_str)
        
        logger.info("=" * 150 + "\n")
        
        # Save table as CSV for easy analysis
        csv_file = os.path.join("logs", f"{self.config.exp_name}_summary_table.csv")
        try:
            import csv
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                writer.writeheader()
                writer.writerows(summary_data)
            logger.info(f"✓ Summary table saved to {csv_file}")
        except Exception as e:
            logger.warning(f"Could not save CSV table: {e}")
        
        # Log summary table to wandb if available
        if self.config.wandb.enabled and self.wandb_run:
            try:
                import wandb
                # Create wandb table
                columns = list(summary_data[0].keys())
                table = wandb.Table(columns=columns)
                for row in summary_data:
                    table.add_data(*row.values())
                self.wandb_run.log({"summary/stage_progression": table})
                logger.info("✓ Summary table logged to wandb")
            except Exception as e:
                logger.warning(f"Could not log table to wandb: {e}")


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
