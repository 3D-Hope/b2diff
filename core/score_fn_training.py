"""
Core training module for B2Diff training pipeline.
Extracts the training logic from run_train.py into a callable function.
"""

import os
import torch
import contextlib
import copy
import json
import tree
from functools import partial
from tqdm import tqdm as tqdm_lib
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusion.ddim_with_logprob import ddim_step_with_logprob
from utils.utils import load_sample_stage, seed_everything

tqdm = partial(tqdm_lib, dynamic_ncols=True)
logger = get_logger(__name__)


def run_score_fn_training(config, stage_idx=None, external_logger=None, wandb_run=None, pipeline=None, trainable_layers=None, training_timesteps=None):
    """
    Run the training phase for a given stage.
    
    Args:
        config: Configuration object (can be OmegaConf or dict)
        stage_idx: Current stage index (optional, for logging)
        external_logger: External logger instance (optional)
        wandb_run: Existing wandb run to log to (optional)
        pipeline: Pre-loaded StableDiffusionPipeline (avoids reloading)
        trainable_layers: Pre-initialized trainable layers
        
    Returns:
        save_dir: Directory where checkpoints were saved
    """
    # Convert OmegaConf to dict if needed
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config
        
    if external_logger:
        external_logger.info(f"Starting training for stage {stage_idx}")
    else:
        print(f"Starting training for stage {stage_idx}")
    
    print("Starting training")
    torch.cuda.set_device(config.dev_id)
    
    # Setup directories
    unique_id = config.exp_name
    os.makedirs(os.path.join(config.save_path, unique_id), exist_ok=True)
    
    stage_id = f"stage{stage_idx}"
    save_dir = os.path.join(config.save_path, unique_id, stage_id)
    
    # Use pre-loaded pipeline (no reloading!)
    if pipeline is None or trainable_layers is None:
        raise ValueError("Pipeline and trainable_layers must be provided - should not reload model each stage!")
    
    # Determine effective number of training timesteps for gradient accumulation
    if getattr(config.train, 'incremental_training', False) and training_timesteps is not None:
        num_train_timesteps_2 = int(len(training_timesteps))
    else:
        num_train_timesteps_2 = int(config.split_step * config.train.timestep_fraction)
    
    # Setup accelerator (per-stage, with correct gradient_accumulation_steps)
    accelerator_config = ProjectConfiguration(
        project_dir=save_dir,
        automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps_2,
    )
    
    # IMPORTANT: Do NOT use log_with="wandb" - prevents separate run creation
    # All logging goes through the parent pipeline's wandb_run
    
    logger.info(f"\n{config}")
    seed_everything(config.seed)
    
    # Setup inference dtype
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    
    # Setup save/load hooks for checkpointing (using original pattern)
    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()
    
    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()
    
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        trainable_layers.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    
    # Generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Prepare everything with accelerator
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)
    
    # Log training info
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )
    
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.train.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    
    if not config.sample.fk:
        assert config.sample.batch_size >= config.train.batch_size
        
        assert config.sample.batch_size % config.train.batch_size == 0
    else:
        assert (config.sample.batch_size * 4 * 2) >= config.train.batch_size
        
        assert (config.sample.batch_size * 4 * 2) % config.train.batch_size == 0
    # No checkpoint loading - model stays in memory across stages!
    # if config.resume_from:
    #     logger.info(f"Resuming from {config.resume_from}")
    #     accelerator.load_state(config.resume_from)
    
    # Load sample data
    samples = load_sample_stage(save_dir)
    accelerator.save_state()
    
    pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
    
    # DDPOSF: Single step training per data collection
    # Equation: ∇θJDDRL = E[∑(t=0 to T) ∇θ log pθ(xt−1 | xt, c) r(x0, c)]
    
    # Get rewards (evaluation scores)
    evaluation_score = samples["eval_scores"].to(accelerator.device)
    
    # Apply reward weighting (beta1 for positive, beta2 for negative)
    temp_beta1 = torch.ones_like(evaluation_score) * config.train.beta1
    temp_beta2 = torch.ones_like(evaluation_score) * config.train.beta2
    sample_weight = torch.where(evaluation_score > 0, temp_beta1, temp_beta2)
    
    # Clip advantages to prevent extreme gradients
    advantages = torch.clamp(
        evaluation_score,
        -config.train.adv_clip_max,
        config.train.adv_clip_max,
    ) * sample_weight

    # Get log probabilities from sampling: log pθ(xt−1 | xt, c)
    trajectories_log_probs = samples["log_probs"].to(accelerator.device)
    
    # DDPOSF loss: -E[∑_t log pθ(xt−1 | xt, c) * r(x0, c)]
    # Sum over timesteps (dim=1), then mean over batch, weighted by advantages
    loss = -torch.mean(torch.sum(trajectories_log_probs, dim=1) * advantages)
    
    # Backward pass
    accelerator.backward(loss)
    
    # Gradient clipping
    total_norm = None
    if accelerator.sync_gradients:
        total_norm = accelerator.clip_grad_norm_(
            trainable_layers.parameters(), 
            config.train.max_grad_norm
        )
    
    # Extract scalar values for logging
    loss_value = loss.cpu().item()
    grad_value = total_norm.cpu().item() if total_norm is not None else None
    
    # Optimizer step - single update per data collection
    optimizer.step()
    optimizer.zero_grad()
    
    # Log metrics to wandb (only if enabled)
    if wandb_run and accelerator.is_main_process and config.wandb.enabled:
        log_dict = {
            "train/loss": loss_value,
            "train/stage": stage_idx,
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "train/eval_score_mean": evaluation_score.mean().cpu().item(),
            "train/eval_score_std": evaluation_score.std().cpu().item(),
            "train/eval_score_min": evaluation_score.min().cpu().item(),
            "train/eval_score_max": evaluation_score.max().cpu().item(),
            "train/advantage_mean": advantages.mean().cpu().item(),
            "train/log_prob_sum_mean": torch.sum(trajectories_log_probs, dim=1).mean().cpu().item(),
        }
        if grad_value is not None:
            log_dict["train/grad_norm"] = grad_value
        wandb_run.log(log_dict)
    
    # Log to console
    if accelerator.is_main_process:
        logger.info(f"Stage {stage_idx} training completed:")
        logger.info(f"  Loss: {loss_value:.6f}")
        logger.info(f"  Grad Norm: {grad_value:.6f}" if grad_value else "  Grad Norm: N/A")
        logger.info(f"  Eval Score Mean: {evaluation_score.mean().cpu().item():.6f}")
    
    # Save checkpoint (only every 10 stages to save disk space)
    if stage_idx is not None and stage_idx % 10 == 0:
        accelerator.save_state()
        if accelerator.is_main_process:
            logger.info(f"  Checkpoint saved at stage {stage_idx}")
    
    # Save metrics
    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    metrics = {
        "loss": loss_value,
        "grad_norm": grad_value,
        "eval_score_mean": evaluation_score.mean().cpu().item(),
        "eval_score_std": evaluation_score.std().cpu().item(),
        "advantage_mean": advantages.mean().cpu().item(),
    }
    with open(os.path.join(save_dir, 'eval', 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if external_logger:
        external_logger.info(f"Training completed for stage {stage_idx}")
    
    return save_dir
