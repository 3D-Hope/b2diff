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


def _should_bootstrap_from_universal(config, stage_idx):
    if not getattr(config, "continue_from_universal", False):
        return False

    start_stage = int(getattr(getattr(config, "pipeline", None), "continue_from_stage", 0))
    if start_stage != 0:
        return False
    if stage_idx is None:
        return True
    return int(stage_idx) == 0


def _load_universal_checkpoint_if_needed(
    *,
    config,
    stage_idx,
    pipeline,
    accelerator,
    external_logger=None,
):
    if not _should_bootstrap_from_universal(config, stage_idx):
        return

    checkpoint_root = getattr(config, "path_to_universal_lora", None)
    if checkpoint_root in (None, ""):
        raise ValueError(
            "continue_from_universal=true requires path_to_universal_lora to be set."
        )

    checkpoint_root = os.path.normpath(os.path.expanduser(str(checkpoint_root)))
    if not os.path.exists(checkpoint_root):
        raise FileNotFoundError(
            f"Universal checkpoint path does not exist: {checkpoint_root}"
        )

    if getattr(config, "threed_scene_layout", False):
        if config.use_lora:
            lora_path = (
                checkpoint_root
                if checkpoint_root.endswith(".pt")
                else os.path.join(checkpoint_root, "lora_weights.pt")
            )
            if not os.path.isfile(lora_path):
                raise FileNotFoundError(
                    f"Could not find LoRA checkpoint file: {lora_path}"
                )
            state = torch.load(lora_path, map_location=accelerator.device)
            pipeline.load_state_dict(state, strict=False)
        else:
            model_path = (
                checkpoint_root
                if checkpoint_root.endswith(".pt")
                else os.path.join(checkpoint_root, "model.pt")
            )
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Could not find model checkpoint file: {model_path}"
                )
            state = torch.load(model_path, map_location=accelerator.device)
            pipeline.model.load_state_dict(state)
    else:
        accelerator.load_state(checkpoint_root)

    message = f"Bootstrapped model from universal checkpoint: {checkpoint_root}"
    logger.info(message)
    if external_logger:
        external_logger.info(message)


def run_training(config, stage_idx=None, external_logger=None, wandb_run=None, pipeline=None, trainable_layers=None, training_timesteps=None):
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
    
    seed_everything(config.seed)
    
    # Setup inference dtype
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    
    # Setup save/load hooks for checkpointing
    threed = getattr(config, 'threed_scene_layout', False)

    if threed:
        def save_model_hook(models, weights, output_dir):
            assert len(models) == 1
            if config.use_lora:
                # Save only the LoRA delta weights
                import peft
                lora_state = {k: v for k, v in models[0].state_dict().items() if 'lora_' in k}
                torch.save(lora_state, os.path.join(output_dir, 'lora_weights.pt'))
            else:
                torch.save(models[0].state_dict(), os.path.join(output_dir, 'model.pt'))
            weights.pop()

        def load_model_hook(models, input_dir):
            assert len(models) == 1
            if config.use_lora:
                lora_state = torch.load(os.path.join(input_dir, 'lora_weights.pt'), map_location='cpu')
                models[0].load_state_dict(lora_state, strict=False)
            else:
                state = torch.load(os.path.join(input_dir, 'model.pt'), map_location='cpu')
                models[0].load_state_dict(state)
            models.pop()
    else:
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

    _load_universal_checkpoint_if_needed(
        config=config,
        stage_idx=stage_idx,
        pipeline=pipeline,
        accelerator=accelerator,
        external_logger=external_logger,
    )
    

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
    
    if not threed:
        # 2D SD path: generate negative prompt embeddings for CFG
        neg_prompt_embed = pipeline.text_encoder(
            pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
        )[0]
    # 3D path: no text encoder, no CFG — neg_prompt_embed not needed

    # For 3D (MiDiffusion), always use nullcontext (no mixed-precision autocast needed).
    # For 2D SD with LoRA, also use nullcontext; otherwise use accelerator.autocast.
    autocast = contextlib.nullcontext if (threed or config.use_lora) else accelerator.autocast

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

    if threed:
        # Set up the DDIM scheduler used during sampling so that timesteps match.
        from diffusers import DDIMScheduler as _DDIMScheduler
        _ddim_3d = _DDIMScheduler(
            num_train_timesteps=getattr(config.midiffusion, 'num_timesteps', 1000),
            beta_start=getattr(config.midiffusion, 'beta_start', 1e-4),
            beta_end=getattr(config.midiffusion, 'beta_end', 0.02),
            clip_sample=False,
            prediction_type="epsilon",
            steps_offset=1,
        )
        _ddim_3d.set_timesteps(config.sample.num_steps, device=accelerator.device)
    else:
        pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)

    init_samples = copy.deepcopy(samples)

    # Guard: ensure eval_scores is always a tensor (selection.py should guarantee this,
    # but defend here too in case an old pickle is loaded).
    _es_raw = init_samples.get("eval_scores", torch.tensor([]))
    if isinstance(_es_raw, list):
        _es_raw = torch.stack(_es_raw) if len(_es_raw) > 0 else torch.tensor([])
        init_samples["eval_scores"] = _es_raw
    if _es_raw.shape[0] == 0:
        logger.warning("No training samples available (empty sample_stage.pkl) — skipping training for this stage.")
        return save_dir

    LossRecord = []
    GradRecord = []
    
    # Key aliases: 3D uses 'scenes'/'next_scenes'; SD uses 'latents'/'next_latents'
    if threed:
        _lat_key      = 'scenes'
        _next_lat_key = 'next_scenes'
        _embed_key    = 'fpbpn'
    else:
        _lat_key      = 'latents'
        _next_lat_key = 'next_latents'
        _embed_key    = 'prompt_embeds'

    # Filter trajectories to only include specified timesteps for uniformly_sample_timesteps
    if config.train.uniformly_sample_timesteps and training_timesteps is not None:
        if external_logger and accelerator.is_local_main_process:
            external_logger.info(f"Filtering trajectories to timesteps: {training_timesteps}")

        if isinstance(training_timesteps, list):
            timestep_indices = torch.tensor(training_timesteps, dtype=torch.long)
        else:
            timestep_indices = training_timesteps

        for key in [_lat_key, _next_lat_key, "log_probs", "timesteps"]:
            if key in samples:
                # samples[key] has shape [batch_size, num_timesteps, ...]
                # We want to keep only the timesteps at the specified indices
                samples[key] = samples[key][:, timestep_indices]
    # Training loop
    for epoch in range(config.train.num_epochs):
        # Shuffle samples along batch dimension
        samples = {}
        LossRecord.append([])
        GradRecord.append([])
        
        total_batch_size = init_samples["eval_scores"].shape[0]
        _grpo_cfg = getattr(config, "grpo", None)
        _grpo_enabled = bool(getattr(_grpo_cfg, "enabled", False)) if _grpo_cfg is not None else False

        if _grpo_enabled:
            # GRPO: do NOT shuffle the batch dimension, so each group's
            # `group_size` rollouts (which are contiguous as written by
            # run_sampling) stay together and flow through the same
            # optimizer step. Otherwise the per-group advantage baseline
            # we computed in selection becomes mismatched at update time.
            group_size = int(_grpo_cfg.group_size)
            train_bs = int(config.train.batch_size)
            ga_steps = int(config.train.gradient_accumulation_steps)
            effective_bs = train_bs * ga_steps
            # Either train_bs is a multiple of group_size (group fits in one
            # minibatch), or the effective batch (after grad accum) covers
            # whole groups (group fits across an accumulation cycle).
            ok = (train_bs % group_size == 0) or (effective_bs % group_size == 0)
            if not ok:
                raise ValueError(
                    f"[GRPO] train.batch_size ({train_bs}) must be a multiple "
                    f"of grpo.group_size ({group_size}), or train.batch_size * "
                    f"gradient_accumulation_steps ({effective_bs}) must be a "
                    f"multiple of group_size. Got neither."
                )
            if external_logger and accelerator.is_local_main_process and epoch == 0:
                external_logger.info(
                    f"[GRPO] training preserving sample order (no batch shuffle). "
                    f"group_size={group_size}, train_bs={train_bs}, ga={ga_steps}."
                )
            samples = {k: v.clone() if torch.is_tensor(v) else v
                       for k, v in init_samples.items()}
        else:
            perm = torch.randperm(total_batch_size)
            samples = {k: v[perm] for k, v in init_samples.items()}
        
        
        
        # Shuffle timesteps (always shuffle, even for progressive training)
        current_num_timesteps = samples["timesteps"].shape[1]
        perms = torch.stack(
            [torch.randperm(current_num_timesteps) for _ in range(total_batch_size)]
        )
        for key in [_lat_key, _next_lat_key, "log_probs", "timesteps"]:
            samples[key] = samples[key][torch.arange(total_batch_size)[:, None], perms]

        # ---- Set the model to training mode ----
        if threed:
            pipeline.model.train()
            _trainable_module = pipeline.model  # used for accumulate()
        else:
            pipeline.unet.train()
            _trainable_module = pipeline.unet

        for idx in tqdm(
            range(0, total_batch_size // 2 * 2, config.train.batch_size),
            desc="Update",
            position=2,
            leave=False,
        ):
            LossRecord[epoch].append([])
            GradRecord[epoch].append([])

            sample = tree.map_structure(lambda value: value[idx:idx + config.train.batch_size].to(accelerator.device), samples)

            if threed:
                fpbpn = sample[_embed_key]  # (B, 256, 4) floor condition
            else:
                sample_batch_size = sample[_embed_key].shape[0]
                train_neg_prompt_embeds = neg_prompt_embed.repeat(sample_batch_size, 1, 1)
                if config.train.cfg:
                    embeds = torch.cat([train_neg_prompt_embeds, sample[_embed_key]])
                else:
                    embeds = sample[_embed_key]

            if getattr(config.train, 'incremental_training', False) and training_timesteps is not None:
                t_indices = training_timesteps
                if external_logger and accelerator.is_local_main_process and idx == 0:
                    external_logger.info(f"Training on {len(t_indices)}/{sample['timesteps'].shape[1]} timesteps: {t_indices}")
            else:
                t_indices = range(sample["timesteps"].shape[1])

            for t in tqdm(
                t_indices,
                desc="Timestep",
                position=3,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):
                evaluation_score = sample["eval_scores"][:]

                with accelerator.accumulate(_trainable_module):
                    with autocast():
                        if threed:
                            # MiDiffusion: no CFG, predict noise from (x_t, t, fpbpn)
                            noise_pred = pipeline.predict_noise(
                                sample[_lat_key][:, t],
                                sample["timesteps"][:, t],
                                fpbpn,
                            )  # (B, N, C)
                            _, total_prob, _ = ddim_step_with_logprob(
                                _ddim_3d,
                                noise_pred,
                                sample["timesteps"][:, t],
                                sample[_lat_key][:, t],
                                eta=getattr(config.sample, 'eta', 0.0),
                                prev_sample=sample[_next_lat_key][:, t],
                            )
                        else:
                            # Stable Diffusion path (with optional CFG)
                            if config.train.cfg:
                                noise_pred = pipeline.unet(
                                    torch.cat([sample[_lat_key][:, t]] * 2),
                                    torch.cat([sample["timesteps"][:, t]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                            else:
                                noise_pred = pipeline.unet(
                                    sample[_lat_key][:, t], sample["timesteps"][:, t], embeds
                                ).sample
                            _, total_prob, _ = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, t],
                                sample[_lat_key][:, t],
                                eta=config.sample.eta,
                                prev_sample=sample[_next_lat_key][:, t],
                            )

                        total_ref_prob = sample["log_probs"][:, t]

                        ratio = torch.exp(total_prob - total_ref_prob)
                        temp_beta1 = torch.ones_like(evaluation_score) * config.train.beta1
                        temp_beta2 = torch.ones_like(evaluation_score) * config.train.beta2
                        sample_weight = torch.where(evaluation_score > 0, temp_beta1, temp_beta2)
                        advantages = torch.clamp(
                            evaluation_score,
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        ) * sample_weight
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.eps,
                            1.0 + config.train.eps,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                    accelerator.backward(loss)
                    total_norm = None
                    if accelerator.sync_gradients:
                        total_norm = accelerator.clip_grad_norm_(trainable_layers.parameters(), config.train.max_grad_norm)

                    loss_value = loss.cpu().item()
                    grad_value = total_norm.cpu().item() if total_norm is not None else None

                    approx_kl = 0.5 * torch.mean((total_prob - total_ref_prob) ** 2)
                    clipfrac  = torch.mean((torch.abs(ratio - 1.0) > config.train.eps).float())

                    LossRecord[epoch][idx // config.train.batch_size].append(loss_value)
                    GradRecord[epoch][idx // config.train.batch_size].append(grad_value)

                    if wandb_run and accelerator.is_main_process and config.wandb.enabled:
                        log_dict = {
                            "train/loss":             loss_value,
                            "train/epoch":            epoch,
                            "train/learning_rate":    optimizer.param_groups[0]['lr'],
                            "train/batch_idx":        idx // config.train.batch_size,
                            "train/eval_score_mean":  evaluation_score.mean().cpu().item(),
                            "train/eval_score_std":   evaluation_score.std().cpu().item(),
                            "train/ratio_mean":       ratio.mean().cpu().item(),
                            "train/approx_kl":        approx_kl.cpu().item(),
                            "train/clipfrac":         clipfrac.cpu().item(),
                        }
                        if grad_value is not None:
                            log_dict["train/grad_norm"] = grad_value
                        wandb_run.log(log_dict)

                    optimizer.step()
                    optimizer.zero_grad()
        
        # Log epoch summary
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1}/{config.train.num_epochs} completed")
            
            # Log epoch-level metrics (only if enabled)
            if wandb_run and config.wandb.enabled:
                epoch_losses = [item for sublist in LossRecord[epoch] for item in sublist]
                epoch_grads = [item for sublist in GradRecord[epoch] for item in sublist if item is not None]
                wandb_run.log({
                    "train/epoch_loss_mean": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
                    "train/epoch_loss_std": (sum((x - sum(epoch_losses) / len(epoch_losses))**2 for x in epoch_losses) / len(epoch_losses))**0.5 if epoch_losses else 0,
                    "train/epoch_grad_mean": sum(epoch_grads) / len(epoch_grads) if epoch_grads else 0,
                    "train/epoch_completed": epoch + 1,
                })
        
        # Save checkpoint
        if (epoch + 1) % config.train.save_interval == 0:
            accelerator.save_state()
    
    # Save final metrics
    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'loss.json'), 'w') as f:
        json.dump(LossRecord, f)
    with open(os.path.join(save_dir, 'eval', 'grad.json'), 'w') as f:
        json.dump(GradRecord, f)
    
    if external_logger:
        external_logger.info(f"Training completed for stage {stage_idx}")
    
    return save_dir
