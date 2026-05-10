import os
import json
import contextlib
import copy
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers

from diffusion.ddim_with_logprob import ddim_step_with_logprob
from utils.utils import seed_everything
from tqdm import tqdm as tqdm_lib
from functools import partial

tqdm = partial(tqdm_lib, dynamic_ncols=True)


def _load_edges(save_dir):
    return torch.load(os.path.join(save_dir, "branch_grpo_edges.pt"), map_location="cpu")






def run_branch_grpo_training(
    config,
    stage_idx=None,
    external_logger=None,
    wandb_run=None,
    pipeline=None,
    trainable_layers=None,
):
    if external_logger:
        external_logger.info(f"Starting BranchGRPO training for stage {stage_idx}")

    torch.cuda.set_device(config.dev_id)
    seed_everything(config.seed)

    unique_id = config.exp_name
    stage_id = f"stage{stage_idx}"
    save_dir = os.path.join(config.save_path, unique_id, stage_id)

    if pipeline is None or trainable_layers is None:
        raise ValueError("Pipeline and trainable_layers must be provided for BranchGRPO training")

    samples = _load_edges(save_dir)
    if external_logger:
        external_logger.info(f"Loaded {samples['latents'].shape[0]} edges for training")

    accelerator_config = ProjectConfiguration(
        project_dir=save_dir,
        automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

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
                config.pretrained.model,
                revision=config.pretrained.revision,
                subfolder="unet",
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

    optimizer = torch.optim.AdamW(
        trainable_layers.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

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

    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)

    total_batch_size = samples["latents"].shape[0]
    edge_microbatch_size = int(getattr(config.branch_grpo, "edge_microbatch_size", 16))


    LossRecord = []
    GradRecord = []

    init_samples = copy.deepcopy(samples)
    for epoch in range(config.train.num_epochs):
        LossRecord.append([])
        GradRecord.append([])
        
        pipeline.unet.train()
        # shuffle edges at the start of each epoch
        # perm = torch.randperm(total_batch_size)
        # for k, v in init_samples.items():
        #     if k == "prompt_embeds": continue
        #     samples[k] = v[perm]

        effective_batch_size = edge_microbatch_size
        effective_total = (total_batch_size // effective_batch_size) * effective_batch_size

        for idx in tqdm(
            range(0, effective_total, effective_batch_size),
            desc="Update",
            position=2,
            leave=False,
        ):
            sample = {k: (v[idx:idx + effective_batch_size].to(accelerator.device) if isinstance(v, torch.Tensor) else v[idx:idx + effective_batch_size]) for k, v in samples.items()}
            prompt_embeds = samples["prompt_embeds"].repeat(effective_batch_size, 1, 1).to(accelerator.device)
            embeds = torch.cat(
                [neg_prompt_embed.repeat(prompt_embeds.shape[0], 1, 1), prompt_embeds],
                dim=0,
            )
            with accelerator.accumulate(pipeline.unet):
                with autocast():
                    if config.train.cfg:
                        noise_pred = pipeline.unet(
                                torch.cat([sample["latents"]] * 2),
                                torch.cat([sample["timesteps"]] * 2),
                                embeds,
                            ).sample
                            
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = pipeline.unet(
                            sample["latents"], sample["timesteps"], embeds
                        ).sample

                    _, new_log_prob, _ = ddim_step_with_logprob(
                        pipeline.scheduler,
                        noise_pred,
                        sample["timesteps"],
                        sample["latents"],
                        eta=config.sample.eta,
                        prev_sample=sample["next_latents"],
                    )

                    ratio = torch.exp(new_log_prob - sample["log_probs"])
                    clipped_adv = torch.clamp(
                        sample["advantages"],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    unclipped_loss = -clipped_adv * ratio
                    clipped_loss = -clipped_adv * torch.clamp(
                        ratio,
                        1.0 - config.train.eps,
                        1.0 + config.train.eps,
                    )
                    loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (effective_batch_size * accelerator.gradient_accumulation_steps)
                    
                accelerator.backward(loss)  
                total_norm = None
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(trainable_layers.parameters(), config.train.max_grad_norm) 
                    # this is working. returns the grad norm before clipping.  
                optimizer.step()
                optimizer.zero_grad()

                loss_value = loss.cpu().item() * (effective_batch_size * accelerator.gradient_accumulation_steps)
                grad_value = total_norm.cpu().item() if total_norm is not None else None
                
                LossRecord[epoch].append(loss_value)
                GradRecord[epoch].append(grad_value)

                if wandb_run and accelerator.is_main_process and config.wandb.enabled:
                    log_dict = {
                        "train/loss": loss_value,
                        "train/epoch": epoch,
                    }
                    if grad_value is not None:
                        log_dict["train/grad_norm"] = grad_value
                    wandb_run.log(log_dict)

        if accelerator.is_main_process:
            if external_logger:
                external_logger.info(f"Epoch {epoch + 1}/{config.train.num_epochs} completed")
            
            if wandb_run and config.wandb.enabled:
                epoch_losses = LossRecord[epoch]
                epoch_grads = [g for g in GradRecord[epoch] if g is not None]
                wandb_run.log({
                    "train/epoch_loss_mean": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
                    "train/epoch_grad_mean": sum(epoch_grads) / len(epoch_grads) if epoch_grads else 0,
                    "train/epoch_completed": epoch + 1,
                })

    # Save final metrics
    
    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'loss.json'), 'w') as f:
        json.dump(LossRecord, f)
    with open(os.path.join(save_dir, 'eval', 'grad.json'), 'w') as f:
        json.dump(GradRecord, f)

    # Save checkpoint once every 45 stages
    if stage_idx is not None and stage_idx > 0 and stage_idx % 5 == 0:
        accelerator.save_state()
        if external_logger:
            external_logger.info(f"Saved checkpoint correctly at stage {stage_idx}")

    if external_logger:
        external_logger.info(f"BranchGRPO training completed for stage {stage_idx}")

    return save_dir
