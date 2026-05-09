import os
import json
import contextlib

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers

from diffusion.ddim_with_logprob import ddim_step_with_logprob
from utils.utils import seed_everything


def _load_edges(save_dir):
    return torch.load(os.path.join(save_dir, "branch_grpo_edges.pt"), map_location="cpu")


def _parse_stop_depth(config):
    stop_depth = config.branch_grpo.pruning_stop_depth
    return int(stop_depth)


def _parse_base_depths(config):
    base_depths = getattr(config.branch_grpo, "depth_pruning_depths", None)
    if base_depths is None:
        raise ValueError("branch_grpo.depth_pruning_depths must be set for depth pruning")
    if isinstance(base_depths, str):
        base_depths = [int(d.strip()) for d in base_depths.split(",") if d.strip()]
    return sorted([int(d) for d in base_depths])


def _active_pruning_depths(config, stage_idx):
    base_depths = _parse_base_depths(config)
    interval = max(1, int(config.branch_grpo.pruning_slide_interval_stages))
    shift_now = max(0, int(stage_idx // interval))

    stop_depth = _parse_stop_depth(config)
    max_shift = max(0, min(base_depths) - stop_depth)
    shift_now = min(shift_now, max_shift)

    return [d - shift_now for d in base_depths]


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

    edge_data = _load_edges(save_dir)
    if external_logger:
        external_logger.info(f"Loaded {edge_data['parent_latents'].shape[0]} edges for training")

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

    total_edges = edge_data["parent_latents"].shape[0]
    trees_per_batch = int(config.train.batch_size)
    edge_microbatch_size = int(getattr(config.branch_grpo, "edge_microbatch_size", 16))
    if trees_per_batch <= 0:
        raise ValueError(f"train.batch_size must be > 0, got {trees_per_batch}")

    tree_to_indices = {}
    for idx, tree_id in enumerate(edge_data["batch_idx"].tolist()):
        tree_to_indices.setdefault(tree_id, []).append(idx)
    tree_ids = sorted(tree_to_indices.keys())
    if len(tree_ids) < trees_per_batch:
        raise ValueError(
            f"Not enough trees for one update: trees={len(tree_ids)}, train.batch_size={trees_per_batch}"
        )

    LossRecord = []
    MetricsRecord = []  # Track all metrics: loss, grad_norm, ratio stats, advantage stats, leaf rewards

    global_step = 0

    for epoch in range(config.train.num_epochs):
        LossRecord.append([])
        MetricsRecord.append({})
        effective_tree_total = (len(tree_ids) // trees_per_batch) * trees_per_batch

        for batch_start in range(0, effective_tree_total, trees_per_batch):
            LossRecord[epoch].append([])
            batch_tree_ids = tree_ids[batch_start : batch_start + trees_per_batch]

            tree_losses = []
            tree_clipfracs = []
            tree_adv_means = []

            with accelerator.accumulate(pipeline.unet):
                for tree_id in batch_tree_ids:
                    indices = torch.tensor(tree_to_indices[tree_id], dtype=torch.long)

                    parent_latents = edge_data["parent_latents"][indices].to(accelerator.device)
                    child_latents = edge_data["child_latents"][indices].to(accelerator.device)
                    timesteps = edge_data["timesteps"][indices].to(accelerator.device)
                    old_log_probs = edge_data["old_log_probs"][indices].to(accelerator.device)
                    advantages = edge_data["advantages"][indices].to(accelerator.device)
                    batch_idx_slice = edge_data["batch_idx"][indices].to(accelerator.device)
                    child_depth = edge_data["child_depth"][indices].to(accelerator.device)

                    if config.branch_grpo.depth_pruning:
                        active_depths = _active_pruning_depths(config, stage_idx)
                        mask = torch.ones_like(child_depth, dtype=torch.bool)
                        for depth in active_depths:
                            mask &= child_depth != depth
                        if not mask.any():
                            continue
                        parent_latents = parent_latents[mask]
                        child_latents = child_latents[mask]
                        timesteps = timesteps[mask]
                        old_log_probs = old_log_probs[mask]
                        advantages = advantages[mask]
                        batch_idx_slice = batch_idx_slice[mask]

                    total_tree_edges = parent_latents.shape[0]
                    if total_tree_edges == 0:
                        continue

                    tree_loss_accum = torch.tensor(0.0, device=accelerator.device)
                    tree_clipfrac_accum = []
                    tree_adv_accum = []
                    tree_ratio_means = []
                    tree_ratio_stds = []
                    tree_ratio_maxs = []
                    tree_kl_divs = []
                    tree_leaf_rewards = []  # Track rewards of leaf nodes (max depth edges)

                    for start in range(0, total_tree_edges, edge_microbatch_size):
                        end = min(start + edge_microbatch_size, total_tree_edges)
                        micro_parent_latents = parent_latents[start:end]
                        micro_child_latents = child_latents[start:end]
                        micro_timesteps = timesteps[start:end]
                        micro_old_log_probs = old_log_probs[start:end]
                        micro_advantages = advantages[start:end]
                        micro_batch_idx = batch_idx_slice[start:end]

                        prompt_embeds = edge_data["prompt_embeds"][micro_batch_idx.cpu()].to(accelerator.device)

                        if config.train.cfg:
                            embeds = torch.cat(
                                [neg_prompt_embed.repeat(prompt_embeds.shape[0], 1, 1), prompt_embeds],
                                dim=0,
                            )
                            latents_input = torch.cat([micro_parent_latents] * 2)
                        else:
                            embeds = prompt_embeds
                            latents_input = micro_parent_latents

                        latents_input = pipeline.scheduler.scale_model_input(latents_input, micro_timesteps)
                        with autocast():
                            noise_pred = pipeline.unet(
                                latents_input,
                                micro_timesteps,
                                encoder_hidden_states=embeds,
                            ).sample

                            if config.train.cfg:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                                    noise_pred_text - noise_pred_uncond
                                )

                            _, new_log_prob, _ = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                micro_timesteps,
                                micro_parent_latents,
                                eta=config.sample.eta,
                                prev_sample=micro_child_latents,
                            )

                            ratio = torch.exp(new_log_prob - micro_old_log_probs)
                            clipped_adv = torch.clamp(
                                micro_advantages,
                                -config.train.adv_clip_max,
                                config.train.adv_clip_max,
                            )
                            unclipped_loss = -clipped_adv * ratio
                            clipped_loss = -clipped_adv * torch.clamp(
                                ratio,
                                1.0 - config.train.eps,
                                1.0 + config.train.eps,
                            )
                            micro_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            
                            # Compute metrics for this microbatch
                            with torch.no_grad():
                                # Ratio statistics
                                ratio_np = ratio.detach().cpu()
                                tree_ratio_means.append(ratio_np.mean().item())
                                tree_ratio_stds.append(ratio_np.std().item())
                                tree_ratio_maxs.append(ratio_np.max().item())
                                
                                # KL divergence estimate
                                approx_kl = 0.5 * torch.mean((new_log_prob - micro_old_log_probs) ** 2)
                                tree_kl_divs.append(approx_kl.detach().cpu().item())
                                
                                # Track leaf rewards (edges at max depth in this microbatch)
                                max_depth_in_batch = child_depth[start:end].max()
                                max_depth_mask = (child_depth[start:end] == max_depth_in_batch)
                                if max_depth_mask.any():
                                    leaf_rewards = micro_advantages[max_depth_mask]
                                    tree_leaf_rewards.append(leaf_rewards.mean().detach().cpu().item())

                        weight = (end - start) / float(total_tree_edges)
                        accelerator.backward(micro_loss * weight)
                        tree_loss_accum = tree_loss_accum + micro_loss.detach() * weight
                        tree_clipfrac_accum.append(
                            torch.mean((torch.abs(ratio - 1.0) > config.train.eps).float()).detach()
                        )
                        tree_adv_accum.append(micro_advantages.mean().detach())

                    tree_losses.append(tree_loss_accum)
                    tree_clipfracs.append(torch.stack(tree_clipfrac_accum).mean())
                    tree_adv_means.append(torch.stack(tree_adv_accum).mean())

                if len(tree_losses) == 0:
                    global_step += 1
                    continue

                batch_loss = torch.stack(tree_losses).mean()
                
                # Compute batch-level metrics
                batch_ratio_mean = sum(tree_ratio_means) / max(len(tree_ratio_means), 1)
                batch_ratio_std = sum(tree_ratio_stds) / max(len(tree_ratio_stds), 1)
                batch_ratio_max = max(tree_ratio_maxs) if tree_ratio_maxs else 0
                batch_kl_div = sum(tree_kl_divs) / max(len(tree_kl_divs), 1)
                batch_leaf_reward = sum(tree_leaf_rewards) / max(len(tree_leaf_rewards), 1) if tree_leaf_rewards else 0
                
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(trainable_layers.parameters(), config.train.max_grad_norm)
                else:
                    grad_norm = None
                    
                optimizer.step()
                optimizer.zero_grad()

            LossRecord[epoch][-1].append(batch_loss.item())

            # Compute aggregated metrics for logging
            if len(tree_losses) > 0:
                batch_metrics = {
                    "loss": batch_loss.item(),
                    "clipfrac": torch.stack(tree_clipfracs).mean().item(),
                    "adv_mean": torch.stack(tree_adv_means).mean().item(),
                    "ratio_mean": batch_ratio_mean,
                    "ratio_std": batch_ratio_std,
                    "ratio_max": batch_ratio_max,
                    "kl_div": batch_kl_div,
                    "leaf_reward": batch_leaf_reward,
                    "grad_norm": grad_norm.item() if grad_norm is not None else 0,
                }
                
                if epoch not in MetricsRecord or not MetricsRecord[epoch]:
                    MetricsRecord[epoch] = {
                        "losses": [],
                        "clipfracs": [],
                        "adv_means": [],
                        "ratio_means": [],
                        "ratio_stds": [],
                        "ratio_maxs": [],
                        "kl_divs": [],
                        "leaf_rewards": [],
                        "grad_norms": [],
                    }
                MetricsRecord[epoch]["losses"].append(batch_metrics["loss"])
                MetricsRecord[epoch]["clipfracs"].append(batch_metrics["clipfrac"])
                MetricsRecord[epoch]["adv_means"].append(batch_metrics["adv_mean"])
                MetricsRecord[epoch]["ratio_means"].append(batch_metrics["ratio_mean"])
                MetricsRecord[epoch]["ratio_stds"].append(batch_metrics["ratio_std"])
                MetricsRecord[epoch]["ratio_maxs"].append(batch_metrics["ratio_max"])
                MetricsRecord[epoch]["kl_divs"].append(batch_metrics["kl_div"])
                MetricsRecord[epoch]["leaf_rewards"].append(batch_metrics["leaf_reward"])
                MetricsRecord[epoch]["grad_norms"].append(batch_metrics["grad_norm"])

                if wandb_run and accelerator.is_main_process and config.wandb.enabled:
                    wandb_run.log({
                        "branch_grpo/train_loss": batch_metrics["loss"],
                        "branch_grpo/clipfrac": batch_metrics["clipfrac"],
                        "branch_grpo/adv_mean": batch_metrics["adv_mean"],
                        "branch_grpo/ratio_mean": batch_metrics["ratio_mean"],
                        "branch_grpo/ratio_std": batch_metrics["ratio_std"],
                        "branch_grpo/ratio_max": batch_metrics["ratio_max"],
                        "branch_grpo/kl_divergence": batch_metrics["kl_div"],
                        "branch_grpo/leaf_reward_mean": batch_metrics["leaf_reward"],
                        "branch_grpo/grad_norm": batch_metrics["grad_norm"],
                        "branch_grpo/step": global_step,
                    })

            global_step += 1

        if accelerator.is_main_process:
            if wandb_run and config.wandb.enabled:
                epoch_losses = [item for sublist in LossRecord[epoch] for item in sublist]
                if len(epoch_losses) > 0 and epoch in MetricsRecord:
                    epoch_metrics = MetricsRecord[epoch]
                    epoch_loss_mean = sum(epoch_losses) / len(epoch_losses)
                    epoch_loss_std = (sum((x - epoch_loss_mean)**2 for x in epoch_losses) / len(epoch_losses))**0.5 if len(epoch_losses) > 1 else 0
                    
                    def safe_mean(lst):
                        return sum(lst) / max(len(lst), 1) if lst else 0
                    
                    def safe_std(lst, mean_val):
                        return (sum((x - mean_val)**2 for x in lst) / len(lst))**0.5 if len(lst) > 1 else 0
                    
                    epoch_clipfrac_mean = safe_mean(epoch_metrics.get("clipfracs", []))
                    epoch_adv_mean = safe_mean(epoch_metrics.get("adv_means", []))
                    epoch_ratio_mean = safe_mean(epoch_metrics.get("ratio_means", []))
                    epoch_ratio_std = safe_mean(epoch_metrics.get("ratio_stds", []))
                    epoch_ratio_max = max(epoch_metrics.get("ratio_maxs", [0]))
                    epoch_kl_mean = safe_mean(epoch_metrics.get("kl_divs", []))
                    epoch_leaf_reward_mean = safe_mean(epoch_metrics.get("leaf_rewards", []))
                    epoch_grad_norm_mean = safe_mean(epoch_metrics.get("grad_norms", []))
                    
                    wandb_run.log({
                        "branch_grpo/epoch_loss_mean": epoch_loss_mean,
                        "branch_grpo/epoch_loss_std": epoch_loss_std,
                        "branch_grpo/epoch_clipfrac_mean": epoch_clipfrac_mean,
                        "branch_grpo/epoch_adv_mean": epoch_adv_mean,
                        "branch_grpo/epoch_ratio_mean": epoch_ratio_mean,
                        "branch_grpo/epoch_ratio_std": epoch_ratio_std,
                        "branch_grpo/epoch_ratio_max": epoch_ratio_max,
                        "branch_grpo/epoch_kl_divergence_mean": epoch_kl_mean,
                        "branch_grpo/epoch_leaf_reward_mean": epoch_leaf_reward_mean,
                        "branch_grpo/epoch_grad_norm_mean": epoch_grad_norm_mean,
                        "branch_grpo/epoch_completed": epoch + 1,
                    })

        if (epoch + 1) % config.train.save_interval == 0:
            accelerator.save_state()

    os.makedirs(os.path.join(save_dir, "eval"), exist_ok=True)
    with open(os.path.join(save_dir, "eval", "branch_grpo_loss.json"), "w") as f:
        json.dump(LossRecord, f)
    with open(os.path.join(save_dir, "eval", "branch_grpo_metrics.json"), "w") as f:
        json.dump(MetricsRecord, f)

    if external_logger:
        external_logger.info(f"BranchGRPO training completed for stage {stage_idx}")

    return save_dir
