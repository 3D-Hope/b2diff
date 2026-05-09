import os
import json
import random
import contextlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.utils.torch_utils import randn_tensor

from diffusion.ddim_with_logprob import latents_decode
from utils.utils import seed_everything


@dataclass
class TreeNode:
    node_id: str
    parent_id: Optional[str]
    child_ids: List[str]
    step: Optional[int]
    depth: int
    batch_idx: int
    log_prob: Optional[torch.Tensor]

    def to_dict(self):
        data = asdict(self)
        if self.log_prob is not None:
            data["log_prob"] = self.log_prob.detach().cpu()
        return data


def _left_broadcast(t, shape):
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(scheduler, timestep, prev_timestep):
    alpha_prod_t = torch.gather(scheduler.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        scheduler.alphas_cumprod.gather(0, prev_timestep.cpu()),
        scheduler.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)


def _compute_prev_sample_mean_and_std(scheduler, model_output, timestep, sample, eta):
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = torch.clamp(prev_timestep, 0, scheduler.config.num_train_timesteps - 1)

    alpha_prod_t = scheduler.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        scheduler.alphas_cumprod.gather(0, prev_timestep.cpu()),
        scheduler.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t

    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_epsilon = model_output
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
    else:
        raise ValueError(f"Unsupported prediction_type {scheduler.config.prediction_type}")

    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range,
            scheduler.config.clip_sample_range,
        )

    variance = _get_variance(scheduler, timestep, prev_timestep)
    std_dev_t = eta * variance ** 0.5
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon
    prev_sample_mean = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

    sample_dtype = sample.dtype
    return prev_sample_mean.to(sample_dtype), std_dev_t.to(sample_dtype)


def _compute_log_prob(prev_sample, prev_sample_mean, std_dev_t):
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t ** 2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(torch.pi)))
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def _save_tree_artifacts(save_dir, tree_data, prompts_for_images, image_tensors):
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

    for idx, image in enumerate(image_tensors):
        pil = torch.clamp(image, 0, 1)
        pil = (pil.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
        from PIL import Image
        Image.fromarray(pil).save(os.path.join(save_dir, f"images/{idx:05}.png"))

    with open(os.path.join(save_dir, "prompt.json"), "w") as f:
        json.dump(prompts_for_images, f)

    torch.save(tree_data, os.path.join(save_dir, "branch_grpo_tree.pt"))


def run_branch_grpo_sampling(
    config,
    stage_idx=None,
    logger=None,
    wandb_run=None,
    pipeline=None,
    trainable_layers=None,
    resume_from_ckpt=False,
):
    if logger:
        logger.info(f"Starting BranchGRPO sampling for stage {stage_idx}")

    torch.cuda.set_device(config.dev_id)
    seed_everything(config.seed)

    unique_id = config.exp_name
    os.makedirs(os.path.join(config.save_path, unique_id), exist_ok=True)
    stage_id = f"stage{stage_idx}"
    save_dir = os.path.join(config.save_path, unique_id, stage_id)
    os.makedirs(save_dir, exist_ok=True)

    if pipeline is None or trainable_layers is None:
        raise ValueError("Pipeline and trainable_layers must be provided for BranchGRPO sampling")

    accelerator_config = ProjectConfiguration(
        project_dir=save_dir,
        automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
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

    if resume_from_ckpt:
        prev_stage = stage_idx - 1
        checkpoint_num = config.train.num_epochs // config.train.save_interval
        checkpoint_path = os.path.join(
            config.save_path,
            config.exp_name,
            f"stage{prev_stage}",
            "checkpoints",
            f"checkpoint_{checkpoint_num}",
        )
        checkpoint_path = os.path.normpath(os.path.expanduser(checkpoint_path))
        if "checkpoint_" not in os.path.basename(checkpoint_path):
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(checkpoint_path)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {checkpoint_path}")
            checkpoint_path = os.path.join(
                checkpoint_path,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        accelerator.load_state(checkpoint_path)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if trainable_layers is not None and not hasattr(trainable_layers, "_hf_hook"):
        trainable_layers = accelerator.prepare(trainable_layers)

    pipeline.unet.eval()

    split_points = set(config.branch_grpo.split_points)
    branch_factor = int(config.branch_grpo.branch_factor)
    split_noise_scale = float(config.branch_grpo.split_noise_scale)

    prompt_list = []
    if len(config.prompt) == 0:
        prompt_file_path = config.prompt_file
        if not os.path.isabs(prompt_file_path):
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            prompt_file_path = os.path.join(project_root, prompt_file_path)
        with open(prompt_file_path) as f:
            prompt_list = json.load(f)

    prompt_idx = 0
    prompt_cnt = len(prompt_list)

    pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    else:
        inference_dtype = torch.float32
    timesteps = pipeline.scheduler.timesteps

    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0].to(inference_dtype)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    node_data = {}
    latent_data = {}
    log_prob_data = {}
    root_ids = []
    leaf_ids = []
    prompts_for_images = []
    image_tensors = []

    prompt_pool = []
    prompt_embeds_pool = []
    global_prompt_idx = 0

    for _ in range(config.sample.num_batches_per_epoch):
        if len(config.prompt) != 0:
            prompts = [config.prompt for _ in range(config.sample.batch_size)]
        elif config.prompt_random_choose:
            prompts = [random.choice(prompt_list) for _ in range(config.sample.batch_size)]
        else:
            prompts = [prompt_list[(prompt_idx + i) % prompt_cnt] for i in range(config.sample.batch_size)]
            prompt_idx += config.sample.batch_size

        prompt_ids = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0].to(inference_dtype)

        for idx in range(len(prompts)):
            prompt_pool.append(prompts[idx])
            prompt_embeds_pool.append(prompt_embeds[idx].detach().cpu())
            batch_idx = global_prompt_idx
            global_prompt_idx += 1

            root_latent = pipeline.prepare_latents(
                1,
                pipeline.unet.config.in_channels,
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                inference_dtype,
                accelerator.device,
                None,
            )
            root_id = f"p{batch_idx}_root"
            root_node = TreeNode(
                node_id=root_id,
                parent_id=None,
                child_ids=[],
                step=None,
                depth=0,
                batch_idx=batch_idx,
                log_prob=None,
            )
            node_data[root_id] = root_node.to_dict()
            latent_data[root_id] = root_latent.detach().cpu()
            log_prob_data[root_id] = None
            root_ids.append(root_id)

            nodes_current = [root_node]

            for step_idx, t in enumerate(timesteps):
                should_split = step_idx in split_points
                nodes_next = []

                for node in nodes_current:
                    latent = latent_data[node.node_id].to(accelerator.device, dtype=inference_dtype)

                    if config.sample.cfg:
                        embeds = torch.cat([neg_prompt_embed, prompt_embeds[idx : idx + 1]], dim=0)
                        latents_input = torch.cat([latent] * 2)
                        timesteps_input = torch.cat([t.unsqueeze(0)] * 2)
                    else:
                        embeds = prompt_embeds[idx : idx + 1]
                        latents_input = latent
                        timesteps_input = t.unsqueeze(0)

                    latents_input = pipeline.scheduler.scale_model_input(latents_input, t)
                    latents_input = latents_input.to(inference_dtype)

                    with torch.no_grad():
                        with autocast():
                            noise_pred = pipeline.unet(
                                latents_input,
                                timesteps_input,
                                encoder_hidden_states=embeds,
                                return_dict=False,
                            )[0]

                    if config.sample.cfg:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    prev_sample_mean, std_dev_t = _compute_prev_sample_mean_and_std(
                        pipeline.scheduler,
                        noise_pred,
                        t,
                        latent,
                        eta=config.sample.eta,
                    )

                    num_children = branch_factor if should_split else 1
                    noise_scale = split_noise_scale if should_split else 1.0

                    for split_idx in range(num_children):
                        noise = randn_tensor(
                            prev_sample_mean.shape,
                            generator=None,
                            device=prev_sample_mean.device,
                            dtype=prev_sample_mean.dtype,
                        )
                        noise = noise * noise_scale
                        prev_sample = prev_sample_mean + std_dev_t * noise
                        log_prob = _compute_log_prob(prev_sample, prev_sample_mean, std_dev_t)

                        if should_split:
                            child_id = f"{node.node_id}_s{step_idx}_{split_idx}"
                        else:
                            child_id = f"{node.node_id}_t{step_idx}"

                        child_node = TreeNode(
                            node_id=child_id,
                            parent_id=node.node_id,
                            child_ids=[],
                            step=step_idx,
                            depth=node.depth + 1,
                            batch_idx=batch_idx,
                            log_prob=log_prob.detach().cpu(),
                        )
                        node.child_ids.append(child_id)
                        node_data[node.node_id]["child_ids"].append(child_id)
                        node_data[child_id] = child_node.to_dict()
                        latent_data[child_id] = prev_sample.detach().cpu()
                        log_prob_data[child_id] = log_prob.detach().cpu()
                        nodes_next.append(child_node)

                nodes_current = nodes_next

            for leaf in nodes_current:
                leaf_ids.append(leaf.node_id)
                prompts_for_images.append(prompt_pool[leaf.batch_idx])
                with torch.no_grad():
                    image = latents_decode(
                        pipeline,
                        latent_data[leaf.node_id].to(accelerator.device),
                        accelerator.device,
                        prompt_embeds.dtype,
                    )
                image_tensors.append(image.squeeze(0).detach().cpu())

    if accelerator.is_local_main_process:
        tree_data = {
            "nodes": node_data,
            "latents": latent_data,
            "log_probs": log_prob_data,
            "root_ids": root_ids,
            "leaf_ids": leaf_ids,
            "prompt_embeds": torch.stack(prompt_embeds_pool, dim=0),
            "timesteps": timesteps.detach().cpu(),
        }
        _save_tree_artifacts(save_dir, tree_data, prompts_for_images, image_tensors)

    if wandb_run:
        wandb_run.log({
            f"branch_grpo_sampling/stage_{stage_idx}/num_prompts": len(prompt_pool),
            f"branch_grpo_sampling/stage_{stage_idx}/num_leaves": len(leaf_ids),
        })

    if logger:
        logger.info(f"BranchGRPO sampling completed for stage {stage_idx}")

    return save_dir
