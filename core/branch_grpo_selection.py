import os
import json
import pickle
from collections import defaultdict

import torch
import wandb

from core.selection import score_fn1, geometric_algebraic_score_fn


def _load_tree(save_dir):
    return torch.load(os.path.join(save_dir, "branch_grpo_tree.pt"), map_location="cpu")


def _save_edges(save_dir, edge_data):
    torch.save(edge_data, os.path.join(save_dir, "branch_grpo_edges.pt"))


def _compute_depth_stats(values, eps=1e-8):
    values_tensor = torch.stack(values)
    if values_tensor.numel() <= 1:
        return values_tensor, torch.tensor(0.0), torch.tensor(0.0), torch.zeros_like(values_tensor)
    mean_val = values_tensor.mean()
    std_val = values_tensor.std()
    if std_val < eps:
        adv = torch.zeros_like(values_tensor)
    else:
        adv = (values_tensor - mean_val) / (std_val + eps)
    return values_tensor, mean_val, std_val, adv


def run_branch_grpo_selection(config, stage_idx=None, logger=None, wandb_run=None):
    if logger:
        logger.info(f"Starting BranchGRPO selection for stage {stage_idx}")

    torch.cuda.set_device(config.dev_id)

    unique_id = config.exp_name
    stage_id = f"stage{stage_idx}"
    save_dir = os.path.join(config.save_path, unique_id, stage_id)

    tree_data = _load_tree(save_dir)
    nodes = tree_data["nodes"]
    leaf_ids = tree_data["leaf_ids"]

    with open(os.path.join(save_dir, "prompt.json"), "r") as f:
        prompts_for_images = json.load(f)

    img_dir = os.path.join(save_dir, "images")
    reward_fn_name = getattr(config, "reward_fn", "clip")
    if reward_fn_name == "geometric":
        eval_scores, _, raw_scores = geometric_algebraic_score_fn(
            prompts_for_images,
            img_dir,
            save_dir,
            config,
            only_raw_scores=True,
        )
    else:
        eval_scores, _, raw_scores = score_fn1(
            prompts_for_images,
            img_dir,
            save_dir,
            config,
            only_raw_scores=True,
        )

    leaf_rewards = {leaf_id: eval_scores[idx] for idx, leaf_id in enumerate(leaf_ids)}
    node_rewards = {}

    depth_to_nodes = defaultdict(list)
    max_depth = 0
    for node_id, node in nodes.items():
        depth = int(node["depth"])
        depth_to_nodes[depth].append(node_id)
        max_depth = max(max_depth, depth)

    # reward fusion
    for leaf_id in leaf_ids:
        node_rewards[leaf_id] = leaf_rewards[leaf_id]

    for depth in range(max_depth - 1, -1, -1):
        for node_id in depth_to_nodes.get(depth, []):
            child_ids = nodes[node_id]["child_ids"]
            if len(child_ids) == 0:
                continue
            if len(child_ids) == 1:
                node_rewards[node_id] = node_rewards[child_ids[0]]
                continue

            child_log_probs = []
            child_rewards = []
            for child_id in child_ids:
                lp = nodes[child_id].get("log_prob")
                lp_val = lp.squeeze().to(torch.float32) if lp is not None else torch.tensor(0.0)
                child_log_probs.append(lp_val)
                child_rewards.append(node_rewards[child_id].to(torch.float32))
            child_log_probs = torch.stack(child_log_probs)
            weights = torch.softmax(child_log_probs, dim=0)
            rewards_tensor = torch.stack(child_rewards).to(weights.device)
            node_rewards[node_id] = torch.sum(weights * rewards_tensor)

    # depthwise normalization per prompt
    node_advantages = {}


    for depth, node_ids in depth_to_nodes.items():
        values = [node_rewards[nid].to(torch.float32) for nid in node_ids]
        values_tensor, mean_val, std_val, adv = _compute_depth_stats(values)
        for idx, node_id in enumerate(node_ids):
            node_advantages[node_id] = adv[idx]



    # create flat data structure for training with depth prunning
    parent_latents = []
    child_latents = []
    timesteps = []
    old_log_probs = []
    advantages = []
    batch_indices = []
    child_depths = []
    step_indices = []
    if config.branch_grpo.depth_pruning:
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

        depths_to_prune = _active_pruning_depths(config, stage_idx)
    else:
        depths_to_prune = []


    for node_id, node in nodes.items():
        parent_id = node.get("parent_id")
        if parent_id is None or node["depth"] in depths_to_prune:
            continue
        parent_latents.append(tree_data["latents"][parent_id])
        child_latents.append(tree_data["latents"][node_id])
        timesteps.append(tree_data["timesteps"][node["step"]])
        old_log_probs.append(node.get("log_prob"))
        advantages.append(node_advantages[node_id])
        # batch_indices.append(node["batch_idx"])
        # child_depths.append(node["depth"])
        # step_indices.append(node["step"])

    parent_latents_tensor = torch.stack(parent_latents, dim=0)
    child_latents_tensor = torch.stack(child_latents, dim=0)
    if parent_latents_tensor.ndim == 5 and parent_latents_tensor.shape[1] == 1:
        parent_latents_tensor = parent_latents_tensor.squeeze(1)
    if child_latents_tensor.ndim == 5 and child_latents_tensor.shape[1] == 1:
        child_latents_tensor = child_latents_tensor.squeeze(1)

    edge_data = {
        "latents": parent_latents_tensor,
        "next_latents": child_latents_tensor,
        "timesteps": torch.stack(timesteps, dim=0),
        "log_probs": torch.stack(old_log_probs, dim=0),
        "advantages": torch.stack(advantages, dim=0),
        "prompt_embeds": tree_data["prompt_embeds"],
        # "batch_idx": torch.tensor(batch_indices, dtype=torch.long),
        # "child_depth": torch.tensor(child_depths, dtype=torch.long),
        # "step_idx": torch.tensor(step_indices, dtype=torch.long),
    }
    _save_edges(save_dir, edge_data)

    num_generated = len(leaf_ids)
    mean_reward = float(raw_scores.mean()) if num_generated > 0 else 0.0


    if logger:
        logger.info(f"BranchGRPO selection completed for stage {stage_idx}")
        logger.info(f"Generated: {num_generated}, Mean reward: {mean_reward:.4f} ")

    return save_dir, mean_reward
