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
    prompt_depth_to_nodes = defaultdict(lambda: defaultdict(list))
    max_depth = 0
    for node_id, node in nodes.items():
        depth = int(node["depth"])
        depth_to_nodes[depth].append(node_id)
        prompt_depth_to_nodes[node["batch_idx"]][depth].append(node_id)
        max_depth = max(max_depth, depth)

    for leaf_id in leaf_ids:
        node_rewards[leaf_id] = leaf_rewards[leaf_id]

    # reward fusion
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
    depth_reward_stats = {}
    depth_adv_stats = {}

    depth_reward_means = defaultdict(list)
    depth_reward_stds = defaultdict(list)
    depth_counts = defaultdict(int)
    depth_adv_values = defaultdict(list)

    for batch_idx, depth_nodes in prompt_depth_to_nodes.items():
        for depth, node_ids in depth_nodes.items():
            values = [node_rewards[nid].to(torch.float32) for nid in node_ids]
            values_tensor, mean_val, std_val, adv = _compute_depth_stats(values)
            depth_reward_means[depth].append(mean_val.item())
            depth_reward_stds[depth].append(std_val.item())
            depth_counts[depth] += int(values_tensor.numel())
            depth_adv_values[depth].append(adv)
            for idx, node_id in enumerate(node_ids):
                node_advantages[node_id] = adv[idx]

    # logging
    for depth in depth_to_nodes.keys():
        means = depth_reward_means.get(depth, [])
        stds = depth_reward_stds.get(depth, [])
        depth_reward_stats[depth] = {
            "mean": float(torch.tensor(means).mean()) if len(means) > 0 else 0.0,
            "std": float(torch.tensor(stds).mean()) if len(stds) > 0 else 0.0,
            "count": depth_counts.get(depth, 0),
        }
        adv_chunks = depth_adv_values.get(depth, [])
        if len(adv_chunks) > 0:
            depth_adv_stats[depth] = torch.cat(adv_chunks, dim=0)
        else:
            depth_adv_stats[depth] = torch.tensor([])

    if wandb_run and config.wandb.enabled:
        log_dict = {}
        if config.branch_grpo.log_depth_stats:
            for depth, stats in depth_reward_stats.items():
                log_dict[f"branch_grpo/reward_depth/{depth}/mean"] = stats["mean"]
                log_dict[f"branch_grpo/reward_depth/{depth}/std"] = stats["std"]
                log_dict[f"branch_grpo/reward_depth/{depth}/count"] = stats["count"]
        if config.branch_grpo.log_adv_histograms:
            for depth, adv in depth_adv_stats.items():
                if adv.numel() > 0:
                    log_dict[f"branch_grpo/adv_depth/{depth}/hist"] = wandb.Histogram(adv.cpu().numpy())
        if log_dict:
            wandb_run.log(log_dict)

    # create flat data structure for training
    parent_latents = []
    child_latents = []
    timesteps = []
    old_log_probs = []
    advantages = []
    batch_indices = []
    child_depths = []
    step_indices = []

    for node_id, node in nodes.items():
        parent_id = node.get("parent_id")
        if parent_id is None:
            continue
        parent_latents.append(tree_data["latents"][parent_id])
        child_latents.append(tree_data["latents"][node_id])
        timesteps.append(tree_data["timesteps"][node["step"]])
        old_log_probs.append(node.get("log_prob"))
        advantages.append(node_advantages[node_id])
        batch_indices.append(node["batch_idx"])
        child_depths.append(node["depth"])
        step_indices.append(node["step"])

    parent_latents_tensor = torch.stack(parent_latents, dim=0)
    child_latents_tensor = torch.stack(child_latents, dim=0)
    if parent_latents_tensor.ndim == 5 and parent_latents_tensor.shape[1] == 1:
        parent_latents_tensor = parent_latents_tensor.squeeze(1)
    if child_latents_tensor.ndim == 5 and child_latents_tensor.shape[1] == 1:
        child_latents_tensor = child_latents_tensor.squeeze(1)

    edge_data = {
        "parent_latents": parent_latents_tensor,
        "child_latents": child_latents_tensor,
        "timesteps": torch.stack(timesteps, dim=0),
        "old_log_probs": torch.stack(old_log_probs, dim=0),
        "advantages": torch.stack(advantages, dim=0),
        "batch_idx": torch.tensor(batch_indices, dtype=torch.long),
        "child_depth": torch.tensor(child_depths, dtype=torch.long),
        "step_idx": torch.tensor(step_indices, dtype=torch.long),
        "prompt_embeds": tree_data["prompt_embeds"],
    }
    _save_edges(save_dir, edge_data)

    num_generated = len(leaf_ids)
    mean_reward = float(raw_scores.mean()) if num_generated > 0 else 0.0
    std_reward = float(raw_scores.std()) if num_generated > 0 else 0.0
    num_positive = int((raw_scores >= config.eval.pos_threshold).sum()) if num_generated > 0 else 0
    num_negative = int((raw_scores < config.eval.neg_threshold).sum()) if num_generated > 0 else 0

    cumulative_queries_path = os.path.join(config.save_path, unique_id, "cumulative_reward_queries.pkl")
    if os.path.exists(cumulative_queries_path):
        with open(cumulative_queries_path, "rb") as f:
            cumulative_reward_queries = pickle.load(f)
    else:
        cumulative_reward_queries = 0
    cumulative_reward_queries += num_generated
    with open(cumulative_queries_path, "wb") as f:
        pickle.dump(cumulative_reward_queries, f)

    selection_metrics = {
        "num_selected": num_generated,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "num_generated": num_generated,
        "num_rejected": 0,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "num_queries": num_generated,
        "cumulative_queries": cumulative_reward_queries,
    }

    if logger:
        logger.info(f"BranchGRPO selection completed for stage {stage_idx}")
        logger.info(f"Generated: {num_generated}, Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")

    return save_dir, selection_metrics
