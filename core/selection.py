"""
Core selection module for B2Diff training pipeline.
Extracts the selection logic from run_select.py into a callable function.
"""

import os
import torch
import torch.nn.functional as F
import pickle
import json
import numpy as np
from PIL import Image
from functools import partial
from tqdm import tqdm as tqdm_lib
from bert_score import score, BERTScorer
import open_clip
from utils.utils import seed_everything
from core.utils.geometric_rewards import ImageGeometricReward

tqdm = partial(tqdm_lib, dynamic_ncols=True)

def compute_aabb_penetration_depth(centers1, sizes1, centers2, sizes2):
    """
    Compute penetration depth between two sets of AABBs.

    Penetration depth is defined as the minimum translation distance needed to
    separate two overlapping objects. For AABBs, this is the minimum overlap
    across all axes.

    Args:
        centers1: (B, N1, 3) - centers of first set of boxes
        sizes1: (B, N1, 3) - sizes of first set of boxes
        centers2: (B, N2, 3) - centers of second set of boxes
        sizes2: (B, N2, 3) - sizes of second set of boxes

    Returns:
        penetration_depth: (B, N1, N2) - penetration depth (0 if no overlap)
    """
    batch_size = centers1.shape[0]
    device = centers1.device

    # Expand dimensions for pairwise comparison
    c1 = centers1.unsqueeze(2)  # (B, N1, 1, 3)
    s1 = sizes1.unsqueeze(2)  # (B, N1, 1, 3)
    c2 = centers2.unsqueeze(1)  # (B, 1, N2, 3)
    s2 = sizes2.unsqueeze(1)  # (B, 1, N2, 3)

    # NOTE: sizes are already half-extents (sx/2, sy/2, sz/2)
    # So we use them directly
    half1 = s1  # (B, N1, 1, 3) - already half-extents
    half2 = s2  # (B, 1, N2, 3) - already half-extents

    # Compute center distance for each axis
    center_dist = torch.abs(c1 - c2)  # (B, N1, N2, 3)

    # Compute sum of half-extents for each axis
    sum_half_extents = half1 + half2  # (B, N1, N2, 3)

    # Overlap on each axis = sum_of_half_extents - center_distance
    # If positive, there's overlap; if negative or zero, no overlap
    overlap_per_axis = sum_half_extents - center_dist  # (B, N1, N2, 3)

    # For AABBs to overlap, they must overlap on ALL axes
    # Check if overlapping on all axes
    is_overlapping_all_axes = (overlap_per_axis > 0).all(dim=3)  # (B, N1, N2)

    # Penetration depth is the MINIMUM overlap across all axes
    # (the smallest amount needed to separate the objects)
    min_overlap_per_pair = overlap_per_axis.min(dim=3)[0]  # (B, N1, N2)

    # Only consider penetration where objects actually overlap on all axes
    penetration_depth = torch.where(
        is_overlapping_all_axes,
        min_overlap_per_pair,
        torch.zeros_like(min_overlap_per_pair),
    )

    # Clamp to ensure non-negative
    penetration_depth = torch.clamp(penetration_depth, min=0.0)

    return penetration_depth


def compute_non_penetration_reward(parsed_scenes, **kwargs):
    """
    Calculate reward based on non-penetration constraint using penetration depth.

    Following the approach from original authors: reward = sum of negative signed distances.
    When objects overlap, we get positive penetration depth, so reward is negative.

    Args:
        parsed_scenes: Dict returned by parse_and_descale_scenes()

    Returns:
        rewards: Tensor of shape (B,) with non-penetration rewards for each scene
    """
    room_type = kwargs["room_type"]
    positions = parsed_scenes["positions"]
    sizes = parsed_scenes["sizes"]
    object_indices = parsed_scenes["object_indices"]
    is_empty = parsed_scenes["is_empty"]
    batch_size = positions.shape[0]
    device = positions.device
    
    # print(f"Parsed scene: pos {positions[:10]} sizes: {sizes[:10]}")
    # print(f"Parsed scene: {parsed_scenes}")

    # Identify ceiling objects (they don't participate in ground-level collisions)
    ceiling_objects = ["ceiling_lamp", "pendant_lamp"]
    idx_to_labels_bedroom = {
        0: "armchair",
        1: "bookshelf",
        2: "cabinet",
        3: "ceiling_lamp",
        4: "chair",
        5: "children_cabinet",
        6: "coffee_table",
        7: "desk",
        8: "double_bed",
        9: "dressing_chair",
        10: "dressing_table",
        11: "kids_bed",
        12: "nightstand",
        13: "pendant_lamp",
        14: "shelf",
        15: "single_bed",
        16: "sofa",
        17: "stool",
        18: "table",
        19: "tv_stand",
        20: "wardrobe",
    }
    ceiling_indices = [
        idx for idx, label in idx_to_labels_bedroom.items() if label in ceiling_objects
    ]
    is_ceiling = torch.zeros_like(is_empty, dtype=torch.bool)
    for ceiling_idx in ceiling_indices:
        is_ceiling |= object_indices == ceiling_idx

    # Create mask for ground objects (non-empty, non-ceiling)
    is_ground_object = ~is_empty & ~is_ceiling

    # Compute pairwise penetration depths
    penetration_depth = compute_aabb_penetration_depth(
        positions, sizes, positions, sizes
    )  # (B, N, N)

    # Create mask to ignore self-overlaps (diagonal), empty objects, and ceiling objects
    mask = is_ground_object.unsqueeze(2) & is_ground_object.unsqueeze(1)  # (B, N, N)

    # Remove self-overlaps (diagonal)
    eye = torch.eye(positions.shape[1], device=device, dtype=torch.bool)
    eye = eye.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, N)
    mask = mask & ~eye

    # Apply mask to penetration depths
    masked_penetration = torch.where(
        mask, penetration_depth, torch.zeros_like(penetration_depth)
    )

    # Sum total penetration depth per scene
    # Divide by 2 because each pair is counted twice (i,j) and (j,i)
    total_penetration = masked_penetration.sum(dim=[1, 2]) / 2.0  # (B,)

    # Convert to reward: +1 if no penetration, else -penetration_depth
    # This provides a clear positive signal for valid scenes and negative for invalid ones
    rewards = torch.where(
        total_penetration == 0,
        torch.ones_like(total_penetration),
        -total_penetration
    )
    return rewards


# Bedroom object index lookup (shared across reward functions)
_IDX_TO_LABEL_BEDROOM = {
    0: "armchair", 1: "bookshelf", 2: "cabinet", 3: "ceiling_lamp",
    4: "chair", 5: "children_cabinet", 6: "coffee_table", 7: "desk",
    8: "double_bed", 9: "dressing_chair", 10: "dressing_table",
    11: "kids_bed", 12: "nightstand", 13: "pendant_lamp", 14: "shelf",
    15: "single_bed", 16: "sofa", 17: "stool", 18: "table",
    19: "tv_stand", 20: "wardrobe",
}
_TV_STAND_IDX = 19
_BED_INDICES  = {8}  # double_bed, kids_bed, single_bed


def compute_tv_bed_presence_reward(parsed_scene, ideal=3.0, sigma=1.0, **kwargs):
    """
    Gaussian-shaped reward for ideal bed–TV distance.
    """
    device = parsed_scene["device"]
    object_indices = parsed_scene["object_indices"]
    positions = parsed_scene["positions"]
    orientations = parsed_scene["orientations"]
    is_empty = parsed_scene["is_empty"]
    idx_to_labels = kwargs.get("idx_to_labels", _IDX_TO_LABEL_BEDROOM)

    # Handle both integer and string keys
    if idx_to_labels and isinstance(list(idx_to_labels.keys())[0], str):
        idx_to_labels = {int(k): v for k, v in idx_to_labels.items()}

    idx_tv = next((k for k, v in idx_to_labels.items() if "tv_stand" in v), None)
    idx_bed = next((k for k, v in idx_to_labels.items() if "bed" in v), None)
    # print(f"TV index: {idx_tv}, Bed index: {idx_bed}")
    if idx_tv is None or idx_bed is None:
        return torch.zeros(len(object_indices), device=device)

    rewards = torch.zeros(len(object_indices), device=device)
    for b in range(len(object_indices)):
        try:
            # Get valid mask - ensure boolean tensor
            if isinstance(is_empty, torch.Tensor):
                valid_mask = ~is_empty[b]
            else:
                valid_mask = ~torch.tensor(is_empty[b], dtype=torch.bool, device=device)

            # Convert to boolean explicitly
            if isinstance(valid_mask, torch.Tensor):
                if valid_mask.dtype != torch.bool:
                    valid_mask = valid_mask.bool()
            else:
                # valid_mask is a Python bool - no valid objects
                continue

            # Check if we have any valid objects
            if valid_mask.sum().item() == 0:
                continue

            valid_indices = object_indices[b][valid_mask]
            valid_pos = positions[b][valid_mask]
            valid_orient = orientations[b][valid_mask]

            if not isinstance(valid_indices, torch.Tensor):
                continue

            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
                valid_pos = valid_pos.unsqueeze(0)
                valid_orient = valid_orient.unsqueeze(0)

            if valid_indices.numel() == 0:
                continue

            # Check for TV and bed - ensure tensor comparisons
            tv_mask = (valid_indices == idx_tv)
            bed_mask = (valid_indices == idx_bed)

            if isinstance(tv_mask, torch.Tensor):
                has_tv = tv_mask.any().item()
            else:
                has_tv = bool(tv_mask)

            if isinstance(bed_mask, torch.Tensor):
                has_bed = bed_mask.any().item()
            else:
                has_bed = bool(bed_mask)

            if not has_bed:
                rewards[b] += -5

            if not has_tv:
                rewards[b] += -5

            if not (has_tv and has_bed):
                continue

            tv_pos = valid_pos[valid_indices == idx_tv][0]
            bed_pos = valid_pos[valid_indices == idx_bed][0]

            dist = torch.norm(tv_pos - bed_pos)
            rewards[b] += torch.exp(-((dist - ideal) ** 2) / (2 * sigma**2))

            # facing reward
            bed_dir = valid_orient[valid_indices == idx_bed][0]

            # Compute direction from bed to TV (in XZ plane, ignore Y)
            dir_bed_to_tv = tv_pos - bed_pos
            # Project to 2D (XZ plane) to match orientation which is [cos, sin] in XZ
            dir_bed_to_tv_2d = torch.tensor([dir_bed_to_tv[0], dir_bed_to_tv[2]], device=device)
            dir_bed_to_tv_2d = dir_bed_to_tv_2d / (torch.norm(dir_bed_to_tv_2d) + 1e-6)

            alignment = F.cosine_similarity(bed_dir.unsqueeze(0), dir_bed_to_tv_2d.unsqueeze(0)).clamp(0, 1)
            rewards[b] += alignment.item()

        except Exception as e:
            print(f"[ERROR] reward_tv_distance batch {b}: {e}")
            continue

    return rewards


def threed_score_fn(x0_raw, save_dir, config, only_raw_scores=True):
    """Non-penetration reward for a batch of denoised 3D scenes, with history-based
    global z-score normalisation mirroring score_fn1.

    Args:
        x0_raw       : (B, N, C) final denoised scene tensor (on any device)
        save_dir     : stage save directory
        config       : OmegaConf config (needs config.midiffusion.room_type / num_classes)
        only_raw_scores: if True, skip history normalisation (FK resampling)

    Returns:
        eval_scores (B,), metrics dict, R (B,) raw rewards
    """
    from core.sampling import parse_and_descale_scenes

    unique_id   = config.exp_name
    room_type   = getattr(config.midiffusion, 'room_type',   'bedroom')
    num_classes = getattr(config.midiffusion, 'num_classes', 22)
    use_tv_bed  = getattr(config, 'tv_bed', False)

    parsed = parse_and_descale_scenes(x0_raw, num_classes=num_classes, room_type=room_type)

    if use_tv_bed:
        R = compute_tv_bed_presence_reward(parsed, room_type=room_type, idx_to_labels=_IDX_TO_LABEL_BEDROOM).cpu()  # (B,)
        scores_filename   = 'tv_bed_scores.pkl'
        history_file_name = 'history_tv_bed_scores.pkl'
    else:
        R = compute_non_penetration_reward(parsed, room_type=room_type).cpu()  # (B,)
        scores_filename   = 'penetration_scores.pkl'
        history_file_name = 'history_3d_scores.pkl'

    if only_raw_scores:
        return R, {}, R

    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', scores_filename), 'wb') as f:
        pickle.dump(R, f)

    # History-based global z-score normalisation
    history_data = []
    if config.eval.history_cnt > 0:
        history_path = os.path.join(config.save_path, unique_id, history_file_name)
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history_data = pickle.load(f)
        if len(history_data) > config.eval.history_cnt:
            history_data = history_data[-config.eval.history_cnt:]

    combined    = torch.cat(history_data + [R]) if history_data else R
    global_mean = combined.mean().item()
    global_std  = combined.std().item()

    history_data.append(R)
    if len(history_data) > config.eval.history_cnt:
        history_data = history_data[-config.eval.history_cnt:]
    with open(os.path.join(config.save_path, unique_id, history_file_name), 'wb') as f:
        pickle.dump(history_data, f)

    eval_scores = torch.tensor([(s - global_mean) / (global_std + 1e-8) for s in R]) # NOTE: this is sensible because we are not doing prompt wise alignment here, collision is similar across floor conditions

    metrics = {
        'raw_scores_mean':        float(R.mean()),
        'raw_scores_std':         float(R.std()),
        'raw_scores_min':         float(R.min()),
        'raw_scores_max':         float(R.max()),
        'normalized_scores_mean': float(eval_scores.mean()),
        'normalized_scores_std':  float(eval_scores.std()),
    }
    return eval_scores, metrics, R


def score_fn1(ground, img_dir, save_dir, config, clip_model=None, clip_preprocess=None, clip_tokenizer=None, only_raw_scores=True):
    """
    Calculate CLIP-based similarity scores for images against text prompts.
    
    Args:
        ground: List of text prompts
        img_dir: Directory containing images
        save_dir: Directory to save results
        config: Configuration object
        clip_model: Pre-loaded CLIP model (optional, will load if not provided)
        clip_preprocess: Pre-loaded CLIP preprocess function (optional)
        clip_tokenizer: Pre-loaded CLIP tokenizer (optional)
        
    Returns:
        sum_scores: Normalized scores for each image
        metrics: Dictionary with mean/std statistics per prompt
    """
    unique_id = config.exp_name
    
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    
    # Use provided CLIP components or load them (backward compatibility)
    if clip_model is None or clip_preprocess is None or clip_tokenizer is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14', 
            pretrained='laion2B-s32B-b79K'
        )
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
        model = model.to(device)
    else:
        model = clip_model
        preprocess = clip_preprocess
        tokenizer = clip_tokenizer
    
    eval_list = sorted(os.listdir(img_dir))
    
    similarity = []
    maximum_onetime = 8
    
    for i in range(0, len(eval_list), maximum_onetime):
        image_input = torch.tensor(
            np.stack([
                preprocess(Image.open(os.path.join(img_dir, image))).numpy() 
                for image in eval_list[i:i+maximum_onetime]
            ])
        ).to(device)
        text_inputs = tokenizer(ground[i:i+maximum_onetime]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Fix: use actual batch size instead of maximum_onetime to handle last batch correctly
        actual_batch_size = len(image_features)
        similarity.append(
            (image_features @ text_features.T)[
                torch.arange(actual_batch_size), 
                torch.arange(actual_batch_size)
            ]
        )
    
    similarity = torch.cat(similarity)
    R = similarity.cpu().detach()
    # print(R[:10])
    if only_raw_scores:
        return R, {}, R
    # Save raw scores
    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'scores.pkl'), 'wb') as f:
        pickle.dump(R, f)
    
    # Organize scores by prompt
    each_score = {}
    for idx, prompt in enumerate(ground):
        if prompt in each_score:
            each_score[prompt].append(R[idx:idx+1])
        else:
            each_score[prompt] = [R[idx:idx+1]]
    
    # Load and update history
    history_data = []
    if config.eval.history_cnt > 0:
        history_path = os.path.join(config.save_path, unique_id, 'history_scores.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history_data = pickle.load(f)
        if len(history_data) > config.eval.history_cnt:
            history_data = history_data[-config.eval.history_cnt:]
    
    # Calculate statistics
    data_mean = {}
    data_std = {}
    cur_data = {}
    combine_data = {}
    
    for k, v in each_score.items():
        cur_data[k] = torch.cat(v, axis=0)
        combine_data[k] = torch.cat(
            [d[k] for d in history_data if k in d] + [cur_data[k]], 
            axis=0
        )
        data_mean[k] = combine_data[k].mean().item()
        data_std[k] = combine_data[k].std().item()
    
    # Update history
    history_data.append(cur_data)
    if len(history_data) > config.eval.history_cnt:
        history_data = history_data[-config.eval.history_cnt:]
    
    with open(os.path.join(config.save_path, unique_id, 'history_scores.pkl'), 'wb') as f:
        pickle.dump(history_data, f)
    
    print(f"{data_mean=}")
    
    # Normalize scores
    if config.sample.normalize_all:
        overall_mean = R.mean().item()
        overall_std = R.std().item()
        sum_scores = [
            (s - overall_mean) / (overall_std + 1e-8) 
            for s in R
        ]
    else:
        sum_scores = [
            (s - data_mean[ground[idx]]) / (data_std[ground[idx]] + 1e-8) 
            for idx, s in enumerate(R)
        ]
    sum_scores = torch.tensor(sum_scores)
    
    # Prepare metrics for logging
    metrics = {
        'raw_scores_mean': float(R.mean()),
        'raw_scores_std': float(R.std()),
        'raw_scores_min': float(R.min()),
        'raw_scores_max': float(R.max()),
        'normalized_scores_mean': float(sum_scores.mean()),
        'normalized_scores_std': float(sum_scores.std()),
        'num_prompts': len(data_mean),
    }
    # Add per-prompt statistics
    for prompt, mean_val in data_mean.items():
        prompt_key = prompt[:50].replace(' ', '_')  # Truncate and sanitize
        metrics[f'prompt_mean/{prompt_key}'] = mean_val
        metrics[f'prompt_std/{prompt_key}'] = data_std[prompt]
    
    # Return both normalized scores (for training) and raw scores (for plotting)
    return sum_scores, metrics, R


def geometric_algebraic_score_fn(ground, img_dir, save_dir, config,
                                  num_samples=5, min_length=20.0, threshold_c=0.03,
                                  only_raw_scores=True,
                                  # unused kwargs kept for signature compatibility with score_fn1
                                  clip_model=None, clip_preprocess=None, clip_tokenizer=None):
    """
    Geometric reward function based on algebraic vanishing-point intersection quality.
    Drop-in replacement for score_fn1 — same call signature and return contract.

    Args:
        ground: list[str] — text prompts, one per image (same order as sorted img_dir)
        img_dir: str — directory of generated PNG images
        save_dir: str — directory to save intermediate results
        config: OmegaConf config object
        num_samples: number of line samples for geometric reward
        min_length: minimum line length threshold
        threshold_c: algebraic intersection threshold
        only_raw_scores: if True, skip history-based normalisation (used during FK resampling)

    Returns:
        sum_scores: torch.Tensor (N,) — z-score normalised advantages for training
        metrics:    dict            — logging statistics
        R:          torch.Tensor (N,) — raw scalar rewards per image
    """
    

    reward_calculator = ImageGeometricReward(min_length=min_length)

    eval_list = sorted(os.listdir(img_dir))

    raw_rewards = []
    for img_name in eval_list:
        img_path = os.path.join(img_dir, img_name)
        try:
            # Load as HxWx3 uint8 numpy array (what ImageGeometricReward expects)
            image = np.array(Image.open(img_path).convert("RGB"))
            reward = reward_calculator.get_algebraic_intersection_reward(
                image, num_samples=num_samples, threshold_c=threshold_c
            )
        except Exception:
            reward = 0.0
        raw_rewards.append(float(reward))

    R = torch.tensor(raw_rewards, dtype=torch.float32)

    if only_raw_scores:
        return R, {}, R

    # ---- history-based per-prompt z-score normalisation (mirrors score_fn1) ----
    unique_id = config.exp_name
    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'geometric_scores.pkl'), 'wb') as f:
        pickle.dump(R, f)

    # Organise by prompt
    each_score = {}
    for idx, prompt in enumerate(ground):
        each_score.setdefault(prompt, []).append(R[idx:idx + 1])

    # Load and update history
    history_data = []
    if config.eval.history_cnt > 0:
        history_path = os.path.join(config.save_path, unique_id, 'history_geometric_scores.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history_data = pickle.load(f)
        if len(history_data) > config.eval.history_cnt:
            history_data = history_data[-config.eval.history_cnt:]

    data_mean, data_std, cur_data, combine_data = {}, {}, {}, {}
    for k, v in each_score.items():
        cur_data[k] = torch.cat(v, dim=0)
        combine_data[k] = torch.cat(
            [d[k] for d in history_data if k in d] + [cur_data[k]], dim=0
        )
        data_mean[k] = combine_data[k].mean().item()
        data_std[k] = combine_data[k].std().item()

    history_data.append(cur_data)
    if len(history_data) > config.eval.history_cnt:
        history_data = history_data[-config.eval.history_cnt:]
    with open(os.path.join(config.save_path, unique_id, 'history_geometric_scores.pkl'), 'wb') as f:
        pickle.dump(history_data, f)

    if config.sample.normalize_all:
        overall_mean = R.mean().item()
        overall_std = R.std().item()
        sum_scores = [(s - overall_mean) / (overall_std + 1e-8) for s in R]
    else:
        sum_scores = [
            (s - data_mean[ground[idx]]) / (data_std[ground[idx]] + 1e-8)
            for idx, s in enumerate(R)
        ]
    sum_scores = torch.tensor(sum_scores)

    metrics = {
        'raw_scores_mean': float(R.mean()),
        'raw_scores_std': float(R.std()),
        'raw_scores_min': float(R.min()),
        'raw_scores_max': float(R.max()),
        'normalized_scores_mean': float(sum_scores.mean()),
        'normalized_scores_std': float(sum_scores.std()),
        'num_prompts': len(data_mean),
    }
    for prompt, mean_val in data_mean.items():
        prompt_key = prompt[:50].replace(' ', '_')
        metrics[f'prompt_mean/{prompt_key}'] = mean_val
        metrics[f'prompt_std/{prompt_key}'] = data_std[prompt]

    return sum_scores, metrics, R



def run_selection(config, stage_idx=None, logger=None, wandb_run=None):
    """
    Run the selection phase for a given stage.
    
    Args:
        config: Configuration object (can be OmegaConf or dict)
        stage_idx: Current stage index (optional, for logging)
        logger: Logger instance (optional)
        wandb_run: Existing wandb run to log to (optional)
        
    Returns:
        save_dir: Directory where selected samples were saved
        metrics: Dictionary of selection metrics for logging
    """
    if logger:
        logger.info(f"Starting reward calculation for stage {stage_idx}")
    else:
        print(f"Starting reward calculation for stage {stage_idx}")
    
    torch.cuda.set_device(config.dev_id)
    seed_everything(config.seed)
    
    unique_id = config.exp_name
    stage_id = f"stage{stage_idx}"
    save_dir = os.path.join(config.save_path, unique_id, stage_id)
    
    # Load samples
    with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as f:
        samples = pickle.load(f)

    threed = getattr(config, 'threed_scene_layout', False)

    if threed:
        # 3D path: floor condition indices as context, non-penetration reward
        with open(os.path.join(save_dir, 'fpbpn_list.json'), 'r') as f:
            ground = json.load(f)

        device  = f"cuda:{config.dev_id}" if torch.cuda.is_available() else "cpu"
        x0_raw  = samples["next_scenes"][:, -1].to(device)  # (B, N, C) final denoised step
        eval_scores, score_metrics, raw_scores = threed_score_fn(
            x0_raw, save_dir, config, only_raw_scores=False
        )
        _embed_key    = 'fpbpn'
        _lat_key      = 'scenes'
        _next_lat_key = 'next_scenes'
        print(f"3D non-penetration rewards: mean={raw_scores.mean():.4f}, std={raw_scores.std():.4f}")
    else:
        # SD path: text prompts, CLIP/geometric reward
        with open(os.path.join(save_dir, 'prompt.json'), 'r') as f:
            ground = json.load(f)

        img_dir        = os.path.join(save_dir, 'images')
        reward_fn_name = getattr(config, 'reward_fn', 'clip')
        if reward_fn_name == 'geometric':
            eval_scores, score_metrics, raw_scores = geometric_algebraic_score_fn(
                ground, img_dir, save_dir, config, only_raw_scores=False
            )
        else:
            eval_scores, score_metrics, raw_scores = score_fn1(
                ground, img_dir, save_dir, config, only_raw_scores=False
            )
        _embed_key    = 'prompt_embeds'
        _lat_key      = 'latents'
        _next_lat_key = 'next_latents'

    raw_clip_scores = raw_scores  # alias kept for rest of function
    print(f"{raw_clip_scores=}, dtype of eval score tensor: {eval_scores.dtype}")
    print(f"{eval_scores=}")
    samples['eval_scores'] = eval_scores  # Normalized scores for training

    if config.sample.save_train_samples_no_train:
        all_scores_list = [float(s) for s in raw_clip_scores]
        rewards_key     = 'penetration_reward' if threed else 'clip_reward'
        mean_rewards_summary = {
            f"{rewards_key}_mean": float(raw_clip_scores.mean()),
            f"{rewards_key}_std":  float(raw_clip_scores.std()),
            f"{rewards_key}_min":  float(raw_clip_scores.min()),
            f"{rewards_key}_max":  float(raw_clip_scores.max()),
            "all_scores": all_scores_list,
        }
        rewards_fname = 'penetration_rewards.json' if threed else 'clip_rewards.json'
        rewards_path  = os.path.join(save_dir, rewards_fname)
        with open(rewards_path, 'w') as f:
            json.dump(mean_rewards_summary, f, indent=2)
        if logger:
            logger.info(f"Reward statistics saved to {rewards_path}")
        else:
            print(f"Reward statistics saved to {rewards_path}")
        import sys; sys.exit(0)

    # Initialize data structure for selected samples
    def get_new_unit():
        return {
            _embed_key:    [],
            'timesteps':   [],
            'log_probs':   [],
            _lat_key:      [],
            _next_lat_key: [],
            'eval_scores': [],
        }

    data = get_new_unit()
    if config.sample.no_selection:
        t_left  = 0
        t_right = config.sample.num_steps
        data = {
            _embed_key:    samples[_embed_key],
            'timesteps':   samples['timesteps'][:, t_left:t_right],
            'log_probs':   samples['log_probs'][:, t_left:t_right],
            _lat_key:      samples[_lat_key][:, t_left:t_right],
            _next_lat_key: samples[_next_lat_key][:, t_left:t_right],
            'eval_scores': samples['eval_scores'],
        }

    else:  # Select positive and negative samples
        total_batch_size = samples['eval_scores'].shape[0]

        if config.sample.fk:
            fk_particles = config.sample.num_particles * (1 if getattr(config.sample, 'only_best_fk', False) else 2)
            data_size    = fk_particles
            batch_size   = total_batch_size // fk_particles
        else:
            data_size  = total_batch_size // config.sample.batch_size
            batch_size = config.sample.batch_size

        for b in range(batch_size):
            cur_sample_num = 1

            if config.sample.fk:
                start_idx     = b * fk_particles
                end_idx       = start_idx + fk_particles
                batch_samples = {k: v[start_idx:end_idx] for k, v in samples.items()}
            else:
                batch_samples = {
                    k: v[torch.arange(b, total_batch_size, batch_size)]
                    for k, v in samples.items()
                }

            if (hasattr(config, 'train') and getattr(config.train, 'incremental_training', False)) or config.sample.fk:
                t_left  = 0
                t_right = config.sample.num_steps
            else:
                t_left  = config.sample.num_steps - config.split_step
                t_right = config.sample.num_steps

            if config.train.only_last_n_steps > 0:
                t_left = config.sample.num_steps - config.train.only_last_n_steps

            idx_sel   = torch.arange(0, data_size, cur_sample_num)
            embeds    = batch_samples[_embed_key][idx_sel]
            timesteps = batch_samples['timesteps'][idx_sel, t_left:t_right]
            log_probs = batch_samples['log_probs'][idx_sel, t_left:t_right]
            lats      = batch_samples[_lat_key][idx_sel, t_left:t_right]
            next_lats = batch_samples[_next_lat_key][idx_sel, t_left:t_right]

            score = batch_samples['eval_scores'][idx_sel]
            print(f"Batch {b}: score shape before reshape: {score.shape}")

            if config.sample.fk:
                score = score.reshape(1, -1)
            else:
                score = score.reshape(-1, config.split_time)
            max_idx = score.argmax(dim=1)
            min_idx = score.argmin(dim=1)

            for j, s in enumerate(score):
                for p_n in range(2):
                    if p_n == 0 and s[max_idx[j]] >= config.eval.pos_threshold:
                        used_idx   = max_idx[j]
                        used_idx_2 = used_idx if config.sample.fk else j * config.split_time + used_idx
                    elif p_n == 1 and s[min_idx[j]] < config.eval.neg_threshold:
                        used_idx   = min_idx[j]
                        used_idx_2 = used_idx if config.sample.fk else j * config.split_time + used_idx
                    else:
                        if config.sample.no_selection:
                            used_idx   = min_idx[j]
                            used_idx_2 = used_idx if config.sample.fk else j * config.split_time + used_idx
                        else:
                            continue

                    data[_embed_key].append(embeds[used_idx_2])
                    data['timesteps'].append(timesteps[used_idx_2])
                    data['log_probs'].append(log_probs[used_idx_2])
                    data[_lat_key].append(lats[used_idx_2])
                    data[_next_lat_key].append(next_lats[used_idx_2])
                    data['eval_scores'].append(s[used_idx])

            if not config.sample.fk:
                cur_sample_num *= config.split_time

    # Stack data if any samples were selected
    if len(data[_embed_key]) > 0:
        first_value = next(iter(data.values()))
        if isinstance(first_value, list):
            data = {k: torch.stack(v, dim=0) for k, v in data.items()}
        if logger:
            logger.info(f"Selected {len(data[_embed_key])} samples")
    else:
        if logger:
            logger.warning("No samples met the selection criteria")
        print("Warning: No samples met the selection criteria")
        # Convert all empty lists to empty tensors so downstream code can call
        # .shape[0], .dtype, etc. without crashing.
        data = {k: (torch.stack(v, dim=0) if (isinstance(v, list) and len(v) > 0) else torch.tensor([])) for k, v in data.items()}

    # Save selected samples
    with open(os.path.join(save_dir, 'sample_stage.pkl'), 'wb') as f:
        pickle.dump(data, f)

    all_samples_mean_reward = float(raw_clip_scores.mean())
    all_samples_std_reward  = float(raw_clip_scores.std())
    num_reward_queries      = len(raw_clip_scores)

    cumulative_queries_path = os.path.join(config.save_path, unique_id, 'cumulative_reward_queries.pkl')
    if os.path.exists(cumulative_queries_path):
        with open(cumulative_queries_path, 'rb') as f:
            cumulative_reward_queries = pickle.load(f)
    else:
        cumulative_reward_queries = 0
    cumulative_reward_queries += num_reward_queries
    with open(cumulative_queries_path, 'wb') as f:
        pickle.dump(cumulative_reward_queries, f)

    print(f"dtype of eval_scores: {data.get('eval_scores', torch.tensor([])).dtype if isinstance(data.get('eval_scores'), torch.Tensor) else 'list (no samples selected)'}")
    _es_raw = data.get('eval_scores', [])
    if isinstance(_es_raw, list):
        _es = torch.stack(_es_raw) if len(_es_raw) > 0 else torch.tensor([])
    else:
        _es = _es_raw
    num_selected  = len(data.get(_embed_key, []))
    num_positive  = int((_es >= config.eval.pos_threshold).sum()) if len(_es) > 0 else 0
    num_negative  = int((_es <  config.eval.neg_threshold).sum()) if len(_es) > 0 else 0
    num_generated = len(raw_clip_scores)
    num_rejected  = num_generated - num_selected

    selection_metrics = {
        "num_selected":       num_selected,
        "num_positive":       num_positive,
        "num_negative":       num_negative,
        "num_generated":      num_generated,
        "num_rejected":       num_rejected,
        "mean_reward":        all_samples_mean_reward,
        "std_reward":         all_samples_std_reward,
        "num_queries":        num_reward_queries,
        "cumulative_queries": cumulative_reward_queries,
    }

    if logger:
        logger.info(f"Selection completed for stage {stage_idx}")
        logger.info(f"Generated: {num_generated}, Selected: {num_selected}, Rejected: {num_rejected}")
        logger.info(f"Positive: {num_positive}, Negative: {num_negative}")
        logger.info(f"Mean reward (all samples): {all_samples_mean_reward:.4f} ± {all_samples_std_reward:.4f}")
        logger.info(f"Cumulative reward queries: {cumulative_reward_queries}")

    return save_dir, selection_metrics
