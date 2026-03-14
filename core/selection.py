"""
Core selection module for B2Diff training pipeline.
Extracts the selection logic from run_select.py into a callable function.
"""

import os
import importlib
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
from core.universal_rewards.not_out_of_bound_reward import (
    compute_boundary_violation_reward,
    SDFCache,
)
from core.universal_rewards.accessibility_reward import (
    compute_accessibility_reward,
    AccessibilityCache,
)
from core.universal_rewards.penetration_reward import compute_non_penetration_reward
from core.universal_rewards.object_count_reward import compute_object_count_reward

tqdm = partial(tqdm_lib, dynamic_ncols=True)
_SDF_CACHE = {}
_ACCESSIBILITY_CACHE = {}
_ROOM_FEATURES_INDEX = {}
_FLOOR_POLYGONS_CACHE = {}
_CUSTOM_REWARD_FN_CACHE = {}


def _get_sdf_cache(config):
    cache_dir = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_sdf_cache/"
    grid_resolution = 0.05
    split = "test"
    key = (cache_dir, grid_resolution, split)
    if key not in _SDF_CACHE:
        _SDF_CACHE[key] = SDFCache(cache_dir, grid_resolution=grid_resolution, split=split)
    return _SDF_CACHE[key]


def _get_floor_polygons_by_condition(config):
    path = getattr(getattr(config, "midiffusion", None), "floor_polygons_path", None)
    if not path:
        raise ValueError("midiffusion.floor_polygons_path must be set to use boundary reward.")

    if path in _FLOOR_POLYGONS_CACHE:
        return _FLOOR_POLYGONS_CACHE[path]

    with open(path, "r") as f:
        floor_polygons = json.load(f)

    _FLOOR_POLYGONS_CACHE[path] = floor_polygons
    return floor_polygons


def _get_accessibility_cache(config):
    cache_dir = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/accessibility_cache"
    grid_resolution = 0.1
    split = "test"
    key = (cache_dir, grid_resolution, split)
    if key not in _ACCESSIBILITY_CACHE:
        _ACCESSIBILITY_CACHE[key] = AccessibilityCache(
            cache_dir, grid_resolution=grid_resolution, split=split
        )
    return _ACCESSIBILITY_CACHE[key]


def _flatten_fpbpn_indices(fpbpn_list):
    if not fpbpn_list:
        return []
    if isinstance(fpbpn_list[0], list):
        return [int(i) for batch in fpbpn_list for i in batch]
    return [int(i) for i in fpbpn_list]


def _fpbpn_key(fpbpn, scale=1e6):
    arr = np.asarray(fpbpn, dtype=np.float64).reshape(-1)
    return tuple(np.round(arr * scale).astype(np.int64).tolist())


def _get_room_features_index_map(config, scale=1e6):
    path = getattr(getattr(config, "midiffusion", None), "floor_conditions", None)
    if not path:
        raise ValueError("midiffusion.floor_conditions must be set to map fpbpn to indices.")

    key = (path, scale)
    if key in _ROOM_FEATURES_INDEX:
        return _ROOM_FEATURES_INDEX[key]

    with open(path, "r") as f:
        room_features = json.load(f)

    index_map = {}
    for idx, fpbpn in enumerate(room_features):
        index_map[_fpbpn_key(fpbpn, scale=scale)] = idx

    _ROOM_FEATURES_INDEX[key] = index_map
    return index_map


def _resolve_fpbpn_indices(fpbpn_list, config):
    if not fpbpn_list:
        return []

    first = fpbpn_list[0]
    if isinstance(first, list):
        if first and isinstance(first[0], list):
            if first[0] and isinstance(first[0][0], (int, float)):
                index_map = _get_room_features_index_map(config)
                indices = []
                for fpbpn in fpbpn_list:
                    key = _fpbpn_key(fpbpn)
                    if key not in index_map:
                        raise ValueError("Could not find fpbpn entry in room_features.json.")
                    indices.append(index_map[key])
                return indices
        # batch of indices
        if first and isinstance(first[0], (int, float)):
            return [int(i) for batch in fpbpn_list for i in batch]
    if isinstance(first, (int, float)):
        return [int(i) for i in fpbpn_list]

    raise ValueError("Unsupported fpbpn_list format; cannot resolve indices.")


def _load_custom_reward_fn(custom_reward_name):
    """Load a custom reward function from core/custom_rewards/<name>.py."""
    if custom_reward_name in _CUSTOM_REWARD_FN_CACHE:
        return _CUSTOM_REWARD_FN_CACHE[custom_reward_name]

    module_name = f"core.custom_rewards.{custom_reward_name}"
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        raise ValueError(
            f"Could not import custom reward module '{module_name}': {e}"
        ) from e

    # Preferred explicit entrypoint
    if hasattr(module, "compute_reward") and callable(module.compute_reward):
        fn = module.compute_reward
        _CUSTOM_REWARD_FN_CACHE[custom_reward_name] = fn
        return fn

    # Backward-compatible naming candidates
    candidate_names = [
        f"compute_{custom_reward_name}_reward",
        f"compute_{custom_reward_name}_presence_reward",
    ]
    for name in candidate_names:
        if hasattr(module, name):
            candidate_fn = getattr(module, name)
            if callable(candidate_fn):
                _CUSTOM_REWARD_FN_CACHE[custom_reward_name] = candidate_fn
                return candidate_fn

    # Last resort: unique callable starting with 'compute_'
    compute_like_fns = [
        getattr(module, attr)
        for attr in dir(module)
        if attr.startswith("compute_") and callable(getattr(module, attr))
    ]
    if len(compute_like_fns) == 1:
        fn = compute_like_fns[0]
        _CUSTOM_REWARD_FN_CACHE[custom_reward_name] = fn
        return fn

    raise ValueError(
        f"No valid reward function found in {module_name}. "
        "Define compute_reward(parsed_scene, **kwargs), or a uniquely identifiable compute_* function."
    )

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





def _compute_threed_reward_components(parsed, config, room_type, indices=None, floor_polygons=None):
    """Compute 3D reward components and return total reward plus per-component tensors."""
    use_universal = getattr(config, "universal_rewards", False)
    custom_reward = getattr(config, "custom_reward", None)
    use_tv_bed = getattr(config, "tv_bed", False)
    threed_reward = getattr(config, "threed_reward", None)
    object_count_mode = getattr(config, "object_count_mode", "nll")

    components = {}

    # Custom-only mode: when universal rewards are disabled but a custom reward is set,
    # use only that custom reward.
    if (not use_universal) and (custom_reward is not None):
        custom_reward_fn = _load_custom_reward_fn(custom_reward)
        components[f"custom_{custom_reward}"] = custom_reward_fn(
            parsed, room_type=room_type, idx_to_labels=_IDX_TO_LABEL_BEDROOM
        )
        total_reward = list(components.values())[0]
        return total_reward, components, f"custom_{custom_reward}"

    if use_universal:
        if indices is None:
            raise ValueError("indices must be provided when universal_rewards=true.")

        accessibility_cache = _get_accessibility_cache(config)
        components["penetration"] = compute_non_penetration_reward(
            parsed, room_type=room_type
        )
        components["boundary"] = compute_boundary_violation_reward(
            parsed,
            floor_polygons=floor_polygons,
        )
        components["object_count"] = compute_object_count_reward(
            parsed, mode=object_count_mode
        )
        components["accessibility"] = compute_accessibility_reward(
            parsed,
            indices=indices,
            accessibility_cache=accessibility_cache,
            room_type=room_type,
        )

        # Custom reward can be optionally added on top of universal rewards.
        if custom_reward is not None:
            custom_reward_fn = _load_custom_reward_fn(custom_reward)
            components[f"custom_{custom_reward}"] = custom_reward_fn(
                parsed, room_type=room_type, idx_to_labels=_IDX_TO_LABEL_BEDROOM
            )

        total_reward = sum(components.values())
        return total_reward, components, "universal"

    # Backward-compatible single reward mode (legacy behavior)
    if threed_reward is None:
        reward_mode = "tv_bed" if use_tv_bed else "penetration"
    else:
        reward_mode = threed_reward

    if reward_mode == "tv_bed":
        tv_bed_reward_fn = _load_custom_reward_fn("tv_bed")
        components["tv_bed"] = tv_bed_reward_fn(
            parsed, room_type=room_type, idx_to_labels=_IDX_TO_LABEL_BEDROOM
        )
    elif reward_mode == "boundary":
        if floor_polygons is None:
            raise ValueError("floor_polygons must be provided when threed_reward='boundary'.")
        components["boundary"] = compute_boundary_violation_reward(
            parsed,
            floor_polygons=floor_polygons,
        )
    elif reward_mode == "object_count":
        components["object_count"] = compute_object_count_reward(
            parsed, mode=object_count_mode
        )
    else:
        components["penetration"] = compute_non_penetration_reward(
            parsed, room_type=room_type
        )

    total_reward = list(components.values())[0]
    return total_reward, components, reward_mode


def threed_score_fn(x0_raw, save_dir, config, indices=None, floor_polygons=None, only_raw_scores=True):
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
    with torch.no_grad():
        parsed = parse_and_descale_scenes(
            x0_raw, num_classes=num_classes, room_type=room_type
        )
        R, reward_components, reward_mode = _compute_threed_reward_components(
            parsed,
            config,
            room_type=room_type,
            indices=indices,
            floor_polygons=floor_polygons,
        )
        R = R.cpu()

    if reward_mode == "universal":
        scores_filename = "universal_scores.pkl"
        history_file_name = "history_universal_scores.pkl"
    elif reward_mode.startswith("custom_"):
        custom_name = reward_mode[len("custom_"):]
        scores_filename = f"custom_{custom_name}_scores.pkl"
        history_file_name = f"history_custom_{custom_name}_scores.pkl"
    elif reward_mode == "tv_bed":
        scores_filename = "tv_bed_scores.pkl"
        history_file_name = "history_tv_bed_scores.pkl"
    elif reward_mode == "boundary":
        scores_filename = "boundary_scores.pkl"
        history_file_name = "history_boundary_scores.pkl"
    elif reward_mode == "object_count":
        scores_filename = "object_count_scores.pkl"
        history_file_name = "history_object_count_scores.pkl"
    else:
        scores_filename = "penetration_scores.pkl"
        history_file_name = "history_3d_scores.pkl"

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
    # Explicit total composed raw mean (sum of universal components + optional custom).
    metrics['component/total_raw_mean'] = float(R.mean())
    for component_name, component_values in reward_components.items():
        component_values = component_values.detach().cpu()
        metrics[f'component/{component_name}_mean'] = float(component_values.mean())
        metrics[f'component/{component_name}_std'] = float(component_values.std())
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
        indices = _resolve_fpbpn_indices(ground, config)
        if len(indices) != x0_raw.shape[0]:
            raise ValueError(
                f"fpbpn_list length ({len(indices)}) does not match samples ({x0_raw.shape[0]})."
            )
        floor_polygons_by_condition = _get_floor_polygons_by_condition(config)
        floor_polygons = [floor_polygons_by_condition[int(i)] for i in indices]
        if any(poly is None for poly in floor_polygons):
            raise ValueError("Missing floor polygons for one or more resolved condition indices.")
        eval_scores, score_metrics, raw_scores = threed_score_fn(
            x0_raw,
            save_dir,
            config,
            indices=indices,
            floor_polygons=floor_polygons,
            only_raw_scores=False,
        )
        _embed_key    = 'fpbpn'
        _lat_key      = 'scenes'
        _next_lat_key = 'next_scenes'
        if getattr(config, 'universal_rewards', False):
            reward_mode = 'universal'
            if getattr(config, 'custom_reward', None) is not None:
                reward_mode = f"{reward_mode}+{config.custom_reward}"
        elif getattr(config, 'custom_reward', None) is not None:
            reward_mode = f"custom_{config.custom_reward}"
        else:
            reward_mode = getattr(config, 'threed_reward', 'penetration')
        print(f"3D {reward_mode} rewards: mean={raw_scores.mean():.4f}, std={raw_scores.std():.4f}")
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
        rewards_key     = 'threed_reward' if threed else 'clip_reward'
        mean_rewards_summary = {
            f"{rewards_key}_mean": float(raw_clip_scores.mean()),
            f"{rewards_key}_std":  float(raw_clip_scores.std()),
            f"{rewards_key}_min":  float(raw_clip_scores.min()),
            f"{rewards_key}_max":  float(raw_clip_scores.max()),
            "all_scores": all_scores_list,
        }
        rewards_fname = 'threed_rewards.json' if threed else 'clip_rewards.json'
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

    # Carry score-level metrics (including per-component raw means) to pipeline logging.
    for metric_name, metric_value in score_metrics.items():
        if isinstance(metric_value, (int, float)):
            selection_metrics[f"score/{metric_name}"] = float(metric_value)

    if logger:
        logger.info(f"Selection completed for stage {stage_idx}")
        logger.info(f"Generated: {num_generated}, Selected: {num_selected}, Rejected: {num_rejected}")
        logger.info(f"Positive: {num_positive}, Negative: {num_negative}")
        logger.info(f"Mean reward (all samples): {all_samples_mean_reward:.4f} ± {all_samples_std_reward:.4f}")
        logger.info(f"Cumulative reward queries: {cumulative_reward_queries}")

    return save_dir, selection_metrics


# ---
# """
# Core selection module for B2Diff training pipeline.
# Extracts the selection logic from run_select.py into a callable function.
# """

# import os
# import importlib
# import torch
# import torch.nn.functional as F
# import pickle
# import json
# import numpy as np
# from PIL import Image
# from functools import partial
# from tqdm import tqdm as tqdm_lib
# from bert_score import score, BERTScorer
# import open_clip
# from utils.utils import seed_everything
# from core.utils.geometric_rewards import ImageGeometricReward
# from core.universal_rewards.not_out_of_bound_reward import (
#     compute_boundary_violation_reward,
#     SDFCache,
# )
# from core.universal_rewards.penetration_reward import compute_non_penetration_reward
# from core.universal_rewards.object_count_reward import compute_object_count_reward

# tqdm = partial(tqdm_lib, dynamic_ncols=True)
# _SDF_CACHE = {}
# _ROOM_FEATURES_INDEX = {}
# _CUSTOM_REWARD_FN_CACHE = {}


# def _get_sdf_cache(config):
#     cache_dir = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_sdf_cache/"
#     grid_resolution = 0.05
#     split = "test"
#     key = (cache_dir, grid_resolution, split)
#     if key not in _SDF_CACHE:
#         _SDF_CACHE[key] = SDFCache(cache_dir, grid_resolution=grid_resolution, split=split)
#     return _SDF_CACHE[key]


# def _flatten_fpbpn_indices(fpbpn_list):
#     if not fpbpn_list:
#         return []
#     if isinstance(fpbpn_list[0], list):
#         return [int(i) for batch in fpbpn_list for i in batch]
#     return [int(i) for i in fpbpn_list]


# def _fpbpn_key(fpbpn, scale=1e6):
#     arr = np.asarray(fpbpn, dtype=np.float64).reshape(-1)
#     return tuple(np.round(arr * scale).astype(np.int64).tolist())


# def _get_room_features_index_map(config, scale=1e6):
#     path = getattr(getattr(config, "midiffusion", None), "floor_conditions", None)
#     if not path:
#         raise ValueError("midiffusion.floor_conditions must be set to map fpbpn to indices.")

#     key = (path, scale)
#     if key in _ROOM_FEATURES_INDEX:
#         return _ROOM_FEATURES_INDEX[key]

#     with open(path, "r") as f:
#         room_features = json.load(f)

#     index_map = {}
#     for idx, fpbpn in enumerate(room_features):
#         index_map[_fpbpn_key(fpbpn, scale=scale)] = idx

#     _ROOM_FEATURES_INDEX[key] = index_map
#     return index_map


# def _resolve_fpbpn_indices(fpbpn_list, config):
#     if not fpbpn_list:
#         return []

#     first = fpbpn_list[0]
#     if isinstance(first, list):
#         if first and isinstance(first[0], list):
#             if first[0] and isinstance(first[0][0], (int, float)):
#                 index_map = _get_room_features_index_map(config)
#                 indices = []
#                 for fpbpn in fpbpn_list:
#                     key = _fpbpn_key(fpbpn)
#                     if key not in index_map:
#                         raise ValueError("Could not find fpbpn entry in room_features.json.")
#                     indices.append(index_map[key])
#                 return indices
#         # batch of indices
#         if first and isinstance(first[0], (int, float)):
#             return [int(i) for batch in fpbpn_list for i in batch]
#     if isinstance(first, (int, float)):
#         return [int(i) for i in fpbpn_list]

#     raise ValueError("Unsupported fpbpn_list format; cannot resolve indices.")


# def _load_custom_reward_fn(custom_reward_name):
#     """Load a custom reward function from core/custom_rewards/<name>.py."""
#     if custom_reward_name in _CUSTOM_REWARD_FN_CACHE:
#         return _CUSTOM_REWARD_FN_CACHE[custom_reward_name]

#     module_name = f"core.custom_rewards.{custom_reward_name}"
#     try:
#         module = importlib.import_module(module_name)
#     except Exception as e:
#         raise ValueError(
#             f"Could not import custom reward module '{module_name}': {e}"
#         ) from e

#     # Preferred explicit entrypoint
#     if hasattr(module, "compute_reward") and callable(module.compute_reward):
#         fn = module.compute_reward
#         _CUSTOM_REWARD_FN_CACHE[custom_reward_name] = fn
#         return fn

#     # Backward-compatible naming candidates
#     candidate_names = [
#         f"compute_{custom_reward_name}_reward",
#         f"compute_{custom_reward_name}_presence_reward",
#     ]
#     for name in candidate_names:
#         if hasattr(module, name):
#             candidate_fn = getattr(module, name)
#             if callable(candidate_fn):
#                 _CUSTOM_REWARD_FN_CACHE[custom_reward_name] = candidate_fn
#                 return candidate_fn

#     # Last resort: unique callable starting with 'compute_'
#     compute_like_fns = [
#         getattr(module, attr)
#         for attr in dir(module)
#         if attr.startswith("compute_") and callable(getattr(module, attr))
#     ]
#     if len(compute_like_fns) == 1:
#         fn = compute_like_fns[0]
#         _CUSTOM_REWARD_FN_CACHE[custom_reward_name] = fn
#         return fn

#     raise ValueError(
#         f"No valid reward function found in {module_name}. "
#         "Define compute_reward(parsed_scene, **kwargs), or a uniquely identifiable compute_* function."
#     )

# # Bedroom object index lookup (shared across reward functions)
# _IDX_TO_LABEL_BEDROOM = {
#     0: "armchair", 1: "bookshelf", 2: "cabinet", 3: "ceiling_lamp",
#     4: "chair", 5: "children_cabinet", 6: "coffee_table", 7: "desk",
#     8: "double_bed", 9: "dressing_chair", 10: "dressing_table",
#     11: "kids_bed", 12: "nightstand", 13: "pendant_lamp", 14: "shelf",
#     15: "single_bed", 16: "sofa", 17: "stool", 18: "table",
#     19: "tv_stand", 20: "wardrobe",
# }
# _TV_STAND_IDX = 19
# _BED_INDICES  = {8}  # double_bed, kids_bed, single_bed





# def _compute_threed_reward_components(parsed, config, room_type, indices=None):
#     """Compute 3D reward components and return total reward plus per-component tensors."""
#     use_universal = getattr(config, "universal_rewards", False)
#     custom_reward = getattr(config, "custom_reward", None)
#     use_tv_bed = getattr(config, "tv_bed", False)
#     threed_reward = getattr(config, "threed_reward", None)
#     object_count_mode = getattr(config, "object_count_mode", "nll")

#     components = {}

#     # Custom-only mode: when universal rewards are disabled but a custom reward is set,
#     # use only that custom reward.
#     if (not use_universal) and (custom_reward is not None):
#         custom_reward_fn = _load_custom_reward_fn(custom_reward)
#         components[f"custom_{custom_reward}"] = custom_reward_fn(
#             parsed, room_type=room_type, idx_to_labels=_IDX_TO_LABEL_BEDROOM
#         )
#         total_reward = list(components.values())[0]
#         return total_reward, components, f"custom_{custom_reward}"

#     if use_universal:
#         if indices is None:
#             raise ValueError("indices must be provided when universal_rewards=true.")

#         sdf_cache = _get_sdf_cache(config)
#         components["penetration"] = compute_non_penetration_reward(
#             parsed, room_type=room_type
#         )
#         components["boundary"] = compute_boundary_violation_reward(
#             parsed, indices=indices, sdf_cache=sdf_cache
#         )
#         components["object_count"] = compute_object_count_reward(
#             parsed, mode=object_count_mode
#         )

#         # Custom reward can be optionally added on top of universal rewards.
#         if custom_reward is not None:
#             custom_reward_fn = _load_custom_reward_fn(custom_reward)
#             components[f"custom_{custom_reward}"] = custom_reward_fn(
#                 parsed, room_type=room_type, idx_to_labels=_IDX_TO_LABEL_BEDROOM
#             )

#         total_reward = sum(components.values())
#         return total_reward, components, "universal"

#     # Backward-compatible single reward mode (legacy behavior)
#     if threed_reward is None:
#         reward_mode = "tv_bed" if use_tv_bed else "penetration"
#     else:
#         reward_mode = threed_reward

#     if reward_mode == "tv_bed":
#         tv_bed_reward_fn = _load_custom_reward_fn("tv_bed")
#         components["tv_bed"] = tv_bed_reward_fn(
#             parsed, room_type=room_type, idx_to_labels=_IDX_TO_LABEL_BEDROOM
#         )
#     elif reward_mode == "boundary":
#         if indices is None:
#             raise ValueError("indices must be provided when threed_reward='boundary'.")
#         sdf_cache = _get_sdf_cache(config)
#         components["boundary"] = compute_boundary_violation_reward(
#             parsed, indices=indices, sdf_cache=sdf_cache
#         )
#     elif reward_mode == "object_count":
#         components["object_count"] = compute_object_count_reward(
#             parsed, mode=object_count_mode
#         )
#     else:
#         components["penetration"] = compute_non_penetration_reward(
#             parsed, room_type=room_type
#         )

#     total_reward = list(components.values())[0]
#     return total_reward, components, reward_mode


# def threed_score_fn(x0_raw, save_dir, config, indices=None, only_raw_scores=True):
#     """Non-penetration reward for a batch of denoised 3D scenes, with history-based
#     global z-score normalisation mirroring score_fn1.

#     Args:
#         x0_raw       : (B, N, C) final denoised scene tensor (on any device)
#         save_dir     : stage save directory
#         config       : OmegaConf config (needs config.midiffusion.room_type / num_classes)
#         only_raw_scores: if True, skip history normalisation (FK resampling)

#     Returns:
#         eval_scores (B,), metrics dict, R (B,) raw rewards
#     """
#     from core.sampling import parse_and_descale_scenes

#     unique_id   = config.exp_name
#     room_type   = getattr(config.midiffusion, 'room_type',   'bedroom')
#     num_classes = getattr(config.midiffusion, 'num_classes', 22)
#     with torch.no_grad():
#         parsed = parse_and_descale_scenes(
#             x0_raw, num_classes=num_classes, room_type=room_type
#         )
#         R, reward_components, reward_mode = _compute_threed_reward_components(
#             parsed,
#             config,
#             room_type=room_type,
#             indices=indices,
#         )
#         R = R.cpu()

#     if reward_mode == "universal":
#         scores_filename = "universal_scores.pkl"
#         history_file_name = "history_universal_scores.pkl"
#     elif reward_mode.startswith("custom_"):
#         custom_name = reward_mode[len("custom_"):]
#         scores_filename = f"custom_{custom_name}_scores.pkl"
#         history_file_name = f"history_custom_{custom_name}_scores.pkl"
#     elif reward_mode == "tv_bed":
#         scores_filename = "tv_bed_scores.pkl"
#         history_file_name = "history_tv_bed_scores.pkl"
#     elif reward_mode == "boundary":
#         scores_filename = "boundary_scores.pkl"
#         history_file_name = "history_boundary_scores.pkl"
#     elif reward_mode == "object_count":
#         scores_filename = "object_count_scores.pkl"
#         history_file_name = "history_object_count_scores.pkl"
#     else:
#         scores_filename = "penetration_scores.pkl"
#         history_file_name = "history_3d_scores.pkl"

#     if only_raw_scores:
#         return R, {}, R

#     os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
#     with open(os.path.join(save_dir, 'eval', scores_filename), 'wb') as f:
#         pickle.dump(R, f)

#     # History-based global z-score normalisation
#     history_data = []
#     if config.eval.history_cnt > 0:
#         history_path = os.path.join(config.save_path, unique_id, history_file_name)
#         if os.path.exists(history_path):
#             with open(history_path, 'rb') as f:
#                 history_data = pickle.load(f)
#         if len(history_data) > config.eval.history_cnt:
#             history_data = history_data[-config.eval.history_cnt:]

#     combined    = torch.cat(history_data + [R]) if history_data else R
#     global_mean = combined.mean().item()
#     global_std  = combined.std().item()

#     history_data.append(R)
#     if len(history_data) > config.eval.history_cnt:
#         history_data = history_data[-config.eval.history_cnt:]
#     with open(os.path.join(config.save_path, unique_id, history_file_name), 'wb') as f:
#         pickle.dump(history_data, f)

#     eval_scores = torch.tensor([(s - global_mean) / (global_std + 1e-8) for s in R]) # NOTE: this is sensible because we are not doing prompt wise alignment here, collision is similar across floor conditions

#     metrics = {
#         'raw_scores_mean':        float(R.mean()),
#         'raw_scores_std':         float(R.std()),
#         'raw_scores_min':         float(R.min()),
#         'raw_scores_max':         float(R.max()),
#         'normalized_scores_mean': float(eval_scores.mean()),
#         'normalized_scores_std':  float(eval_scores.std()),
#     }
#     # Explicit total composed raw mean (sum of universal components + optional custom).
#     metrics['component/total_raw_mean'] = float(R.mean())
#     for component_name, component_values in reward_components.items():
#         component_values = component_values.detach().cpu()
#         metrics[f'component/{component_name}_mean'] = float(component_values.mean())
#         metrics[f'component/{component_name}_std'] = float(component_values.std())
#     return eval_scores, metrics, R


# def score_fn1(ground, img_dir, save_dir, config, clip_model=None, clip_preprocess=None, clip_tokenizer=None, only_raw_scores=True):
#     """
#     Calculate CLIP-based similarity scores for images against text prompts.
    
#     Args:
#         ground: List of text prompts
#         img_dir: Directory containing images
#         save_dir: Directory to save results
#         config: Configuration object
#         clip_model: Pre-loaded CLIP model (optional, will load if not provided)
#         clip_preprocess: Pre-loaded CLIP preprocess function (optional)
#         clip_tokenizer: Pre-loaded CLIP tokenizer (optional)
        
#     Returns:
#         sum_scores: Normalized scores for each image
#         metrics: Dictionary with mean/std statistics per prompt
#     """
#     unique_id = config.exp_name
    
#     device = f"cuda" if torch.cuda.is_available() else "cpu"
    
#     # Use provided CLIP components or load them (backward compatibility)
#     if clip_model is None or clip_preprocess is None or clip_tokenizer is None:
#         model, _, preprocess = open_clip.create_model_and_transforms(
#             'ViT-H-14', 
#             pretrained='laion2B-s32B-b79K'
#         )
#         tokenizer = open_clip.get_tokenizer('ViT-H-14')
#         model = model.to(device)
#     else:
#         model = clip_model
#         preprocess = clip_preprocess
#         tokenizer = clip_tokenizer
    
#     eval_list = sorted(os.listdir(img_dir))
    
#     similarity = []
#     maximum_onetime = 8
    
#     for i in range(0, len(eval_list), maximum_onetime):
#         image_input = torch.tensor(
#             np.stack([
#                 preprocess(Image.open(os.path.join(img_dir, image))).numpy() 
#                 for image in eval_list[i:i+maximum_onetime]
#             ])
#         ).to(device)
#         text_inputs = tokenizer(ground[i:i+maximum_onetime]).to(device)
        
#         with torch.no_grad():
#             image_features = model.encode_image(image_input)
#             text_features = model.encode_text(text_inputs)
        
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
        
#         # Fix: use actual batch size instead of maximum_onetime to handle last batch correctly
#         actual_batch_size = len(image_features)
#         similarity.append(
#             (image_features @ text_features.T)[
#                 torch.arange(actual_batch_size), 
#                 torch.arange(actual_batch_size)
#             ]
#         )
    
#     similarity = torch.cat(similarity)
#     R = similarity.cpu().detach()
#     # print(R[:10])
#     if only_raw_scores:
#         return R, {}, R
#     # Save raw scores
#     os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
#     with open(os.path.join(save_dir, 'eval', 'scores.pkl'), 'wb') as f:
#         pickle.dump(R, f)
    
#     # Organize scores by prompt
#     each_score = {}
#     for idx, prompt in enumerate(ground):
#         if prompt in each_score:
#             each_score[prompt].append(R[idx:idx+1])
#         else:
#             each_score[prompt] = [R[idx:idx+1]]
    
#     # Load and update history
#     history_data = []
#     if config.eval.history_cnt > 0:
#         history_path = os.path.join(config.save_path, unique_id, 'history_scores.pkl')
#         if os.path.exists(history_path):
#             with open(history_path, 'rb') as f:
#                 history_data = pickle.load(f)
#         if len(history_data) > config.eval.history_cnt:
#             history_data = history_data[-config.eval.history_cnt:]
    
#     # Calculate statistics
#     data_mean = {}
#     data_std = {}
#     cur_data = {}
#     combine_data = {}
    
#     for k, v in each_score.items():
#         cur_data[k] = torch.cat(v, axis=0)
#         combine_data[k] = torch.cat(
#             [d[k] for d in history_data if k in d] + [cur_data[k]], 
#             axis=0
#         )
#         data_mean[k] = combine_data[k].mean().item()
#         data_std[k] = combine_data[k].std().item()
    
#     # Update history
#     history_data.append(cur_data)
#     if len(history_data) > config.eval.history_cnt:
#         history_data = history_data[-config.eval.history_cnt:]
    
#     with open(os.path.join(config.save_path, unique_id, 'history_scores.pkl'), 'wb') as f:
#         pickle.dump(history_data, f)
    
#     print(f"{data_mean=}")
    
#     # Normalize scores
#     if config.sample.normalize_all:
#         overall_mean = R.mean().item()
#         overall_std = R.std().item()
#         sum_scores = [
#             (s - overall_mean) / (overall_std + 1e-8) 
#             for s in R
#         ]
#     else:
#         sum_scores = [
#             (s - data_mean[ground[idx]]) / (data_std[ground[idx]] + 1e-8) 
#             for idx, s in enumerate(R)
#         ]
#     sum_scores = torch.tensor(sum_scores)
    
#     # Prepare metrics for logging
#     metrics = {
#         'raw_scores_mean': float(R.mean()),
#         'raw_scores_std': float(R.std()),
#         'raw_scores_min': float(R.min()),
#         'raw_scores_max': float(R.max()),
#         'normalized_scores_mean': float(sum_scores.mean()),
#         'normalized_scores_std': float(sum_scores.std()),
#         'num_prompts': len(data_mean),
#     }
#     # Add per-prompt statistics
#     for prompt, mean_val in data_mean.items():
#         prompt_key = prompt[:50].replace(' ', '_')  # Truncate and sanitize
#         metrics[f'prompt_mean/{prompt_key}'] = mean_val
#         metrics[f'prompt_std/{prompt_key}'] = data_std[prompt]
    
#     # Return both normalized scores (for training) and raw scores (for plotting)
#     return sum_scores, metrics, R


# def geometric_algebraic_score_fn(ground, img_dir, save_dir, config,
#                                   num_samples=5, min_length=20.0, threshold_c=0.03,
#                                   only_raw_scores=True,
#                                   # unused kwargs kept for signature compatibility with score_fn1
#                                   clip_model=None, clip_preprocess=None, clip_tokenizer=None):
#     """
#     Geometric reward function based on algebraic vanishing-point intersection quality.
#     Drop-in replacement for score_fn1 — same call signature and return contract.

#     Args:
#         ground: list[str] — text prompts, one per image (same order as sorted img_dir)
#         img_dir: str — directory of generated PNG images
#         save_dir: str — directory to save intermediate results
#         config: OmegaConf config object
#         num_samples: number of line samples for geometric reward
#         min_length: minimum line length threshold
#         threshold_c: algebraic intersection threshold
#         only_raw_scores: if True, skip history-based normalisation (used during FK resampling)

#     Returns:
#         sum_scores: torch.Tensor (N,) — z-score normalised advantages for training
#         metrics:    dict            — logging statistics
#         R:          torch.Tensor (N,) — raw scalar rewards per image
#     """
    

#     reward_calculator = ImageGeometricReward(min_length=min_length)

#     eval_list = sorted(os.listdir(img_dir))

#     raw_rewards = []
#     for img_name in eval_list:
#         img_path = os.path.join(img_dir, img_name)
#         try:
#             # Load as HxWx3 uint8 numpy array (what ImageGeometricReward expects)
#             image = np.array(Image.open(img_path).convert("RGB"))
#             reward = reward_calculator.get_algebraic_intersection_reward(
#                 image, num_samples=num_samples, threshold_c=threshold_c
#             )
#         except Exception:
#             reward = 0.0
#         raw_rewards.append(float(reward))

#     R = torch.tensor(raw_rewards, dtype=torch.float32)

#     if only_raw_scores:
#         return R, {}, R

#     # ---- history-based per-prompt z-score normalisation (mirrors score_fn1) ----
#     unique_id = config.exp_name
#     os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
#     with open(os.path.join(save_dir, 'eval', 'geometric_scores.pkl'), 'wb') as f:
#         pickle.dump(R, f)

#     # Organise by prompt
#     each_score = {}
#     for idx, prompt in enumerate(ground):
#         each_score.setdefault(prompt, []).append(R[idx:idx + 1])

#     # Load and update history
#     history_data = []
#     if config.eval.history_cnt > 0:
#         history_path = os.path.join(config.save_path, unique_id, 'history_geometric_scores.pkl')
#         if os.path.exists(history_path):
#             with open(history_path, 'rb') as f:
#                 history_data = pickle.load(f)
#         if len(history_data) > config.eval.history_cnt:
#             history_data = history_data[-config.eval.history_cnt:]

#     data_mean, data_std, cur_data, combine_data = {}, {}, {}, {}
#     for k, v in each_score.items():
#         cur_data[k] = torch.cat(v, dim=0)
#         combine_data[k] = torch.cat(
#             [d[k] for d in history_data if k in d] + [cur_data[k]], dim=0
#         )
#         data_mean[k] = combine_data[k].mean().item()
#         data_std[k] = combine_data[k].std().item()

#     history_data.append(cur_data)
#     if len(history_data) > config.eval.history_cnt:
#         history_data = history_data[-config.eval.history_cnt:]
#     with open(os.path.join(config.save_path, unique_id, 'history_geometric_scores.pkl'), 'wb') as f:
#         pickle.dump(history_data, f)

#     if config.sample.normalize_all:
#         overall_mean = R.mean().item()
#         overall_std = R.std().item()
#         sum_scores = [(s - overall_mean) / (overall_std + 1e-8) for s in R]
#     else:
#         sum_scores = [
#             (s - data_mean[ground[idx]]) / (data_std[ground[idx]] + 1e-8)
#             for idx, s in enumerate(R)
#         ]
#     sum_scores = torch.tensor(sum_scores)

#     metrics = {
#         'raw_scores_mean': float(R.mean()),
#         'raw_scores_std': float(R.std()),
#         'raw_scores_min': float(R.min()),
#         'raw_scores_max': float(R.max()),
#         'normalized_scores_mean': float(sum_scores.mean()),
#         'normalized_scores_std': float(sum_scores.std()),
#         'num_prompts': len(data_mean),
#     }
#     for prompt, mean_val in data_mean.items():
#         prompt_key = prompt[:50].replace(' ', '_')
#         metrics[f'prompt_mean/{prompt_key}'] = mean_val
#         metrics[f'prompt_std/{prompt_key}'] = data_std[prompt]

#     return sum_scores, metrics, R



# def run_selection(config, stage_idx=None, logger=None, wandb_run=None):
#     """
#     Run the selection phase for a given stage.
    
#     Args:
#         config: Configuration object (can be OmegaConf or dict)
#         stage_idx: Current stage index (optional, for logging)
#         logger: Logger instance (optional)
#         wandb_run: Existing wandb run to log to (optional)
        
#     Returns:
#         save_dir: Directory where selected samples were saved
#         metrics: Dictionary of selection metrics for logging
#     """
#     if logger:
#         logger.info(f"Starting reward calculation for stage {stage_idx}")
#     else:
#         print(f"Starting reward calculation for stage {stage_idx}")
    
#     torch.cuda.set_device(config.dev_id)
#     seed_everything(config.seed)
    
#     unique_id = config.exp_name
#     stage_id = f"stage{stage_idx}"
#     save_dir = os.path.join(config.save_path, unique_id, stage_id)
    
#     # Load samples
#     with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as f:
#         samples = pickle.load(f)

#     threed = getattr(config, 'threed_scene_layout', False)

#     if threed:
#         # 3D path: floor condition indices as context, non-penetration reward
#         with open(os.path.join(save_dir, 'fpbpn_list.json'), 'r') as f:
#             ground = json.load(f)

#         device  = f"cuda:{config.dev_id}" if torch.cuda.is_available() else "cpu"
#         x0_raw  = samples["next_scenes"][:, -1].to(device)  # (B, N, C) final denoised step
#         indices = _resolve_fpbpn_indices(ground, config)
#         if len(indices) != x0_raw.shape[0]:
#             raise ValueError(
#                 f"fpbpn_list length ({len(indices)}) does not match samples ({x0_raw.shape[0]})."
#             )
#         eval_scores, score_metrics, raw_scores = threed_score_fn(
#             x0_raw, save_dir, config, indices=indices, only_raw_scores=False
#         )
#         _embed_key    = 'fpbpn'
#         _lat_key      = 'scenes'
#         _next_lat_key = 'next_scenes'
#         if getattr(config, 'universal_rewards', False):
#             reward_mode = 'universal'
#             if getattr(config, 'custom_reward', None) is not None:
#                 reward_mode = f"{reward_mode}+{config.custom_reward}"
#         elif getattr(config, 'custom_reward', None) is not None:
#             reward_mode = f"custom_{config.custom_reward}"
#         else:
#             reward_mode = getattr(config, 'threed_reward', 'penetration')
#         print(f"3D {reward_mode} rewards: mean={raw_scores.mean():.4f}, std={raw_scores.std():.4f}")
#     else:
#         # SD path: text prompts, CLIP/geometric reward
#         with open(os.path.join(save_dir, 'prompt.json'), 'r') as f:
#             ground = json.load(f)

#         img_dir        = os.path.join(save_dir, 'images')
#         reward_fn_name = getattr(config, 'reward_fn', 'clip')
#         if reward_fn_name == 'geometric':
#             eval_scores, score_metrics, raw_scores = geometric_algebraic_score_fn(
#                 ground, img_dir, save_dir, config, only_raw_scores=False
#             )
#         else:
#             eval_scores, score_metrics, raw_scores = score_fn1(
#                 ground, img_dir, save_dir, config, only_raw_scores=False
#             )
#         _embed_key    = 'prompt_embeds'
#         _lat_key      = 'latents'
#         _next_lat_key = 'next_latents'

#     raw_clip_scores = raw_scores  # alias kept for rest of function
#     print(f"{raw_clip_scores=}, dtype of eval score tensor: {eval_scores.dtype}")
#     print(f"{eval_scores=}")
#     samples['eval_scores'] = eval_scores  # Normalized scores for training

#     if config.sample.save_train_samples_no_train:
#         all_scores_list = [float(s) for s in raw_clip_scores]
#         rewards_key     = 'threed_reward' if threed else 'clip_reward'
#         mean_rewards_summary = {
#             f"{rewards_key}_mean": float(raw_clip_scores.mean()),
#             f"{rewards_key}_std":  float(raw_clip_scores.std()),
#             f"{rewards_key}_min":  float(raw_clip_scores.min()),
#             f"{rewards_key}_max":  float(raw_clip_scores.max()),
#             "all_scores": all_scores_list,
#         }
#         rewards_fname = 'threed_rewards.json' if threed else 'clip_rewards.json'
#         rewards_path  = os.path.join(save_dir, rewards_fname)
#         with open(rewards_path, 'w') as f:
#             json.dump(mean_rewards_summary, f, indent=2)
#         if logger:
#             logger.info(f"Reward statistics saved to {rewards_path}")
#         else:
#             print(f"Reward statistics saved to {rewards_path}")
#         import sys; sys.exit(0)

#     # Initialize data structure for selected samples
#     def get_new_unit():
#         return {
#             _embed_key:    [],
#             'timesteps':   [],
#             'log_probs':   [],
#             _lat_key:      [],
#             _next_lat_key: [],
#             'eval_scores': [],
#         }

#     data = get_new_unit()
#     if config.sample.no_selection:
#         t_left  = 0
#         t_right = config.sample.num_steps
#         data = {
#             _embed_key:    samples[_embed_key],
#             'timesteps':   samples['timesteps'][:, t_left:t_right],
#             'log_probs':   samples['log_probs'][:, t_left:t_right],
#             _lat_key:      samples[_lat_key][:, t_left:t_right],
#             _next_lat_key: samples[_next_lat_key][:, t_left:t_right],
#             'eval_scores': samples['eval_scores'],
#         }

#     else:  # Select positive and negative samples
#         total_batch_size = samples['eval_scores'].shape[0]

#         if config.sample.fk:
#             fk_particles = config.sample.num_particles * (1 if getattr(config.sample, 'only_best_fk', False) else 2)
#             data_size    = fk_particles
#             batch_size   = total_batch_size // fk_particles
#         else:
#             data_size  = total_batch_size // config.sample.batch_size
#             batch_size = config.sample.batch_size

#         for b in range(batch_size):
#             cur_sample_num = 1

#             if config.sample.fk:
#                 start_idx     = b * fk_particles
#                 end_idx       = start_idx + fk_particles
#                 batch_samples = {k: v[start_idx:end_idx] for k, v in samples.items()}
#             else:
#                 batch_samples = {
#                     k: v[torch.arange(b, total_batch_size, batch_size)]
#                     for k, v in samples.items()
#                 }

#             if (hasattr(config, 'train') and getattr(config.train, 'incremental_training', False)) or config.sample.fk:
#                 t_left  = 0
#                 t_right = config.sample.num_steps
#             else:
#                 t_left  = config.sample.num_steps - config.split_step
#                 t_right = config.sample.num_steps

#             if config.train.only_last_n_steps > 0:
#                 t_left = config.sample.num_steps - config.train.only_last_n_steps

#             idx_sel   = torch.arange(0, data_size, cur_sample_num)
#             embeds    = batch_samples[_embed_key][idx_sel]
#             timesteps = batch_samples['timesteps'][idx_sel, t_left:t_right]
#             log_probs = batch_samples['log_probs'][idx_sel, t_left:t_right]
#             lats      = batch_samples[_lat_key][idx_sel, t_left:t_right]
#             next_lats = batch_samples[_next_lat_key][idx_sel, t_left:t_right]

#             score = batch_samples['eval_scores'][idx_sel]
#             print(f"Batch {b}: score shape before reshape: {score.shape}")

#             if config.sample.fk:
#                 score = score.reshape(1, -1)
#             else:
#                 score = score.reshape(-1, config.split_time)
#             max_idx = score.argmax(dim=1)
#             min_idx = score.argmin(dim=1)

#             for j, s in enumerate(score):
#                 for p_n in range(2):
#                     if p_n == 0 and s[max_idx[j]] >= config.eval.pos_threshold:
#                         used_idx   = max_idx[j]
#                         used_idx_2 = used_idx if config.sample.fk else j * config.split_time + used_idx
#                     elif p_n == 1 and s[min_idx[j]] < config.eval.neg_threshold:
#                         used_idx   = min_idx[j]
#                         used_idx_2 = used_idx if config.sample.fk else j * config.split_time + used_idx
#                     else:
#                         if config.sample.no_selection:
#                             used_idx   = min_idx[j]
#                             used_idx_2 = used_idx if config.sample.fk else j * config.split_time + used_idx
#                         else:
#                             continue

#                     data[_embed_key].append(embeds[used_idx_2])
#                     data['timesteps'].append(timesteps[used_idx_2])
#                     data['log_probs'].append(log_probs[used_idx_2])
#                     data[_lat_key].append(lats[used_idx_2])
#                     data[_next_lat_key].append(next_lats[used_idx_2])
#                     data['eval_scores'].append(s[used_idx])

#             if not config.sample.fk:
#                 cur_sample_num *= config.split_time

#     # Stack data if any samples were selected
#     if len(data[_embed_key]) > 0:
#         first_value = next(iter(data.values()))
#         if isinstance(first_value, list):
#             data = {k: torch.stack(v, dim=0) for k, v in data.items()}
#         if logger:
#             logger.info(f"Selected {len(data[_embed_key])} samples")
#     else:
#         if logger:
#             logger.warning("No samples met the selection criteria")
#         print("Warning: No samples met the selection criteria")
#         # Convert all empty lists to empty tensors so downstream code can call
#         # .shape[0], .dtype, etc. without crashing.
#         data = {k: (torch.stack(v, dim=0) if (isinstance(v, list) and len(v) > 0) else torch.tensor([])) for k, v in data.items()}

#     # Save selected samples
#     with open(os.path.join(save_dir, 'sample_stage.pkl'), 'wb') as f:
#         pickle.dump(data, f)

#     all_samples_mean_reward = float(raw_clip_scores.mean())
#     all_samples_std_reward  = float(raw_clip_scores.std())
#     num_reward_queries      = len(raw_clip_scores)

#     cumulative_queries_path = os.path.join(config.save_path, unique_id, 'cumulative_reward_queries.pkl')
#     if os.path.exists(cumulative_queries_path):
#         with open(cumulative_queries_path, 'rb') as f:
#             cumulative_reward_queries = pickle.load(f)
#     else:
#         cumulative_reward_queries = 0
#     cumulative_reward_queries += num_reward_queries
#     with open(cumulative_queries_path, 'wb') as f:
#         pickle.dump(cumulative_reward_queries, f)

#     print(f"dtype of eval_scores: {data.get('eval_scores', torch.tensor([])).dtype if isinstance(data.get('eval_scores'), torch.Tensor) else 'list (no samples selected)'}")
#     _es_raw = data.get('eval_scores', [])
#     if isinstance(_es_raw, list):
#         _es = torch.stack(_es_raw) if len(_es_raw) > 0 else torch.tensor([])
#     else:
#         _es = _es_raw
#     num_selected  = len(data.get(_embed_key, []))
#     num_positive  = int((_es >= config.eval.pos_threshold).sum()) if len(_es) > 0 else 0
#     num_negative  = int((_es <  config.eval.neg_threshold).sum()) if len(_es) > 0 else 0
#     num_generated = len(raw_clip_scores)
#     num_rejected  = num_generated - num_selected

#     selection_metrics = {
#         "num_selected":       num_selected,
#         "num_positive":       num_positive,
#         "num_negative":       num_negative,
#         "num_generated":      num_generated,
#         "num_rejected":       num_rejected,
#         "mean_reward":        all_samples_mean_reward,
#         "std_reward":         all_samples_std_reward,
#         "num_queries":        num_reward_queries,
#         "cumulative_queries": cumulative_reward_queries,
#     }

#     # Carry score-level metrics (including per-component raw means) to pipeline logging.
#     for metric_name, metric_value in score_metrics.items():
#         if isinstance(metric_value, (int, float)):
#             selection_metrics[f"score/{metric_name}"] = float(metric_value)

#     if logger:
#         logger.info(f"Selection completed for stage {stage_idx}")
#         logger.info(f"Generated: {num_generated}, Selected: {num_selected}, Rejected: {num_rejected}")
#         logger.info(f"Positive: {num_positive}, Negative: {num_negative}")
#         logger.info(f"Mean reward (all samples): {all_samples_mean_reward:.4f} ± {all_samples_std_reward:.4f}")
#         logger.info(f"Cumulative reward queries: {cumulative_reward_queries}")

#     return save_dir, selection_metrics
