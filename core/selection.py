"""
Core selection module for B2Diff training pipeline.
Extracts the selection logic from run_select.py into a callable function.
"""

import os
import torch
import pickle
import json
import numpy as np
from PIL import Image
from functools import partial
from tqdm import tqdm as tqdm_lib
from bert_score import score, BERTScorer
import open_clip
from utils.utils import seed_everything

tqdm = partial(tqdm_lib, dynamic_ncols=True)


def score_fn1(ground, img_dir, save_dir, config):
    """
    Calculate CLIP-based similarity scores for images against text prompts.
    
    Args:
        ground: List of text prompts
        img_dir: Directory containing images
        save_dir: Directory to save results
        config: Configuration object
        
    Returns:
        sum_scores: Normalized scores for each image
        metrics: Dictionary with mean/std statistics per prompt
    """
    unique_id = config.exp_name
    
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', 
        pretrained='laion2B-s32B-b79K'
    )
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    model = model.to(device)
    
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
    
    print(data_mean)
    
    # Normalize scores
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
    
    # Load samples and prompts
    with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as f:
        samples = pickle.load(f)
    with open(os.path.join(save_dir, 'prompt.json'), 'r') as f:
        ground = json.load(f)
    
    # Evaluate scores
    img_dir = os.path.join(save_dir, 'images')
    eval_scores, score_metrics, raw_clip_scores = score_fn1(ground, img_dir, save_dir, config)
    samples['eval_scores'] = eval_scores  # Normalized scores for training
    
    # Initialize data structure for selected samples
    def get_new_unit():
        return {
            'prompt_embeds': [],
            'timesteps': [],
            'log_probs': [],
            'latents': [],
            'next_latents': [],
            'eval_scores': []
        }
    
    data = get_new_unit()
    
    # Select positive and negative samples
    total_batch_size = samples['eval_scores'].shape[0]
    data_size = total_batch_size // config.sample.batch_size
    
    for b in range(config.sample.batch_size):
        cur_sample_num = 1
        batch_samples = {
            k: v[torch.arange(b, total_batch_size, config.sample.batch_size)] 
            for k, v in samples.items()
        }
        
        # When incremental training is enabled, keep the full trajectory
        # Otherwise, preserve original behavior (use last `split_step` timesteps)
        if hasattr(config, 'train') and getattr(config.train, 'incremental_training', False):
            t_left = 0
            t_right = config.sample.num_steps
        else:
            t_left = config.sample.num_steps - config.split_step
            t_right = config.sample.num_steps
        
        prompt_embeds = batch_samples['prompt_embeds'][torch.arange(0, data_size, cur_sample_num)]
        timesteps = batch_samples['timesteps'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        log_probs = batch_samples['log_probs'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        latents = batch_samples['latents'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        next_latents = batch_samples['next_latents'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        
        score = batch_samples['eval_scores'][torch.arange(0, data_size, cur_sample_num)]
        score = score.reshape(-1, config.split_time)
        max_idx = score.argmax(dim=1)
        min_idx = score.argmin(dim=1)
        
        for j, s in enumerate(score):
            for p_n in range(2):
                if p_n == 0 and s[max_idx[j]] >= config.eval.pos_threshold:
                    used_idx = max_idx[j]
                    used_idx_2 = j * config.split_time + max_idx[j]
                elif p_n == 1 and s[min_idx[j]] < config.eval.neg_threshold:
                    used_idx = min_idx[j]
                    used_idx_2 = j * config.split_time + min_idx[j]
                else:
                    if config.sample.no_branching:
                        used_idx = min_idx[j]
                        used_idx_2 = j * config.split_time + min_idx[j]
                    continue
                
                data['prompt_embeds'].append(prompt_embeds[used_idx_2])
                data['timesteps'].append(timesteps[used_idx_2])
                data['log_probs'].append(log_probs[used_idx_2])
                data['latents'].append(latents[used_idx_2])
                data['next_latents'].append(next_latents[used_idx_2])
                data['eval_scores'].append(s[used_idx])
        
        cur_sample_num *= config.split_time
    
    # Stack data if any samples were selected
    if len(data['prompt_embeds']) > 0:
        data = {k: torch.stack(v, dim=0) for k, v in data.items()}
        
        if logger:
            logger.info(f"Selected {len(data['prompt_embeds'])} samples")
    else:
        if logger:
            logger.warning("No samples met the selection criteria")
        print("Warning: No samples met the selection criteria")
    
    # Save selected samples
    with open(os.path.join(save_dir, 'sample_stage.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    # Calculate mean reward of ALL generated samples (before selection)
    # IMPORTANT: Use raw CLIP scores for reward tracking (not normalized), to match paper's plot
    all_samples_mean_reward = float(raw_clip_scores.mean())
    all_samples_std_reward = float(raw_clip_scores.std())
    num_reward_queries = len(raw_clip_scores)
    
    # Load and update cumulative reward query count
    cumulative_queries_path = os.path.join(config.save_path, unique_id, 'cumulative_reward_queries.pkl')
    if os.path.exists(cumulative_queries_path):
        with open(cumulative_queries_path, 'rb') as f:
            cumulative_reward_queries = pickle.load(f)
    else:
        cumulative_reward_queries = 0
    
    cumulative_reward_queries += num_reward_queries
    
    # Save updated cumulative count
    with open(cumulative_queries_path, 'wb') as f:
        pickle.dump(cumulative_reward_queries, f)
    
    # Prepare clean metrics for aggregation at pipeline level
    # Return metrics WITHOUT per-stage prefixes - pipeline will handle aggregation
    num_selected = len(data.get('prompt_embeds', []))
    num_positive = int((data.get('eval_scores', torch.tensor([])) >= config.eval.pos_threshold).sum()) if len(data.get('eval_scores', [])) > 0 else 0
    num_negative = int((data.get('eval_scores', torch.tensor([])) < config.eval.neg_threshold).sum()) if len(data.get('eval_scores', [])) > 0 else 0
    num_generated = len(raw_clip_scores)
    num_rejected = num_generated - num_selected
    
    selection_metrics = {
        # Clean metrics for consolidation
        "num_selected": num_selected,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "num_generated": num_generated,
        "num_rejected": num_rejected,
        "mean_reward": all_samples_mean_reward,
        "std_reward": all_samples_std_reward,
        "num_queries": num_reward_queries,
        "cumulative_queries": cumulative_reward_queries,
    }
    
    if logger:
        logger.info(f"Selection completed for stage {stage_idx}")
        logger.info(f"Generated: {num_generated}, Selected: {num_selected}, Rejected: {num_rejected}")
        logger.info(f"Positive: {num_positive}, Negative: {num_negative}")
        logger.info(f"Mean reward (all samples): {all_samples_mean_reward:.4f} ± {all_samples_std_reward:.4f}")
        logger.info(f"Cumulative reward queries: {cumulative_reward_queries}")
    
    return save_dir, selection_metrics
