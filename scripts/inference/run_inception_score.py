#!/usr/bin/env python
"""
code to compute Inception Score using CLIP-ViT-H-14-laion2B-s32B-b79K model
Based on: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
"""

import os
import sys
import json
import argparse
import gc
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from scipy.stats import entropy
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from tqdm import tqdm

# Add project root to path for imports
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(project_root)
from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
from utils.utils import seed_everything


class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def compute_inception_score_clip(clip_model, imgs, cuda=True, batch_size=32, splits=1):
    """
    Computes the inception score of the generated images using CLIP model
    
    Args:
        clip_model: CLIP model for encoding images
        imgs: Torch dataset of (3xHxW) images (preprocessed by CLIP's transform)
        cuda: whether or not to run on GPU
        batch_size: batch size for feeding into CLIP
        splits: number of splits for computing IS
        
    Returns:
        mean: Mean inception score across splits
        std: Standard deviation of inception scores
    """
    N = len(imgs)
    
    assert batch_size > 0
    assert N > batch_size, f"Number of images ({N}) must be greater than batch_size ({batch_size})"
    
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor
    
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    
    def get_pred(x):
        """Get softmax predictions from CLIP image encoder"""
        with torch.no_grad():
            embeddings = clip_model.encode_image(x)
            return F.softmax(embeddings, dim=-1).cpu().numpy()
    
    preds = []
    print("Encoding images...")
    for i, batch in enumerate(dataloader):
        batch = batch.type(dtype)
        preds.append(get_pred(batch))
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * batch_size}/{N} images")
    
    preds = np.concatenate(preds, axis=0)
    print(f"Predictions shape: {preds.shape}")
    
    # Compute the mean KL-divergence
    split_scores = []
    
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)


def sample_and_compute_is_baseline(
    num_images=1000,
    prompt_file="config/prompt/template1_train.json",
    output_dir="outputs/baseline_sd14_samples",
    batch_size=8,
    num_inference_steps=20,
    guidance_scale=5.0,
    splits=10
):
    """
    Sample images from baseline SD v1-4 model and compute Inception Score
    
    Args:
        num_images: Total number of images to generate
        prompt_file: Path to JSON file containing prompts
        output_dir: Directory to save generated images
        batch_size: Batch size for image generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        splits: Number of splits for IS computation
        
    Returns:
        is_mean: Mean inception score
        is_std: Standard deviation of inception score
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load prompts
    print(f"Loading prompts from {prompt_file}...")
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")
    
    # Load baseline Stable Diffusion v1-4 model
    print("Loading Stable Diffusion v1-4 baseline model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    print("Model loaded successfully!")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images
    print(f"\nGenerating {num_images} images...")
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Determine actual batch size for this iteration
        current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
        
        # Sample prompts (cycle through if needed)
        batch_prompts = [prompts[(batch_idx * batch_size + i) % len(prompts)] 
                        for i in range(current_batch_size)]
        
        # Generate images
        with torch.no_grad():
            images = pipeline(
                batch_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images
        
        # Save images
        for i, img in enumerate(images):
            img_idx = batch_idx * batch_size + i
            img.save(os.path.join(output_dir, f"{img_idx:05d}.png"))
    
    print(f"Generated {num_images} images saved to {output_dir}")
    
    # Now compute Inception Score using CLIP
    print("\nLoading CLIP model for IS computation...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', 
        pretrained='laion2B-s32B-b79K'
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    print("CLIP model loaded!")
    
    # Load generated images
    print(f"Loading generated images...")
    dataset = ImageDataset(output_dir, transform=preprocess)
    print(f"Found {len(dataset)} images")
    
    # Compute Inception Score
    print("Computing Inception Score...")
    is_mean, is_std = compute_inception_score_clip(
        clip_model=clip_model,
        imgs=dataset,
        cuda=torch.cuda.is_available(),
        batch_size=32,
        splits=splits
    )
    
    # Print and save results
    print("\n" + "="*60)
    print(f"Baseline SD v1-4 Inception Score Results:")
    print(f"  Number of images: {num_images}")
    print(f"  Mean: {is_mean:.4f}")
    print(f"  Std:  {is_std:.4f}")
    print("="*60)
    
    # Save results
    result_file = os.path.join(output_dir, "inception_score.txt")
    with open(result_file, 'w') as f:
        f.write(f"Baseline Stable Diffusion v1-4 Inception Score\n")
        f.write(f"="*60 + "\n")
        f.write(f"Model: CompVis/stable-diffusion-v1-4\n")
        f.write(f"Prompt File: {prompt_file}\n")
        f.write(f"Number of Images: {num_images}\n")
        f.write(f"Inference Steps: {num_inference_steps}\n")
        f.write(f"Guidance Scale: {guidance_scale}\n")
        f.write(f"IS Splits: {splits}\n")
        f.write(f"\n")
        f.write(f"Inception Score Mean: {is_mean:.4f}\n")
        f.write(f"Inception Score Std:  {is_std:.4f}\n")
    
    print(f"\nResults saved to: {result_file}")
    
    return is_mean, is_std


def get_inception_score(image_dir):
    # Configuration
    batch_size = 32
    splits = 10
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading CLIP-ViT-H-14-laion2B-s32B-b79K model...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    print(f"Loading images from {image_dir}...")
    dataset = ImageDataset(image_dir, transform=preprocess)
    print(f"Found {len(dataset)} images")
    
    print("Computing Inception Score...")
    use_cuda = torch.cuda.is_available()
    is_mean, is_std = compute_inception_score_clip(
        clip_model=model,
        imgs=dataset,
        cuda=use_cuda,
        batch_size=batch_size,
        splits=splits
    )
    
    # Print results
    print("\n" + "="*50)
    print(f"Inception Score Results:")
    print(f"  Mean: {is_mean:.4f}")
    print(f"  Std:  {is_std:.4f}")
    print("="*50)
    
    # Save results
    output_file = os.path.join(os.path.dirname(image_dir), "inception_score.txt")
    with open(output_file, 'w') as f:
        f.write(f"Inception Score (CLIP-ViT-H-14-laion2B-s32B-b79K)\n")
        f.write(f"Image Directory: {image_dir}\n")
        f.write(f"Number of Images: {len(dataset)}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Splits: {splits}\n")
        f.write(f"Mean: {is_mean:.4f}\n")
        f.write(f"Std:  {is_std:.4f}\n")
    
    print(f"\nResults saved to: {output_file}")

# if __name__ == "__main__":
#     img_dir = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/b2diffu_try2/images/"
#     main(img_dir)
#     parser = argparse.ArgumentParser(description="Compute Inception Score using CLIP")
#     parser.add_argument(
#         "--mode",
#         type=str,
#         choices=["existing", "baseline", "lora"],
#         default="lora",
#         help="Mode: 'existing' to compute IS on existing images, 'baseline' to generate and compute IS for baseline SD v1-4, 'lora' to generate from LoRA checkpoint and compute IS"
#     )
#     parser.add_argument(
#         "--image_dir",
#         type=str,
#         default="/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/test",
#         help="Directory containing images (for 'existing' mode)"
#     )
#     parser.add_argument(
#         "--checkpoint_path",
#         type=str,
#         default="/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/only_5_steps/pytorch_lora_weights.safetensors",
#         help="Path to LoRA checkpoint (for 'lora' mode)"
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/only_5_steps/images",
#         help="Output directory for generated images (for 'lora' mode)"
#     )
#     parser.add_argument(
#         "--num_images",
#         type=int,
#         default=1000,
#         help="Number of images to generate (for 'baseline' and 'lora' modes)"
#     )
#     parser.add_argument(
#         "--prompt_file",
#         type=str,
#         default="configs/prompt/template1_train.json",
#         help="Path to prompt file (for 'baseline' and 'lora' modes)"
#     )
#     parser.add_argument(
#         "--base_model",
#         type=str,
#         default="CompVis/stable-diffusion-v1-4",
#         help="Base model for generation (for 'lora' mode)"
#     )
#     parser.add_argument(
#         "--gen_batch_size",
#         type=int,
#         default=4,
#         help="Batch size for generation (for 'baseline' and 'lora' modes)"
#     )
#     parser.add_argument(
#         "--eval_batch_size",
#         type=int,
#         default=32,
#         help="Batch size for IS computation"
#     )
#     parser.add_argument(
#         "--num_inference_steps",
#         type=int,
#         default=20,
#         help="Number of inference steps (for 'lora' mode)"
#     )
#     parser.add_argument(
#         "--guidance_scale",
#         type=float,
#         default=5.0,
#         help="Guidance scale (for 'lora' mode)"
#     )
#     parser.add_argument(
#         "--eta",
#         type=float,
#         default=1.0,
#         help="DDIM eta parameter (for 'lora' mode)"
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=300,
#         help="Random seed (for 'lora' mode)"
#     )
#     parser.add_argument(
#         "--splits",
#         type=int,
#         default=10,
#         help="Number of splits for IS computation"
#     )
    
#     args = parser.parse_args()
    
#     if args.mode == "lora":
#         # Generate images from LoRA checkpoint and compute IS
#         sample_and_compute_is_lora(
#             checkpoint_path=args.checkpoint_path,
#             output_dir=args.output_dir,
#             prompt_file=args.prompt_file,
#             num_images=args.num_images,
#             base_model=args.base_model,
#             gen_batch_size=args.gen_batch_size,
#             num_inference_steps=args.num_inference_steps,
#             guidance_scale=args.guidance_scale,
#             eta=args.eta,
#             seed=args.seed,
#             eval_batch_size=args.eval_batch_size,
#             splits=args.splits
#         )
#     elif args.mode == "baseline":
#         # Generate images from baseline and compute IS
#         sample_and_compute_is_baseline(
#             num_images=args.num_images,
#             prompt_file=args.prompt_file,
#             output_dir=args.output_dir if args.output_dir != "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/b2diffu_try2/inference_results_88_ckpt" else "outputs/baseline_sd14_samples",
#             batch_size=args.gen_batch_size,
#             splits=args.splits
#         )
#     else:
#         # Compute IS on existing images
#         main(args.image_dir)

if __name__ == "__main__":
    img_dir = "/home/pramish_paudel/codes/b2diff/tmp"
    get_inception_score(img_dir)
