#!/usr/bin/env python3
"""
Script to generate a GIF from images in a directory.
Automatically sorts images and creates an animated GIF.
"""

import os
import glob
from PIL import Image
import argparse
import re


def natural_sort_key(s):
    """Sort strings with numbers naturally (e.g., img1, img2, img10)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def create_gif(image_dir, output_path='output.gif', duration=200, loop=0):
    """
    Create a GIF from images in a directory.
    
    Args:
        image_dir: Directory containing images
        output_path: Path for the output GIF file
        duration: Duration of each frame in milliseconds (default: 200ms)
        loop: Number of loops (0 = infinite, default: 0)
    """
    # Supported image formats
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff']
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # Sort files naturally
    image_files.sort(key=natural_sort_key)
    
    print(f"Found {len(image_files)} images")
    print(f"First image: {os.path.basename(image_files[0])}")
    print(f"Last image: {os.path.basename(image_files[-1])}")
    
    # Load images
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            # Convert to RGB if necessary (for PNG with transparency, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            print(f"Loaded: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    if not images:
        print("No valid images loaded")
        return
    
    # Save as GIF
    print(f"\nCreating GIF with {len(images)} frames...")
    print(f"Duration per frame: {duration}ms")
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
    print(f"\n✓ GIF created successfully!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Frames: {len(images)}")


def main():
    parser = argparse.ArgumentParser(
        description='Create an animated GIF from images in a directory'
    )
    parser.add_argument(
        'image_dir',
        nargs='?',
        default='/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/traj_viz',
        help='Directory containing images (default: traj_viz directory)'
    )
    parser.add_argument(
        '-o', '--output',
        default='trajectory_animation.gif',
        help='Output GIF filename (default: trajectory_animation.gif)'
    )
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=200,
        help='Duration of each frame in milliseconds (default: 200)'
    )
    parser.add_argument(
        '-l', '--loop',
        type=int,
        default=0,
        help='Number of loops (0 = infinite, default: 0)'
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.image_dir):
        print(f"Error: Directory '{args.image_dir}' does not exist")
        return
    
    create_gif(args.image_dir, args.output, args.duration, args.loop)


if __name__ == '__main__':
    main()