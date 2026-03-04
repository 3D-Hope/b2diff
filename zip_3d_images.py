#!/usr/bin/env python3
"""
Image Organization Script
--------------------------
This script organizes images from four directories into a zip file
with subdirectories: Pretrained, DDPO, B2DIFF, and Ours
"""

import os
import shutil
import zipfile
from pathlib import Path

# Source directories
source_dirs = {
    'Pretrained': '/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test_pretrained_6k',
    # 'DDPO': '/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/ddpo_tv_bed/stage92/',
    # 'B2DIFF': '/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/b2_tv_bed/stage76/',
    # 'Ours': '/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/4_particles_incremental_branch_fk_tv_bed/stage50'
}

# Image name patterns to match
image_patterns = [
    '0000_Bedroom-15797',
    '0046_SecondBedroom-22370',
    '0056_Bedroom-1541',
    '0062_MasterBedroom-40255',
    '0110_MasterBedroom-6351',
    '0124_SecondBedroom-266050',
    '0147_SecondBedroom-11167',
    '0180_MasterBedroom-13658',
    '0184_Bedroom-35352',
    '0186_Bedroom-6993',
    '0200_SecondBedroom-30334',
    '0215_SecondBedroom-24075',
    '0224_MasterBedroom-9302',
    '0298_MasterBedroom-132469',
    '0334_SecondBedroom-36408',
    '0342_MasterBedroom-113',
    '0402_MasterBedroom-9583',
    '0413_SecondBedroom-52584',
    '0439_MasterBedroom-9302',
    '0449_SecondBedroom-69136',
    '0520_MasterBedroom-3151',
    '0528_SecondBedroom-9923',
    '0564_MasterBedroom-28978',
    '0582_MasterBedroom-132469',
    '0642_MasterBedroom-12352',
    '0655_MasterBedroom-33296',
    '0662_MasterBedroom-12352',
    '0677_MasterBedroom-75552',
    '0707_MasterBedroom-75552',
    '0727_MasterBedroom-16450',
    '0729_MasterBedroom-109561',
    '0002_SecondBedroom-37794',
    '0013_Bedroom-41491',
    '0055_SecondBedroom-1145',
]

# Common image extensions
image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp', '.BMP', '.tiff', '.TIFF']

def main():
    # Output zip file location (change this to your desired location)
    output_zip = 'pretrained_outputs.zip'
    
    # Create temporary directory structure
    temp_dir = 'temp_organized_images'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Statistics - FIXED: includes all four directories
    # stats = {'Pretrained': 0, 'DDPO': 0, 'B2DIFF': 0, 'Ours': 0, 'missing': []}
    stats = {'Pretrained': 0, 'missing': []}
    
    print("="*70)
    print("IMAGE ORGANIZATION SCRIPT")
    print("="*70)
    
    # Process each source directory
    for dest_folder, source_path in source_dirs.items():
        dest_path = os.path.join(temp_dir, dest_folder)
        os.makedirs(dest_path, exist_ok=True)
        
        print(f"\nProcessing {dest_folder}...")
        print(f"Source: {source_path}")
        
        if not os.path.exists(source_path):
            print(f"  ⚠ WARNING: Source directory does not exist!")
            continue
        
        # Get all files in source directory
        try:
            all_files = os.listdir(source_path)
            print(f"  Found {len(all_files)} files in source directory")
        except Exception as e:
            print(f"  ✗ ERROR: Cannot read directory - {e}")
            continue
        
        # For each pattern, find matching files
        for pattern in image_patterns:
            found = False
            for filename in all_files:
                # Check if filename starts with pattern and has image extension
                if filename.startswith(pattern):
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in [e.lower() for e in image_extensions]:
                        source_file = os.path.join(source_path, filename)
                        dest_file = os.path.join(dest_path, filename)
                        
                        try:
                            shutil.copy2(source_file, dest_file)
                            print(f"  ✓ Copied {filename}")
                            stats[dest_folder] += 1
                            found = True
                            break
                        except Exception as e:
                            print(f"  ✗ ERROR copying {filename}: {e}")
            
            if not found:
                stats['missing'].append(f"{dest_folder}/{pattern}")
    
    # Create zip file
    print(f"\n{'='*70}")
    print(f"Creating zip file: {output_zip}")
    print(f"{'='*70}")
    
    try:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
                    print(f"  → {arcname}")
        
        zip_size = os.path.getsize(output_zip)
        print(f"\n✓ Zip file created successfully ({zip_size / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"\n✗ ERROR creating zip file: {e}")
    
    # Cleanup temporary directory
    try:
        shutil.rmtree(temp_dir)
        print("✓ Cleaned up temporary files")
    except Exception as e:
        print(f"⚠ Warning: Could not remove temporary directory: {e}")
    
    # Print statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Pretrained: {stats['Pretrained']:3d} images copied")
    # print(f"DDPO:       {stats['DDPO']:3d} images copied")
    # print(f"B2DIFF:     {stats['B2DIFF']:3d} images copied")
    # print(f"Ours:       {stats['Ours']:3d} images copied")
    # print(f"{'─'*70}")
    # print(f"Total:      {stats['Pretrained'] + stats['DDPO'] + stats['B2DIFF'] + stats['Ours']:3d} images")
    
    if stats['missing']:
        print(f"\n⚠ Missing images ({len(stats['missing'])}):")
        for missing in stats['missing']:
            print(f"  - {missing}")
    
    print(f"\n{'='*70}")
    print(f"✓ Complete! Zip file saved as: {os.path.abspath(output_zip)}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()