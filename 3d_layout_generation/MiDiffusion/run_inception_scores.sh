#!/bin/bash
# names=(b2_tv_bed
# inc_b2_tv_bed
# 4_particles_incremental_branch_fk_tv_bed
# 4_particles_incremental_fk_tv_bed)

# BASE=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results

# for name in "${names[@]}"; do
#     echo "=== Running inception score for: $name ==="
#     python scripts/inference/run_inception_score.py \
#         --img_dir "$BASE/$name"
# done

# python scripts/inference/run_inception_score.py \
#     --img_dir /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test_pretrained_6k

python scripts/inference/run_inception_score.py \
    --img_dir  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/ddpo

python scripts/inference/run_inception_score.py \
    --img_dir  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/3d_b2_lora_16_test


python scripts/inference/run_inception_score.py \
    --img_dir /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/inc_b2

python scripts/inference/run_inception_score.py \
    --img_dir  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/8_particles_incremental_branch_fk

python scripts/inference/run_inception_score.py \
    --img_dir  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/8_particles_incremental_fk



python scripts/inference/run_inception_score.py \
    --img_dir  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/4_particles_incremental_fk_tv_bed

python scripts/inference/run_inception_score.py \
    --img_dir  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/lambda_10_4_particles_inference_time_fk/stage0

python scripts/inference/run_inception_score.py \
    --img_dir  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/4_particles_inference_time_fk/stage0


python scripts/inference/run_inception_score.py \
    --img_dir  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/incremental_fk