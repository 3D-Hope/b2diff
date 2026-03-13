cd /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion
PYTHONPATH=. python scripts/generate_results.py model.pt --result_tag test --n_syn_scenes 128

PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/test_150_inf/stage0/sample_00003/result.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

python ../ThreedFront/scripts/render_results.py /tmp/vis_steps/sample_000.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag pretrained_3d_layout_custom_attn --with_wandb_logger
PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test --with_wandb_logger --overfit_test


PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag test_pretrained_6k --n_syn_scenes 1080 --batch_size 512



# ---
run_name="3d_b2"
python3 ./scripts/training/train_pipeline.py \
    exp_name="test" \
    seed=42 \
    sample.batch_size=32 \
    train.batch_size=32 \
    sample.num_batches_per_epoch=16 \
    wandb.enabled=true \
    threed_scene_layout=true \



PYTHONPATH=. python scripts/train_diffusion.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test


PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag ddpo_36stage --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/ddpo/stage36/checkpoints/checkpoint_1/lora_weights.pt

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag pretrained --n_syn_scenes 1080 --batch_size 512 --num_denoising_steps 150

# ---

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag ddpo_tv_bed --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/ddpo_tv_bed/stage80/checkpoints/checkpoint_1/lora_weights.pt


PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 8_particles_incremental_fk --n_syn_scenes 128 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/8_particles_incremental_fk/stage38/checkpoints/checkpoint_1/lora_weights.pt


PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 4_particles_incremental_branch_fk --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/4_particles_incremental_branch_fk/stage42/checkpoints/checkpoint_1/lora_weights.pt

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 4_particles_incremental_fk --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/4_particles_incremental_fk/stage48/checkpoints/checkpoint_1/lora_weights.pt


# ----
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/ddpo_tv_bed/results.pkl --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


-
python ../ThreedFront/scripts/render_results.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/8_particles_incremental_fk/results.pkl --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/lambda_10_4_particles_inference_time_fk/stage0/final_best_samples.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/8_particles_inference_time_fk/stage0/final_best_samples.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl



# ---
python scripts/visualize_intermediate_steps.py     output/log/pretrained_3d_layout_custom_attn/model_06000     --num_samples 4     --num_denoising_steps 20     --output_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/traj_viz     --gpu 0


python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/traj_viz/sample_000.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


# ---
PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag b2_tv_bed --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/b2_tv_bed/stage66/checkpoints/checkpoint_1/lora_weights.pt

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag inc_b2_tv_bed --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/inc_b2_tv_bed/stage91/checkpoints/checkpoint_1/lora_weights.pt

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 4_particles_incremental_branch_fk_tv_bed --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/4_particles_incremental_branch_fk_tv_bed/stage83/checkpoints/checkpoint_1/lora_weights.pt

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 4_particles_incremental_fk_tv_bed --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/4_particles_incremental_fk_tv_bed/stage89/checkpoints/checkpoint_1/lora_weights.pt

# ---

names=(b2_tv_bed
inc_b2_tv_bed
4_particles_incremental_branch_fk_tv_bed
4_particles_incremental_fk_tv_bed)
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/{name}/results.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/{name}/results.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/b2_tv_bed/stage76/results.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/4_particles_incremental_branch_fk_tv_bed/stage50/results.pkl  --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


# ---
names=(b2_tv_bed
inc_b2_tv_bed
4_particles_incremental_branch_fk_tv_bed
4_particles_incremental_fk_tv_bed)
./eval.sh /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/{name}/results.pkl




# 
python scripts/inference/run_inception_score.py --img_dir /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/b2_tv_bed/stage76/


python scripts/inference/run_inception_score.py --img_dir /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/ddpo_tv_bed/stage92/


/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/4_particles_incremental_branch_fk_tv_bed

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/ddpo_tv_bed/stage92/

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/b2_tv_bed/stage76/





/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test_pretrained_6k/results.pkl

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/lambda_10_4_particles_inference_time_fk_tv_bed/stage0/final_best_samples.pkl

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/ddpo_tv_bed/stage92/results.pkl

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/b2_tv_bed/stage76/results.pkl

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/4_particles_incremental_branch_fk_tv_bed/results.pkl


# 
python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test_render_pretrain.pkl  --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test_pretrained_6k/results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/ddpo_tv_bed/stage92/results.pkl  --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


# 
python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test_render_pretrain.pkl  --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --export_glb


python ../ThreedFront/scripts/render_results_better.py {pkl_file}  --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --export_glb


# /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/tmp_for_rendering/pretrained/results.pkl

# /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/tmp_for_rendering/ddpo/results.pkl



# /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/tmp_for_rendering/b2/results.pkl

# /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/tmp_for_rendering/ours/results.pkl



# ---
/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test_pretrained_6k/results.pkl

  

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/lambda_10_4_particles_inference_time_fk_tv_bed/stage0/final_best_samples.pkl

  

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/ddpo_tv_bed/stage92/results.pkl

  

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/b2_tv_bed/stage76/results.pkl

  

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/4_particles_incremental_branch_fk_tv_bed/stage50/results.pkl

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 4_particles_incremental_fk_tv_bed --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/4_particles_incremental_branch_fk_tv_bed/stage99/checkpoints/checkpoint_1/lora_weights.pt

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 4_particles_incremental_fk_tv_bed --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/b2_tv_bed/stage76/checkpoints/checkpoint_1/lora_weights.pt


PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 4_particles_incremental_fk_tv_bed --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/ddpo_tv_bed/stage92/checkpoints/checkpoint_1/lora_weights.pt

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 4_particles_incremental_fk_tv_bed --n_syn_scenes 1080 --batch_size 512


# ---
python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/tmp_for_rendering/pretrained/results.pkl \
    --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl\
    --export_glb --without_walls --without_door



python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/tmp_for_rendering/b2/results.pkl \
    --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl\
    --export_glb --without_walls --without_door

cp /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/b2_tv_bed/stage76/results.pkl /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/tmp_for_rendering/b2/results.pkl

python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/tmp_for_rendering/ddpo/results.pkl \
    --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl\
    --export_glb --without_walls --without_door



python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/tmp_for_rendering/ours/results.pkl \
    --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl\
    --export_glb --without_walls --without_door