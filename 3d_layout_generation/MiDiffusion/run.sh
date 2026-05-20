cd /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion
PYTHONPATH=. python scripts/generate_results.py model.pt --result_tag original_midiffusion --n_syn_scenes 1080 --batch_size 512

PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/finetuned_2k/results.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/tv_bed_top_of_universal/stage130/results.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test --with_wandb_logger
PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test --with_wandb_logger --overfit_test


PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/finetune_from_filtered_try2/model_00200 \
    --result_tag finetuned_try2_200 --n_syn_scenes 1080 --batch_size 512




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

# 
PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag study_march22 --n_syn_scenes 10 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/study_only/stage50/checkpoints/checkpoint_1/lora_weights.pt


python ../ThreedFront/scripts/render_results.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/pretrained_1000steps/results.pkl --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

# 

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag pretrained_1000steps --n_syn_scenes 50 --batch_size 512 --num_denoising_steps 1000

# ---

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag curriculum_tv_bed_study_124 --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/curriculum_tv_bed_study/stage124/checkpoints/checkpoint_1/lora_weights.pt

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 8_particles_incremental_fk --n_syn_scenes 128 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/8_particles_incremental_fk/stage38/checkpoints/checkpoint_1/lora_weights.pt


PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 4_particles_incremental_branch_fk --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/4_particles_incremental_branch_fk/stage42/checkpoints/checkpoint_1/lora_weights.pt

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag 4_particles_incremental_fk --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/4_particles_incremental_fk/stage48/checkpoints/checkpoint_1/lora_weights.pt


# ----

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag non_curriculum_tv_bed_study --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/non_curriculum_tv_bed_study/stage199/checkpoints/checkpoint_1/lora_weights.pt

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/finetuned_try2_200/results.pkl --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


-
python ../ThreedFront/scripts/render_results.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/study_simul_uni_99/results.pkl --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

python ../ThreedFront/scripts/render_results.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/study_simul_uni_99/results.pkl --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/lambda_10_4_particles_inference_time_fk/stage0/final_best_samples.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/8_particles_inference_time_fk/stage0/final_best_samples.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl



# ---
python scripts/visualize_intermediate_steps.py     output/log/pretrained_3d_layout_custom_attn/model_06000     --num_samples 4     --num_denoising_steps 20     --output_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/traj_viz     --gpu 0


python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/finetuned_6k/results.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


# ---
PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/finetune_from_filtered/model_02000 --result_tag finetuned_2k --n_syn_scenes 1080 --batch_size 2048

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

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --result_tag tv_bed_top_of_universal --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/tv_bed_top_of_universal/stage99/checkpoints/checkpoint_1/lora_weights.pt

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



    ---

PYTHONPATH=. python scripts/ashok_generate_results_with_rejection.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag tv_bed_universal_rejection_samp --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/tv_bed_top_of_universal/stage130/checkpoints/checkpoint_1/lora_weights.pt  --reward_file ../../core/custom_rewards/tv_bed.py --reward_threshold -4 

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test_pretrained_6k/results.pkl --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


PYTHONPATH=. python scripts/ashok_generate_results_with_rejection.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag tv_bed_only_rejection --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/universal_only_oob_area/stage92/checkpoints/checkpoint_1/lora_weights.pt --reward_file ../../core/custom_rewards/tv_bed.py --reward_threshold -4



PYTHONPATH=. python scripts/ashok_generate_results_with_rejection.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000     --result_tag robot_1m_universal_rejection --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/robot_fetch_1m_high_top_of_universal/stage110/checkpoints/checkpoint_1/lora_weights.pt --reward_file ../../core/custom_rewards/robot_fetch_from_table_1m_high.py --reward_threshold 2 --use_universal_pre_rejection 

PYTHONPATH=. python scripts/ashok_generate_results_with_rejection.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000     --result_tag non_curriculum_tv_bed_study_ --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/study_simul_uni/stage99/checkpoints/checkpoint_1/lora_weights.pt  --reward_file ../../core/custom_rewards/desk_chair_for_study.py --reward_threshold 2 --use_universal_pre_rejection 


python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/AshokSaugatResearch/ATISS/training-outputs/atiss_baseline/metrics_smoke/results/sampled_scenes_results.pkl --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


PYTHONPATH=. python scripts/ashok_generate_results_with_rejection.py \
  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
  --result_tag non_curriculum_tv_bed_study_114 \
  --n_syn_scenes 1080 \
  --batch_size 512 \
  --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/non_curriculum_tv_bed_study/stage114/checkpoints/checkpoint_1/lora_weights.pt \
  --reward_files ../../core/custom_rewards/desk_chair_for_study.py ../../core/custom_rewards/tv_bed.py \
  --reward_thresholds 2 -4



  ---

PYTHONPATH=. python scripts/ashok_generate_results_with_rejection.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \ 
--result_tag good_samples_pretrained --batch_size 4096 --use_universal_pre_rejection --reward_file ../../core/custom_rewards/tv_bed.py --reward_threshold -4


---
PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag finetune_from_filtered_try2  --synthetic_only --weight_file /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --with_wandb_logger


/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/tv_bed_only_rejection/results.pkl

/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/tv_bed_only_97/results.pkl




PYTHONPATH=. python scripts/ashok_generate_results_with_rejection.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000     --result_tag non_curriculum_tv_bed_study_ --n_syn_scenes 1080 --batch_size 512 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/study_simul_uni/stage99/checkpoints/checkpoint_1/lora_weights.pt  --reward_file ../../core/custom_rewards/desk_chair_for_study.py --reward_threshold 2 --use_universal_pre_rejection 


# ---
python test_random_floor_condition.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/universal_only_oob_area/stage92/checkpoints/checkpoint_1/lora_weights.pt --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --num_scenes 4 --batch_size 128 --num_denoising_steps 20 --output_dir user_app_outputs/test_random_floor --seed 42


# may 18

PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrain_bedroom_v2/model_37400 --result_tag pretrain_bedroom_v2_37400 --n_syn_scenes 1024 --batch_size 512



python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/pretrain_bedroom_v2_37400/results.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl


PYTHONPATH=. python scripts/ashok_generate_results.py \
  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrain_bedroom_theta/model_10400 \
  --result_tag bedroom_theta \
  --n_syn_scenes 100 \
  --batch_size 32 \

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/bedroom_theta/results.pkl
  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

  
