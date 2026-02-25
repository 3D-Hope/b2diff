cd /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion
PYTHONPATH=. python scripts/generate_results.py model.pt --result_tag test --n_syn_scenes 128

PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test/results.pkl  --no_texture --without_floor --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test/results.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag pretrained_3d_layout_custom_attn --with_wandb_logger
PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test --with_wandb_logger --overfit_test


PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag test --n_syn_scenes 128 --batch_size 64



# ---
run_name="3d_b2"
python3 ./scripts/training/train_pipeline.py \
    exp_name="3d_b2" \
    seed=42 \
    sample.batch_size=32 \
    train.batch_size=32 \
    sample.num_batches_per_epoch=16 \
    wandb.enabled=true \
    threed_scene_layout=true


PYTHONPATH=. python scripts/train_diffusion.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test


PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/model_06000 \
    --result_tag test --n_syn_scenes 128 --batch_size 64 --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/3d_b2/stage32/checkpoints/checkpoint_1/lora_weights.pt