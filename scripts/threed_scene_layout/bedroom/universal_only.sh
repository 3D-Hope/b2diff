run_name="may_19_universal_only"
python3 ./scripts/training/train_pipeline.py \
    exp_name=${run_name} \
    seed=42 \
    sample.batch_size=32 \
    train.batch_size=256 \
    sample.num_batches_per_epoch=16 \
    wandb.enabled=true \
    threed_scene_layout=true \
    sample.no_branching=true \
    sample.no_selection=true \
    split_time=1 \
    sample.normalize_all=false \
    train.incremental_training=true \
    sample.num_steps=20 \
    train.num_stages_per_increment=10 \
    universal_rewards=true \
    midiffusion.checkpoint_path="/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/best_model.pt"