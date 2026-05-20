run_name="study_only"
python3 ./scripts/training/train_pipeline.py \
    exp_name=${run_name} \
    seed=42 \
    sample.batch_size=32 \
    train.batch_size=32 \
    sample.num_batches_per_epoch=16 \
    wandb.enabled=true \
    threed_scene_layout=true \
    sample.no_branching=true \
    sample.no_selection=true \
    split_time=1 \
    sample.normalize_all=true \
    train.incremental_training=true \
    sample.num_steps=20 \
    train.num_stages_per_increment=20 \
    custom_reward="desk_chair_for_study" \