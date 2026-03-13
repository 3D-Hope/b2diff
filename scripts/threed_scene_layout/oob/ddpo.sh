run_name="ddpo_3d_collision"
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
    train.only_train_steps=150 \
    train.incremental_training=true \
    sample.normalize_all=true \
    sample.num_steps=150 \
