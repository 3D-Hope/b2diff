run_name="b2_3d_tv_bed"
python3 ./scripts/training/train_pipeline.py \
    exp_name=${run_name} \
    seed=42 \
    sample.batch_size=32 \
    train.batch_size=32 \
    sample.num_batches_per_epoch=16 \
    wandb.enabled=true \
    threed_scene_layout=true \
    sample.num_steps=150 \
    pipeline.split_step_left=100 \
    pipeline.split_step_right=150 \
    tv_bed=true
