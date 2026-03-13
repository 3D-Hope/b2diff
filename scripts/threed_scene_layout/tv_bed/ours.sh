run_name="ours_3d_tv_bed"
python3 ./scripts/training/train_pipeline.py \
    exp_name=${run_name} \
    seed=42 \
    sample.batch_size=32 \
    train.batch_size=32 \
    sample.num_batches_per_epoch=16 \
    wandb.enabled=true \
    threed_scene_layout=true \
    train.incremental_training=true \
    sample.fk=true \
    sample.num_particles=4 \
    sample.only_best_fk=true \
    sample.fk_mix_ratio=1 \
    sample.potential_type="max" \
    sample.fk_lambda=2.0 \
    sample.resample_frequency=10 \
    sample.resampling_t_start=110 \
    sample.resampling_t_end=150 \
    sample.brach_at_before_fk=75 \
    sample.num_steps=150 \
    tv_bed=true
    
