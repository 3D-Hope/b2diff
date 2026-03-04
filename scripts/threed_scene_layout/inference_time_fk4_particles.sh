run_name="lambda_10_4_particles_inference_time_fk_tv_bed"
python3 ./scripts/training/train_pipeline.py \
    exp_name=${run_name} \
    seed=42 \
    sample.batch_size=45 \
    train.batch_size=32 \
    sample.num_batches_per_epoch=6 \
    threed_scene_layout=true \
    train.incremental_training=true \
    sample.fk=true \
    sample.num_particles=4 \
    sample.only_best_fk=true \
    sample.fk_mix_ratio=1 \
    sample.potential_type="max" \
    sample.fk_lambda=10.0 \
    sample.resample_frequency=3 \
    sample.resampling_t_start=13 \
    sample.resampling_t_end=19 \
    sample.save_train_samples_no_train=true \
    tv_bed=true
    # sample.brach_at_before_fk=5 \
    # wandb.enabled=false \
  
