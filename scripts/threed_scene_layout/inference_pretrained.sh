run_name="pretrained_150"
python3 ./scripts/training/train_pipeline.py \
    exp_name=${run_name} \
    seed=42 \
    sample.batch_size=45 \
    train.batch_size=32 \
    sample.num_batches_per_epoch=24 \
    sample.save_train_samples_no_train=true \
    threed_scene_layout=true \
    train.incremental_training=true \
    split_time=1 \
    wandb.enabled=false \
    sample.num_steps=150 \
    
    # sample.save_per_sample_trajectories=true \
    # tv_bed=true
    # sample.fk=true \
    # sample.num_particles=4 \
    # sample.only_best_fk=true \
    # sample.fk_mix_ratio=1 \
    # sample.potential_type="max" \
    # sample.fk_lambda=10.0 \
    # sample.resample_frequency=3 \
    # sample.resampling_t_start=13 \
    # sample.resampling_t_end=19 \
    # sample.save_train_samples_no_train=true \
    # sample.brach_at_before_fk=5 \
    # wandb.enabled=false \
  
# ---
# run_name="tv_bed_pretrained"
# python3 ./scripts/training/train_pipeline.py \
#     exp_name=${run_name} \
#     seed=42 \
#     sample.batch_size=45 \
#     train.batch_size=32 \
#     sample.num_batches_per_epoch=6 \
#     threed_scene_layout=true \
#     train.incremental_training=true \
#     sample.save_train_samples_no_train=true \
#     sample.fk=true \
#     sample.num_particles=4 \
#     sample.only_best_fk=true \
#     sample.fk_mix_ratio=1 \
#     sample.potential_type="max" \
#     sample.fk_lambda=10.0 \
#     sample.resample_frequency=4 \
#     sample.resampling_t_start=4 \
#     sample.resampling_t_end=19 \
#     sample.save_train_samples_no_train=true \
#     pipeline.continue_from_stage=99 # TODO: USE CORRECT RUN NAME AND STAGE NUMBER
#     # sample.brach_at_before_fk=5 \ # TODO: IF YOU WANT BRANCH THEN FK, UNCOMMENT AND CHANGE THE FOLLOWING 
#     #    sample.resample_frequency=4 \
#     # sample.resampling_t_start=8 \
#     # sample.resampling_t_end=19 \
  
