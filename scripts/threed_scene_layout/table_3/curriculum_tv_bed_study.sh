run_name="curriculum_tv_bed_study_new_study"
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
    universal_rewards=true \
    'custom_reward=[tv_bed,desk_chair_for_study]' \
    pipeline.stage_cnt=200 \
    continue_from_universal=true \
    path_to_universal_lora=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/tv_bed_top_of_universal/stage130/checkpoints/checkpoint_1
    
    # GET THE LORA PATH AFTER CHOOSING THE OPERATING POINT ON TV BED