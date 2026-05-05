#!/bin/bash
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate b2
rm -rf ./model/lora/test

# python ./scripts/training/train_pipeline.py sample.batch_size=1 train.batch_size=1 sample.num_batches_per_epoch=1 train.incremental_training=false exp_name=test wandb.enabled=false sample.no_branching=true split_time=3 sample.no_selection=false sample.num_steps=5
# python3 ./scripts/training/train_pipeline.py \
#     exp_name=test \
#     train.incremental_training=true \
#     seed=42 \
#     sample.no_branching=false \
#     sample.no_selection=false \
#     sample.fk=false \
#     split_time=3 \
#     sample.batch_size=1 \
#     train.batch_size=1 \
#     sample.num_batches_per_epoch=1 
# python ./scripts/training/train_pipeline.py sample.batch_size=1 train.batch_size=1 sample.num_batches_per_epoch=1 train.progressive_incremental_training=true exp_name=test

# 1 451 701 251 951
run_name="test"
# python3 ./scripts/training/train_pipeline.py \
#     exp_name="${run_name}" \
#     train.incremental_training=true \
#     seed=42 \
#     sample.no_branching=false \
#     sample.no_selection=false \
#     split_time=3 \
#     sample.batch_size=1 \
#     train.batch_size=1 \
#     wandb.enabled=false \
#     sample.num_batches_per_epoch=1
# python3 ./scripts/training/train_pipeline.py \
#     exp_name="${run_name}" \
#     train.incremental_training=true \
#     sample.fk=true \
#     sample.only_best_fk=true \
#     sample.fk_mix_ratio=1 \
#     seed=42 \
#     sample.no_branching=false \
#     sample.no_selection=false \
#     split_time=4 \
#     sample.batch_size=1 \
#     train.batch_size=16 \
#     sample.num_batches_per_epoch=100
# python3 ./scripts/training/train_pipeline.py \
#     exp_name="test" \
#     train.incremental_training=true \
#     train.score_fn_training=false \
#     sample.fk=true \
#     sample.num_particles=1 \
#     sample.normalize_all=true \
#     sample.only_best_fk=true \
#     sample.fk_mix_ratio=1 \
#     seed=42 \
#     sample.no_branching=false \
#     sample.no_selection=false \
#     split_time=4 \
#     sample.batch_size=2 \
#     train.batch_size=4 \
#     sample.num_batches_per_epoch=1 \
#     wandb.enabled=false


# run_name="b2"
# python3 ./scripts/training/train_pipeline.py \
#     exp_name="${run_name}" \
#     train.incremental_training=true \
#     sample.fk=true \
#     sample.only_best_fk=true \
#     sample.fk_mix_ratio=1 \
#     seed=42 \
#     sample.no_branching=false \
#     sample.no_selection=false \
#     split_time=4 \
#     sample.batch_size=1 \
#     train.batch_size=100 \
#     sample.num_batches_per_epoch=125 \
#     pipeline.continue_from_stage=35 \
#     wandb.enabled=false
run_name="test_always_split_15_b2"
python3 ./scripts/training/train_pipeline.py \
    exp_name="${run_name}" \
    train.bpt="default" \
    sample.always_branch_at=15 \
    seed=42 \
    split_time=3 \
    sample.batch_size=3 \
    train.batch_size=4 \
    sample.num_batches_per_epoch=16 \
    train.learning_rate=3e-4 \
    train.max_grad_norm=0.005 \