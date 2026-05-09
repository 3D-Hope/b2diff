# Generate images from LoRA checkpoint and compute Inception Score
# python scripts/inference/run_inception_score.py \
#     --mode lora \
#     --checkpoint_path /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/b2diffu_try2/ckpts/stage88/checkpoints/checkpoint_1/pytorch_lora_weights.safetensors \
#     --output_dir /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/b2diffu_try2/inference_results_88_ckpt \
#     --num_images 1000 \
#     --gen_batch_size 4 \
#     --eval_batch_size 32 \
#     --splits 10 \
#     --seed 42 \
#     --num_inference_steps 20




# python scripts/inference/inference_lora_clip_reward.py \
#     --checkpoint_path /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/b2diff/outputs/b2diffu_try2/ckpts/stage88/checkpoints/checkpoint_1/pytorch_lora_weights.safetensors\
#     --output_dir outputs/quick_eval \
#     --num_images 32 \
#     --batch_size 4

# python3 ./scripts/training/train_pipeline.py \
#     exp_name=test70 \
#     seed=42 \
#     split_time=8 \
#     sample.batch_size=2 \
#     train.batch_size=1 \
#     sample.num_batches_per_epoch=1 \
#     train.learning_rate=3e-4 \
#     train.max_grad_norm=0.005 \
#     train.incremental_training=false \
#     sample.no_branching=true \
#     sample.no_selection=true \
#     prompt_file=configs/prompt/template1_train.json \
#     pipeline.use_grpo=true \
#     wandb.enabled=false



# python3 ./scripts/training/train_pipeline.py \
#     exp_name=test123 \
#     seed=42 \
#     split_time=1 \
#     sample.batch_size=4 \
#     train.batch_size=2 \
#     sample.num_batches_per_epoch=1 \
#     train.learning_rate=3e-4 \
#     train.max_grad_norm=0.005 \
#     train.incremental_training=false \
#     sample.no_branching=true \
#     sample.no_selection=true \
#     train.incremental_timesteps=[4,8,12,16] \
#     train.num_stages_per_increment=10 \
#     prompt_file=configs/prompt/template2_train.json

# python3 ./scripts/training/train_pipeline.py \
#     exp_name=test70 \
#     seed=42 \
#     sample.batch_size=1 \
#     train.batch_size=1 \
#     sample.num_batches_per_epoch=1 \
#     wandb.enabled=false \
#     train.incremental_training=true \
#     sample.no_branching=false \
#     sample.no_selection=true \
#     prompt_file=configs/prompt/template1_train.json \
#     pipeline.use_iadd_grpo=true \
#     sample.fk=true \
#     sample.num_particles=2 \
#     sample.only_best_fk=true \
#     sample.fk_mix_ratio=1 \
#     sample.potential_type="max" \
#     sample.fk_lambda=2.0 \
#     sample.resample_frequency=4 \
#     sample.resampling_t_start=8 \
#     sample.resampling_t_end=16 \
#     sample.brach_at_before_fk=5 \
#     train.incremental_timesteps=[5,10,15,20] \
#     train.num_stages_per_increment=10



python3 ./scripts/training/train_pipeline.py \
    exp_name=test68 \
    seed=42 \
    sample.batch_size=1 \
    train.batch_size=1 \
    sample.num_batches_per_epoch=2 \
    train.learning_rate=3e-4 \
    train.max_grad_norm=0.005 \
    train.incremental_training=true \
    sample.no_branching=false \
    sample.no_selection=true \
    prompt_file=configs/prompt/template1_train.json \
    pipeline.use_branch_grpo=true \
    wandb.enabled=false \
    branch_grpo.split_points=[0,4] \
    branch_grpo.edge_microbatch_size=1

