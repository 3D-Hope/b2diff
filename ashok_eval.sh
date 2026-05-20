cd /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion

# RESULT_TAG=universal_only EXPERIMENT_TAG=may_19_universal_only CHECKPOINT=last ./scripts/run_generate.sh
PYTHONPATH=. python scripts/ashok_generate_results.py \
  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout_custom_attn/best_model.pt\
  --result_tag universal_only \
  --n_syn_scenes 100 \
  --batch_size 32 \
  --lora /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/model/lora/may_19_universal_only/stage0/checkpoints/checkpoint_1/lora_weights.pt
RESULT_TAG=universal_only ROOM_TYPE=bedroom ./scripts/run_render.sh
RESULT_TAG=universal_only ROOM_TYPE=bedroom ./scripts/run_evaluate.sh ⁠
