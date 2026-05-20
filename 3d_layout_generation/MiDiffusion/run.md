python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/universal_only_oob_area/stage92/results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --output_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/ours_3d_results --export_glb --without_walls


python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test_pretrained_6k/results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --output_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/mi_3d_results --export_glb --without_walls

python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/finetuned_try2_200/results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --output_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/mi_finetuned_2_3d_results --export_glb --without_walls


python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/tv_bed_only_rejection/results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --output_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/ours_tv_bed --export_glb --without_walls

python ../ThreedFront/scripts/render_results_better.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/robot_1m_only_rejection/results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --output_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/ours_robot_height_1m --export_glb --without_walls

<!-- iARCS Dataset -->

<!-- Pretrained Livingroom -->
Run Tag: pretrain_livingroom_v2

<!-- Generation -->
PYTHONPATH=. python scripts/ashok_generate_results.py \
  output/log/pretrain_livingroom_v2/best_model.pt \
  --result_tag pretrain_livingroom_v2 \
  --n_syn_scenes 1000 \
  --batch_size 32 \
  --gpu 0

<!-- Rendering -->
[NO SV SHARE : BLOCKER]
conda activate midiffusion && cd /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/ThreedFront &&python scripts/render_results.py \
  ../MiDiffusion/output/predicted_results/pretrain_livingroom_v2/results.pkl \
  --path_to_pickled_3d_future_model \
  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/ThreedFront/output/threed_future_model_livingroom.pkl \
  --no_texture \
  --output_directory ../MiDiffusion/output/predicted_results/pretrain_livingroom_v2/renders



<!-- Pretrained Bedroom -->
MD=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion && TF=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/ThreedFront && BED_DATA=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/bedroom &&LIV_DATA=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/livingroom

<!-- Normal -->
<!-- Generation -->
cd "$MD" && PYTHONPATH=. python scripts/ashok_generate_results.py \
  output/log/pretrained_3d_layout_custom_attn/best_model.pt \
  --result_tag bedroom_cos_sin_pretrained_2 --n_syn_scenes 1000 --batch_size 32 --gpu 0

cd "$MD" && PYTHONPATH=. python scripts/ashok_generate_results.py \
  output/log/pretrain_bedroom_v2/best_model.pt \
  --result_tag bedroom_cos_sin_v2 --n_syn_scenes 1000 --batch_size 32 --gpu 0

<!-- Rendering -->
cd "$TF" && python scripts/render_results.py \
  "$MD/output/predicted_results/bedroom_cos_sin_pretrained_2/results.pkl" \
  --path_to_pickled_3d_future_model "$TF/output/threed_future_model_bedroom.pkl" \
  --no_texture \
  --output_directory "$MD/output/predicted_results/bedroom_cos_sin_pretrained_2/renders"

<!-- Evaluation -->

<!-- CKL -->
cd "$TF"
python scripts/evaluate_kl_divergence_object_category.py \
  "$MD/output/predicted_results/bedroom_cos_sin_pretrained/results.pkl" \
  --output_directory "$MD/output/predicted_results/bedroom_cos_sin_pretrained/kl_stats"

<!-- OOB stats -->
cd "$TF"
python scripts/bbox_analysis.py "$MD/output/predicted_results/bedroom_cos_sin_pretrained/results.pkl"

<!-- Physcene Metrics -->
cd "$TF"
python scripts/physcene_metrics.py "$MD/output/predicted_results/bedroom_cos_sin_pretrained/results.pkl"

<!-- FID KID -->
cd "$TF"
python scripts/compute_fid_scores.py \
  "$MD/output/predicted_results/bedroom_cos_sin_pretrained/results.pkl" \
  --output_directory "$MD/output/predicted_results/bedroom_cos_sin_pretrained/fid_tmps" \
  --no_texture \
  --dataset_directory $BED_DATA \
  --synthesized_directory "$MD/output/predicted_results/bedroom_cos_sin_pretrained/renders"

cd "$TF"
python scripts/compute_fid_scores.py \
  "$MD/output/predicted_results/bedroom_cos_sin_pretrained/results.pkl" \
  --output_directory "$MD/output/predicted_results/bedroom_cos_sin_pretrained/fid_tmps" \
  --no_texture \
  --dataset_directory $BED_DATA \
  --synthesized_directory "$MD/output/predicted_results/bedroom_cos_sin_pretrained/renders" \
  --compute_kid

<!-- SCA -->
cd "$TF"
python scripts/synthetic_vs_real_classifier.py \
  "$MD/output/predicted_results/bedroom_cos_sin_pretrained/results.pkl" \
  --output_directory "$MD/output/predicted_results/bedroom_cos_sin_pretrained/classifier_tmps" \
  --no_texture \
  --dataset_directory "$BED_DATA" \
  --synthesized_directory "$MD/output/predicted_results/bedroom_cos_sin_pretrained/renders"


<!-- Theta -->
RESULT_TAG=bedroom_theta_check EXPERIMENT_TAG=pretrain_bedroom_theta CHECKPOINT=model_28000 ./scripts/run_generate.sh
RESULT_TAG=bedroom_theta_check ROOM_TYPE=bedroom ./scripts/run_render.sh
RESULT_TAG=bedroom_theta_check ROOM_TYPE=bedroom ./scripts/run_evaluate.sh ⁠