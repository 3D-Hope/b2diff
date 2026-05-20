
<!-- Pickling Data -->
python scripts/pickle_threed_front_dataset.py /mnt/sv-share/3DFRONT/data/3D-FRONT /mnt/sv-share/3DFRONT/data/3D-FUTURE-model /mnt/sv-share/3DFRONT/data/3D-FUTURE-model/model_info.json

python scripts/pickle_threed_front_dataset.py /mnt/sv-share/3DFRONT/data/3D-FRONT /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3dfuture_data/3D-FUTURE-model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3dfuture_data/3D-FUTURE-model/model_info.json

python scripts/pickle_threed_future_dataset.py threed_front_bedroom
python scripts/pickle_threed_future_dataset.py threed_front_livingroom


python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-27/10-12-32/sampled_scenes_results.pkl
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-27/10-15-17/sampled_scenes_results.pkl
<!-- Preprocessing -->

python scripts/preprocess_data.py threed_front_bedroom --add_objfeats --no_texture

python scripts/preprocess_data.py threed_front_livingroom --no_texture --output_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/livingroom

<!-- For no texture and no floor (Select 1 if need to replace) --> 
python scripts/preprocess_data.py threed_front_bedroom --no_texture --output_directory /mnt/sv-share/MiData
python scripts/test_preprocess.py threed_front_bedroom --no_texture --output_directory /mnt/sv-share/MiData

`
<!-- For Floorplan Data -->
python scripts/preprocess_floorplan.py /mnt/sv-share/MiData/bedroom --room_side 3.1
python scripts/preprocess_floorplan_cuboid_scene.py /mnt/sv-share/MiData/test_data --room_side 3.1
python scripts/preprocess_floorplan.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data/bedroom --room_side 3.1

python scripts/preprocess_floorplan.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/livingroom --room_side 6.1

python scripts/preprocess_floorplan.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data/bedroom --room_side 3.1
python scripts/preprocess_floorplan.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/livingrooms_objfeats_32_64 --room_side 6.1
python scripts/preprocess_floorplan.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/livingroom --room_side 6.1


<!-- Test rendering -->
python scripts/render_threedfront_scene.py MasterBedroom-2888 --without_screen --with_walls --with_door_and_windows

<!-- Metrics -->
source ../steerable-scene-generation/.venv/bin/activate

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-27/10-12-32/sampled_scenes_results.pkl --no_texture --without_floor
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-27/10-15-17/sampled_scenes_results.pkl --no_texture --without_floor
<!-- FID -->
python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/b2_tv_bed/results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/livingroom --no_floor

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/livingroom --no_floor
<!-- KID -->
python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-16/10-45-35/sampled_scenes_results.pkl --compute_kid --output_directory ./fid_tmps --no_texture  --dataset_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/livingroom --no_floor

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl --compute_kid --output_directory ./fid_tmps --no_texture  --dataset_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/livingroom --no_floor
<!-- Classifier -->
python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl --output_directory ./classifier_tmps --no_texture --no_floor  --dataset_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/livingroom

python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-16/10-45-35/sampled_scenes_results.pkl --output_directory ./classifier_tmps --no_texture --no_floor  --dataset_directory /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData/livingroom
<!-- OOB and MBL -->
source ../steerable-scene-generation/.venv/bin/activate
python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/pretrained/results.pkl




python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl

python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-16/10-45-35/sampled_scenes_results.pkl

<!-- KL-Divergence -->
source ../steerable-scene-generation/.venv/bin/activate

python scripts/evaluate_kl_divergence_object_category.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/rejection_samp_rl_for_mi/results.pkl --output_directory ./kl_tmps

python scripts/evaluate_kl_divergence_object_category.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl --output_directory ./kl_tmps

source ../steerable-scene-generation/.venv/bin/activate

python scripts/evaluate_kl_divergence_object_category.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-13/05-36-07/sampled_scenes_results.pkl --output_directory ./kl_tmps
<!-- Obj Metric -->
python scripts/calculate_num_obj.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/finetuned_2k/results.pkl

python scripts/calculate_num_obj.py source ../steerable-scene-generation/.venv/bin/activate
python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-30/11-43-57/sampled_scenes_results.pkl


python scripts/physcene_metrics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-02/04-11-45/sampled_scenes_results.pkl



<!-- bedroom -->

<!-- FID -->
source ../steerable-scene-generation/.venv/bin/activate

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/finetuned_try2_200/results.pkl   --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/
<!-- KID -->
python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test_pretrained_6k/results.pkl

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl --compute_kid --output_directory ./fid_tmps --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/
<!-- Classifier -->
python scripts/synthetic_vs_real_classifier.py media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/tv_bed_only_97/results.pkl --output_directory ./classifier_tmps --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

source ../steerable-scene-generation/.venv/bin/activate
python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-11-07/03-45-53/sampled_scenes_results.pkl --output_directory ./classifier_tmps --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

<!-- OOB and MBL -->
source ../steerable-scene-generation/.venv/bin/activate
python scripts/bbox_analysis.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/finetuned_try2_200/results.pkl


python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/finetuned_6k/results.pkl

python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/incremental_fk/results.pkl

python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/incremental_branch_fk/results.pkl


<!-- physcene metrics -->
python ../steerable-scene-generation/scripts/physcene_metrics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/finetuned_try2_200/results.pkl


python scripts/evaluate_kl_divergence_object_category.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-23/19-25-11/sampled_scenes_results.pkl --output_directory ./kl_tmps


<!-- diversity  -->
python scripts/evaluate_scene_statistics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/finetuned_try2_200/results.pkl


<!-- threshold -->
python scripts/evaluate_reward_satisfaction.py \
  --source gt \
  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom \
  --annotation_file dataset_files/bedroom_threed_front_splits.csv \
  --reward_file scripts/tv_bed.py \
  --threshold -4 \
  --comparison ge \
  --save_filtered_results \
  --output_results_file output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl \
  --output_indices_file output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_scene_indices.pkl

<!-- todo -->
python scripts/evaluate_reward_satisfaction.py \
  --source gt \
  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom \
  --annotation_file dataset_files/bedroom_threed_front_splits.csv \
  --reward_file scripts/tv_bed.py \
  --threshold 4 \
  --comparison ge \
  --save_filtered_results \
  --output_results_file output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl \
  --output_indices_file output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_scene_indices.pkl

<!-- todo -->
python scripts/evaluate_reward_satisfaction.py \
  --source gt \
  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom \
  --annotation_file dataset_files/bedroom_threed_front_splits.csv \
  --reward_file scripts/tv_bed.py \
  --threshold 4 \
  --comparison ge \
  --save_filtered_results \
  --output_results_file output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl \
  --output_indices_file output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_scene_indices.pkl


python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/bed_headboard_wall_filtered_gt_results.pkl


python scripts/evaluate_scene_statistics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/bed_headboard_wall_filtered_gt_results.pkl

python ../steerable-scene-generation/scripts/physcene_metrics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/bed_headboard_wall_filtered_gt_results.pkl --filtered_gt


---

source ../steerable-scene-generation/.venv/bin/activate

python scripts/bbox_analysis.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/good_samples_pretrained/results.pkl




python scripts/evaluate_scene_statistics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/good_samples_pretrained/results.pkl



python ../steerable-scene-generation/scripts/physcene_metrics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/good_samples_pretrained/results.pkl


python scripts/evaluate_kl_divergence_object_category.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/AshokSaugatResearch/ATISS/training-outputs/atiss_baseline/metrics_smoke/results/sampled_scenes_results.pkl --output_directory ./kl_tmps



===

SAUGAT
### filtered gt

- bed tv
    
    /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/tv_bed_filtered_gt_results.pkl
    
- study desk chair
    
    /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/desk_chair_for_study_filtered_gt_results.pkl
    
- robot height 1m
    
    /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/robot_fetch_from_table_1m_high_filtered_gt_results.pkl
    

### ours

- tv bed
    
    /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/tv_bed_only_rejection/results.pkl
    
- study desk chair
    
    /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/study_simul_uni_universal_rejection99/results.pkl
    
- robot height 1m
    
    /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/robot_1m_only_rejection/results.pkl

cd /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion

conda activate b2

cd /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront

source ../steerable-scene-generation/.venv/bin/activate



python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/tv_bed_only_rejection/results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/


<!-- Runs i did for FID-->
<!-- Ours vs Original -->

<!-- TV Bed -->
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/tv_bed_only_rejection/results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --no_texture

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/tv_bed_only_rejection/results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/tv_bed_only_rejection/results.pkl --output_directory ./classifier_tmps_2 --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/


<!-- Study -->
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/study_simul_uni_universal_rejection99/results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --no_texture

python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/study_simul_uni_universal_rejection99/results.pkl


python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/study_simul_uni_universal_rejection99/results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/study_simul_uni_universal_rejection99/results.pkl --output_directory ./classifier_tmps_3 --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

FID: 

<!-- Robot -->
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/robot_1m_only_rejection/results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --no_texture

python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/robot_1m_only_rejection/results.pkl

python ../steerable-scene-generation/scripts/physcene_metrics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/robot_1m_only_rejection/results.pkl --filtered_gt

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/robot_1m_only_rejection/results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/robot_1m_only_rejection/results.pkl --output_directory ./classifier_tmps_3 --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

FID: 

<!-- Filtered vs Original -->
mkdir -p /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered
mkdir -p /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/desk_chair_for_study_filtered
mkdir -p /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/robot_fetch_from_table_1m_high_filtered

mv /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/tv_bed_filtered_gt_results.pkl /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered

mv  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/desk_chair_for_study_filtered_gt_results.pkl /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/desk_chair_for_study_filtered

mv /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/robot_fetch_from_table_1m_high_filtered_gt_results.pkl /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/robot_fetch_from_table_1m_high_filtered


<!-- TV Bed -->

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered/tv_bed_filtered_gt_results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --no_texture

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered/tv_bed_filtered_gt_results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

<!-- collision filtered -->

python scripts/evaluate_reward_satisfaction.py \
  --source gt \
  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom \
  --annotation_file dataset_files/bedroom_threed_front_splits.csv \
  --reward_file scripts/tv_bed.py \
  --threshold -4 \
  --comparison ge \
  --save_filtered_results \
  --output_results_file output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl \
  --output_indices_file output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_scene_indices.pkl

python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl

python ../steerable-scene-generation/scripts/physcene_metrics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl --filtered_gt

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --no_texture

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl --output_directory ./classifier_tmps --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

<!-- Study -->

<!-- collision filtered -->

python scripts/evaluate_reward_satisfaction.py \
  --source gt \
  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom \
  --annotation_file dataset_files/bedroom_threed_front_splits.csv \
  --reward_file scripts/desk_chair_for_study.py \
  --threshold 2 \
  --comparison ge \
  --save_filtered_results \
  --output_results_file output/predicted_results/desk_chair_for_study_filtered_collision/desk_chair_for_study_filtered_collision_gt_results.pkl \
  --output_indices_file output/predicted_results/desk_chair_for_study_filtered_collision/desk_chair_for_study_filtered_collision_gt_scene_indices.pkl

python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/desk_chair_for_study_filtered_collision/desk_chair_for_study_filtered_collision_gt_results.pkl


python ../steerable-scene-generation/scripts/physcene_metrics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/desk_chair_for_study_filtered_collision/desk_chair_for_study_filtered_collision_gt_results.pkl --filtered_gt

python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl --output_directory ./classifier_tmps_3 --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/desk_chair_for_study_filtered_collision/desk_chair_for_study_filtered_collision_gt_results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --no_texture

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/desk_chair_for_study_filtered_collision/desk_chair_for_study_filtered_collision_gt_results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/tv_bed_filtered_collision/tv_bed_filtered_collision_gt_results.pkl --output_directory ./classifier_tmps --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

<!-- Robot -->

<!-- collision filtered -->

python scripts/evaluate_reward_satisfaction.py \
  --source gt \
  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom \
  --annotation_file dataset_files/bedroom_threed_front_splits.csv \
  --reward_file scripts/robot_fetch_from_table_1m_high.py \
  --threshold 2 \
  --comparison ge \
  --save_filtered_results \
  --output_results_file output/predicted_results/robot_fetch_from_table_1m_high_filtered_collision/robot_fetch_from_table_1m_high_filtered_collision_gt_results.pkl \
  --output_indices_file output/predicted_results/robot_fetch_from_table_1m_high_filtered_collision/robot_fetch_from_table_1m_high_filtered_collision_gt_scene_indices.pkl

python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/robot_fetch_from_table_1m_high_filtered_collision/robot_fetch_from_table_1m_high_filtered_collision_gt_results.pkl

python ../steerable-scene-generation/scripts/physcene_metrics.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/robot_fetch_from_table_1m_high_filtered_collision/robot_fetch_from_table_1m_high_filtered_collision_gt_results.pkl --filtered_gt

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/robot_fetch_from_table_1m_high_filtered_collision/robot_fetch_from_table_1m_high_filtered_collision_gt_results.pkl --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl --no_texture

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/robot_fetch_from_table_1m_high_filtered_collision/robot_fetch_from_table_1m_high_filtered_collision_gt_results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/predicted_results/robot_fetch_from_table_1m_high_filtered_collision/robot_fetch_from_table_1m_high_filtered_collision_gt_results.pkl --output_directory ./classifier_tmps --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

=====


FID vs original 3D-FRONT Table

          tv bed | study | robot height
Filtered. 2.9434 | 5.6516 (not thresholded) | 4.36
Ours.     2.90545 | 2.62867 | 3.1438

          tv bed | study | robot height
Filtered. 4.7055 (246) | 14.49 (42) | 4.36 (229)
Ours.     2.90545 | 2.62867 | 3.1438

Filtered Scene count
tv bed: 493 (threshold collision: -1) , 
study: 203 (threshold collision: -0.25), 14.49
robot height: 498 (threshold collision: -0.25) 31.6% , -1 : 2.2