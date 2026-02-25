PYTHONPATH=. python scripts/generate_results.py model.pt --result_tag test --n_syn_scenes 5

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test/results.pkl  --no_texture --without_floor --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl
