cd /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion
PYTHONPATH=. python scripts/generate_results.py model.pt --result_tag test --n_syn_scenes 5

PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test/results.pkl  --no_texture --without_floor --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/predicted_results/test/results.pkl  --no_texture --path_to_pickled_3d_future_model /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_future_model_bedroom.pkl

PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag pretrained_3d_layout --with_wandb_logger
PYTHONPATH=. python scripts/ashok_train.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/config.yaml --experiment_tag test --with_wandb_logger --overfit_test


PYTHONPATH=. python scripts/ashok_generate_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/log/pretrained_3d_layout/best_model.pt \
    --result_tag test --n_syn_scenes 4 --batch_size 64 --overfit_test
