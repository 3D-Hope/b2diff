#!/bin/bash

# run_name="5_only_lambda_2_fk_4particles"
# run_name="10_only_lambda_2_fk_4particles"

# run_name="best_worst_lambda_10_fk_4particles"
# run_name="10_only_lambda_2_fk_4particles"
# run_name="lambda_10_fk_4particles"

# run_names=("15_only_lambda_2_fk_4particles" "20_only_lambda_2_fk_4particles" "best_worst_lambda_2_fk_4particles" "incremental_5_10_15_only_lambda_2_fk_4particles")
# run_names=("incremental_5_10_15_only_lambda_2_fk_4particles")
# run_names=("incremental_4_8_12_16_only_lambda_2_fk_4particles" "incremental_5_10_15_only_lambda_2_fk_4particles")
# run_names=("branch_lambda_2_fk_4particles" "bevist_worst_branch_lambda_2_fk_4particles")
# run_names=("custom_bpt_incremental_4_8_12_16_fk" "fk_then_branch_run" "new_incremental_4_8_12_16_only_lambda_2_fk_4particles")
# run_names=("incremental_5_10_15_only_lambda_2_fk_4particles" "new_incremental_4_8_12_16_only_lambda_2_fk_4particles")
# run_names=("5_only_lambda_2_fk_4particle_seed_69" "always_split_15_b2_only_5" "always_split_15_b2_inc" "all_norm_inc" "only_10_branch_lambda_2_fk_4particles")
# # run_names=("only_5_branch_lambda_2_fk_4particles")
# run_names=("only_10_branch_lambda_2_fk_4particles")
# run_names=("new_100inc_b2diffu")
# run_names=("5_only_lambda_2_fk_4particles")
# run_names=("5_only_lambda_2_fk_4particle_seed_69")
# run_names=("another_only_5_steps")
# run_names=("last_only_10_all_norm")
# run_names=("new_fk_only")

# run_names=("template2_branch" "template2_fk_only" "template2_b2")
# run_names=("infer_in_cluster_template_1_pretrained")
# run_names=("vanilla_ddpo" "incremental_branch_lambda_2_fk_4particles")
# run_names=("uniform_only_10_all_norm")
# run_names=("template2_branch_fk" "template2_b2" "template2_ddpo" "template3_branch_fk")
# run_names=("template3_branch_fk_v2" "template2_branch_fk_v2" "geometric_branch_fk_v2")
# run_names=("geometric_ddpo_new" "geometric_branch_fk" "geometric_b2_new")
# run_names=("incremental_branch_lambda_2_fk_4particles_new")
# run_names=("template2_pretrained_new" "template3_pretrained_new" "geometric_pretrained_new")
# run_names=("geometric_fk_inference" "template1_fk_inference" "template2_fk_inference" "template3_fk_inference")
# run_names=("geometric_branch_fk_v2" "geometric_b2_new" "geometric_ddpo_new" "geometric_branch_fk")
# run_names=("template1_ddpo_kl" "template1_b2_kl" "template1_branch_fk_kl")
# run_names=("geometric_ddpo_new_test" "geometric_b2_new_test" "geometric_branch_fk_v2_test")
# run_names=("geometric_pretrained_test_set")
run_names=("uniform_10", "last_10")
for run_name in "${run_names[@]}"; do
    mkdir -p ./outputs/${run_name}
    # rsync -avz -e "ssh -J insait" pramish_paudel@msp3-login-0:/home/pramish_paudel/codes/b2diff/outputs/${run_name} ./outputs/
    # rsync -avz -e "ssh -J insait" pramish_paudel@sof1-h200-1:/home/pramish_paudel/codes/b2diff/outputs/${run_name} ./outputs/
    # rsync -avz -e "ssh -J insait" pramish_paudel@sof1-h200-1:/work/pramish/b2diff/outputs/${run_name} ./outputs/

    rsync -avz -e "ssh -J insait" pramish_paudel@msp3-login-0:/home/pramish_paudel/codes/b2diff/model/lora/${run_name} ./outputs/
    # rsync -avz -e "ssh -J insait" pramish_paudel@sof1-h200-1:/home/pramish_paudel/codes/b2diff/model/lora/${run_name} ./outputs/

