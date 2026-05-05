experiment_names=("template3" "geometric")
config_names=("template3_train.json" "template4_train.json")

for i    in ${!experiment_names[@]}; do
    experiment_name=${experiment_names[i]}
    config_name=${config_names[i]}
    mkdir -p scripts/${experiment_name}

    cp scripts/template2/template2_ddpo.sh scripts/${experiment_name}/${experiment_name}_ddpo.sh
    cp scripts/template2/template2_b2.sh scripts/${experiment_name}/${experiment_name}_b2.sh
    cp scripts/template2/template2_branch.sh scripts/${experiment_name}/${experiment_name}_branch.sh
    cp scripts/template2/template2_branch_fk.sh scripts/${experiment_name}/${experiment_name}_branch_fk.sh
    cp scripts/template2/template2_fk_only.sh scripts/${experiment_name}/${experiment_name}_fk_only.sh

    cp scripts/template2/infer_in_cluster_template2_fk_inference.sh scripts/${experiment_name}/infer_in_cluster_${experiment_name}_fk_inference.sh
    cp scripts/template2/infer_in_cluster_template2_branch.sh scripts/${experiment_name}/infer_in_cluster_${experiment_name}_branch.sh
    cp scripts/template2/infer_in_cluster_template2_branch_fk.sh scripts/${experiment_name}/infer_in_cluster_${experiment_name}_branch_fk.sh
    cp scripts/template2/infer_in_cluster_template2_fk_only.sh scripts/${experiment_name}/infer_in_cluster_${experiment_name}_fk_only.sh
    cp scripts/template2/infer_in_cluster_template2_ddpo.sh scripts/${experiment_name}/infer_in_cluster_${experiment_name}_ddpo.sh
    cp scripts/template2/infer_in_cluster_template2_b2.sh scripts/${experiment_name}/infer_in_cluster_${experiment_name}_b2.sh

    # replace #SBATCH --job-name=infer_in_cluster_template2_branch
    # with #SBATCH --job-name=infer_in_cluster_${experiment_name}_branch
    sed -i "s/#SBATCH --job-name=infer_in_cluster_template2_branch/#SBATCH --job-name=infer_in_cluster_${experiment_name}_branch/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_branch.sh
    sed -i "s/#SBATCH --job-name=infer_in_cluster_template2_branch_fk/#SBATCH --job-name=infer_in_cluster_${experiment_name}_branch_fk/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_branch_fk.sh
    sed -i "s/#SBATCH --job-name=infer_in_cluster_template2_fk_only/#SBATCH --job-name=infer_in_cluster_${experiment_name}_fk_only/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_fk_only.sh
    sed -i "s/#SBATCH --job-name=infer_in_cluster_template2_ddpo/#SBATCH --job-name=infer_in_cluster_${experiment_name}_ddpo/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_ddpo.sh
    sed -i "s/#SBATCH --job-name=infer_in_cluster_template2_b2/#SBATCH --job-name=infer_in_cluster_${experiment_name}_b2/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_b2.sh
    sed -i "s/#SBATCH --job-name=infer_in_cluster_template2_fk_inference/#SBATCH --job-name=infer_in_cluster_${experiment_name}_fk_inference/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_fk_inference.sh

    # replace #SBATCH --job-name=template2_ddpo
    # with #SBATCH --job-name=${experiment_name}_ddpo
    sed -i "s/#SBATCH --job-name=template2_ddpo/#SBATCH --job-name=${experiment_name}_ddpo/g" scripts/${experiment_name}/${experiment_name}_ddpo.sh
    sed -i "s/#SBATCH --job-name=template2_b2/#SBATCH --job-name=${experiment_name}_b2/g" scripts/${experiment_name}/${experiment_name}_b2.sh
    sed -i "s/#SBATCH --job-name=template2_branch/#SBATCH --job-name=${experiment_name}_branch/g" scripts/${experiment_name}/${experiment_name}_branch.sh
    sed -i "s/#SBATCH --job-name=template2_branch_fk/#SBATCH --job-name=${experiment_name}_branch_fk/g" scripts/${experiment_name}/${experiment_name}_branch_fk.sh
    sed -i "s/#SBATCH --job-name=template2_fk_only/#SBATCH --job-name=${experiment_name}_fk_only/g" scripts/${experiment_name}/${experiment_name}_fk_only.sh

    # replace run_name="template2_b2" with run_name="${experiment_name}_b2"
    sed -i "s/run_name=\"template2_b2\"/run_name=\"${experiment_name}_b2\"/g" scripts/${experiment_name}/${experiment_name}_b2.sh
    sed -i "s/run_name=\"template2_ddpo\"/run_name=\"${experiment_name}_ddpo\"/g" scripts/${experiment_name}/${experiment_name}_ddpo.sh
    sed -i "s/run_name=\"template2_branch\"/run_name=\"${experiment_name}_branch\"/g" scripts/${experiment_name}/${experiment_name}_branch.sh
    sed -i "s/run_name=\"template2_branch_fk\"/run_name=\"${experiment_name}_branch_fk\"/g" scripts/${experiment_name}/${experiment_name}_branch_fk.sh
    sed -i "s/run_name=\"template2_fk_only\"/run_name=\"${experiment_name}_fk_only\"/g" scripts/${experiment_name}/${experiment_name}_fk_only.sh

    # replace run_name="infer_in_cluster_template2_fk_inference" with run_name="infer_in_cluster_${experiment_name}_fk_inference"
    sed -i "s/run_name=\"infer_in_cluster_template2_fk_inference\"/run_name=\"infer_in_cluster_${experiment_name}_fk_inference\"/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_fk_inference.sh
    sed -i "s/run_name=\"infer_in_cluster_template2_branch\"/run_name=\"infer_in_cluster_${experiment_name}_branch\"/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_branch.sh
    sed -i "s/run_name=\"infer_in_cluster_template2_branch_fk\"/run_name=\"infer_in_cluster_${experiment_name}_branch_fk\"/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_branch_fk.sh
    sed -i "s/run_name=\"infer_in_cluster_template2_fk_only\"/run_name=\"infer_in_cluster_${experiment_name}_fk_only\"/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_fk_only.sh
    sed -i "s/run_name=\"infer_in_cluster_template2_ddpo\"/run_name=\"infer_in_cluster_${experiment_name}_ddpo\"/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_ddpo.sh
    sed -i "s/run_name=\"infer_in_cluster_template2_b2\"/run_name=\"infer_in_cluster_${experiment_name}_b2\"/g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_b2.sh

    # replace prompt_file=configs/prompt/template2_train.json with prompt_file=configs/prompt/${config_name}
    sed -i "s|prompt_file=configs/prompt/template2_train.json|prompt_file=configs/prompt/${config_name}|g" scripts/${experiment_name}/${experiment_name}_b2.sh
    sed -i "s|prompt_file=configs/prompt/template2_train.json|prompt_file=configs/prompt/${config_name}|g" scripts/${experiment_name}/${experiment_name}_ddpo.sh
    sed -i "s|prompt_file=configs/prompt/template2_train.json|prompt_file=configs/prompt/${config_name}|g" scripts/${experiment_name}/${experiment_name}_branch.sh
    sed -i "s|prompt_file=configs/prompt/template2_train.json|prompt_file=configs/prompt/${config_name}|g" scripts/${experiment_name}/${experiment_name}_branch_fk.sh
    sed -i "s|prompt_file=configs/prompt/template2_train.json|prompt_file=configs/prompt/${config_name}|g" scripts/${experiment_name}/${experiment_name}_fk_only.sh
    sed -i "s|prompt_file=configs/prompt/template2_train.json|prompt_file=configs/prompt/${config_name}|g" scripts/${experiment_name}/infer_in_cluster_${experiment_name}_fk_inference.sh

done