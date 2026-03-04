#!/bin/bash

# Array of all run names
runs=(
    "ddpo_tv_bed"
    "b2_tv_bed"
    "inc_b2_tv_bed"
    "4_particles_incremental_branch_fk_tv_bed"
    "4_particles_incremental_fk_tv_bed"
)

# Start and end indices
START=0
END=99

# Function to run eval for a single run
run_eval() 
    local run_name=$1
    echo "Starting evaluation for: $run_name"
    ./eval_all_stages.sh "$run_name" $START $END
    echo "Completed evaluation for: $run_name"
}

# Export the function so it can be used by parallel
export -f run_eval
export START END

# Run all evaluations in parallel
# You can adjust the number of parallel jobs with -j flag
# -j 5 means run all 5 at once
# -j 2 means run 2 at a time, etc.
printf '%s\n' "${runs[@]}" | parallel -j 5 run_eval {}

echo "All evaluations completed!"