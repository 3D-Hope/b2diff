#!/bin/bash

# Array of all run names
runs=(
    "b2_3d_tv_bed"
    "ours_3d_tv_bed"
    "ddpo_3d_collision"
    "ddpo_3d_tv_bed"
    "b2_3d_collision"
    "ours_3d_collision"
)

# Start and end indices
START=0
END=99

# Function to run eval for a single run
run_eval() {
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
# -j 6 means run all 6 at once
# -j 2 means run 2 at a time, etc.
printf '%s\n' "${runs[@]}" | parallel -j 6 run_eval {}

echo "All evaluations completed!"