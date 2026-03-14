#!/bin/bash

# Array of all run names
runs=(
    # "tv_bed_only"
    # "tv_bed_universal"
    # "3d_oob"
    # "3d_collision"
    # "universal_only"
    "universal_only_oob_area"
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