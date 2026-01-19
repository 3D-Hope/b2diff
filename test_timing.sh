#!/bin/bash
set -e

# Record start time
START_TIME=$(date +%s)
START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
echo "=========================================="
echo "Test started at: ${START_TIME_READABLE}"
echo "=========================================="

# Simulate some work (sleep for 5 seconds)
echo "Simulating work..."
sleep 5

# Record end time and calculate duration
END_TIME=$(date +%s)
END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED_SECONDS=$((END_TIME - START_TIME))

# Convert to hours, minutes, seconds
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
SECONDS=$((ELAPSED_SECONDS % 60))

echo ""
echo "=========================================="
echo "Test completed!"
echo "Start time: ${START_TIME_READABLE}"
echo "End time:   ${END_TIME_READABLE}"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Total seconds: ${ELAPSED_SECONDS}"
echo "=========================================="
