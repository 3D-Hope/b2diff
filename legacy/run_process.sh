set -e

# Record start time
START_TIME=$(date +%s)
START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
# Detect number of GPUs
# Check CUDA_VISIBLE_DEVICES first, then fall back to nvidia-smi
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count GPUs from CUDA_VISIBLE_DEVICES (e.g., "0,1,2" or "0")
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    # Use nvidia-smi to count available GPUs
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    else
        # Default to 1 if nvidia-smi not available
        NUM_GPUS=1
    fi
fi

echo "=========================================="
echo "Training started at: ${START_TIME_READABLE}"
echo "Number of GPUs detected: ${NUM_GPUS}"
echo "=========================================="

SaveInterval=2
SavePath="./model/lora"
PromptFile="config/prompt/template1_train.json"
RandomPrompt=1
ExpName="exp_B2DiffuRL" # experiment name
Seed=300
Beta1=1
Beta2=1
BatchCnt=1 # num batch per epoch
StageCnt=100
SplitStepLeft=14
SplitStepRight=20
TrainEpoch=2
AccStep=64
LR=0.0001
ModelVersion="CompVis/stable-diffusion-v1-4"
NumStep=20
History_Cnt=8
PosThreshold=0.5
NegThreshold=-0.5
SplitTime=3
Dev_Id=0

CUDA_FALGS="--config.dev_id ${Dev_Id}"
SAMPLE_FLAGS="--config.sample.num_batches_per_epoch ${BatchCnt} --config.sample.num_steps ${NumStep} --config.prompt_file ${PromptFile} --config.prompt_random_choose ${RandomPrompt} --config.split_time ${SplitTime}" # 
EXP_FLAGS="--config.exp_name ${ExpName} --config.save_path ${SavePath} --config.pretrained.model ${ModelVersion}"


for i in $(seq 0 $((StageCnt-1)))
do
    interval=$((SplitStepRight-SplitStepLeft+1))
    level=$((i*interval/StageCnt))
    cur_split_step=$((level+SplitStepLeft))

    RUN_FLAGS="--config.run_name stage${i} --config.split_step ${cur_split_step} --config.eval.history_cnt ${History_Cnt} --config.eval.pos_threshold ${PosThreshold} --config.eval.neg_threshold ${NegThreshold}"
    temp_seed=$((Seed+i))
    RANDOM_FLAGS="--config.seed ${temp_seed}"
    TRAIN_FLAGS="--config.train.save_interval ${SaveInterval} --config.train.num_epochs ${TrainEpoch} --config.train.beta1 ${Beta1} --config.train.beta2 ${Beta2} --config.train.gradient_accumulation_steps ${AccStep} --config.train.learning_rate ${LR}"
    LORA_FLAGS=""
    if [ $i != 0 ]; then
        minus_i=$((i-1))
        cur_epoch=${TrainEpoch}
        checkpoint=$((cur_epoch/SaveInterval))
        LORA_FLAGS="--config.resume_from ${SavePath}/${ExpName}/stage${minus_i}/checkpoints/checkpoint_${checkpoint}"
    fi

    echo "||=========== round: ${i} ===========||"
    echo $CUDA_FALGS
    echo $TRAIN_FLAGS
    echo $SAMPLE_FLAGS
    echo $RANDOM_FLAGS
    echo $EXP_FLAGS
    echo $RUN_FLAGS
    echo $LORA_FLAGS

    python3 run_sample.py $CUDA_FALGS $TRAIN_FLAGS $SAMPLE_FLAGS $RANDOM_FLAGS $EXP_FLAGS $RUN_FLAGS $LORA_FLAGS $SELECT_FLAGS
    python3 run_select.py $CUDA_FALGS $TRAIN_FLAGS $SAMPLE_FLAGS $RANDOM_FLAGS $EXP_FLAGS $RUN_FLAGS $LORA_FLAGS $SELECT_FLAGS
    python3 run_train.py $CUDA_FALGS $TRAIN_FLAGS $SAMPLE_FLAGS $RANDOM_FLAGS $EXP_FLAGS $RUN_FLAGS $LORA_FLAGS $SELECT_FLAGS

    sleep 2
done

# Record end time and calculate duration
END_TIME=$(date +%s)
END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED_SECONDS=$((END_TIME - START_TIME))

# Convert to hours, minutes, seconds
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
SECONDS=$((ELAPSED_SECONDS % 60))

# Calculate GPU hours (wall clock time in hours × number of GPUs)
# Use bc for floating point calculation, fallback to awk if bc not available
if command -v bc &> /dev/null; then
    ELAPSED_HOURS=$(echo "scale=4; ${ELAPSED_SECONDS} / 3600" | bc)
    GPU_HOURS=$(echo "scale=4; ${ELAPSED_HOURS} * ${NUM_GPUS}" | bc)
else
    # Fallback to awk if bc is not available
    ELAPSED_HOURS=$(awk "BEGIN {printf \"%.4f\", ${ELAPSED_SECONDS} / 3600}")
    GPU_HOURS=$(awk "BEGIN {printf \"%.4f\", ${ELAPSED_HOURS} * ${NUM_GPUS}}")
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "Start time: ${START_TIME_READABLE}"
echo "End time:   ${END_TIME_READABLE}"
echo "Number of GPUs used: ${NUM_GPUS}"
echo "----------------------------------------"
echo "Wall clock time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"
echo "Total GPU hours: ${GPU_HOURS} GPU-hours"
echo "=========================================="

# Save timing info to a file
TIMING_LOG="log/${ExpName}_timing.txt"
mkdir -p log
echo "Training Timing Information" > ${TIMING_LOG}
echo "=============================" >> ${TIMING_LOG}
echo "Experiment: ${ExpName}" >> ${TIMING_LOG}
echo "Start time: ${START_TIME_READABLE}" >> ${TIMING_LOG}
echo "End time:   ${END_TIME_READABLE}" >> ${TIMING_LOG}
echo "Number of GPUs used: ${NUM_GPUS}" >> ${TIMING_LOG}
echo "----------------------------------------" >> ${TIMING_LOG}
echo "Wall clock time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)" >> ${TIMING_LOG}
echo "Total GPU hours: ${GPU_HOURS} GPU-hours" >> ${TIMING_LOG}
echo "----------------------------------------" >> ${TIMING_LOG}
echo "Number of stages: ${StageCnt}" >> ${TIMING_LOG}
echo "Average time per stage: $((ELAPSED_SECONDS / StageCnt)) seconds" >> ${TIMING_LOG}
if command -v bc &> /dev/null; then
    AVG_GPU_HOURS=$(echo "scale=4; ${GPU_HOURS} / ${StageCnt}" | bc)
else
    AVG_GPU_HOURS=$(awk "BEGIN {printf \"%.4f\", ${GPU_HOURS} / ${StageCnt}}")
fi
echo "Average GPU hours per stage: ${AVG_GPU_HOURS} GPU-hours" >> ${TIMING_LOG}
echo "Timing log saved to: ${TIMING_LOG}"