#!/bin/bash

## This file runs each of the different benchmark algorithms using Floris, and then evaluates their performance on FAST.Farm

# Configuration
BASE_DIR="/home/reuben/code/wfcrl-benchmark"
LOG_FILE="$BASE_DIR/logs/batch_run_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p "$BASE_DIR/logs"

# Logging functions
log_time() {
    local task="$1" event="$2" timestamp=$(date '+%d/%m/%y %H:%M')
    local message
    
    if [ "$event" = "START" ]; then
        export task_start_time=$(date +%s)
        message="========== [$event] $task - $timestamp =========="
    else
        local duration=$(( $(date +%s) - task_start_time ))
        local minutes_decimal=$(echo "scale=1; $duration/60" | bc)
        message="========== [$event] $task - $timestamp duration ${minutes_decimal} min =========="
    fi
    
    echo "$message"
    echo "$message" >> "$LOG_FILE"
}

train_algorithm() {
    local algo="$1" script="$2" extra_args="$3"
    
    log_time "$algo Training" "START"
    python "$BASE_DIR/algorithms/$script" \
        --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
        --$track --$save_model --wandb_project_name $wandb_project_name \
        $extra_args
    log_time "$algo Training" "END"
    
    # Log model path
    local model_path_file="$BASE_DIR/scripts/most_recent_models/${algo,,}_path.txt"
    read -r model_path < "$model_path_file"
    echo "$algo model saved to: $model_path" | tee -a "$LOG_FILE"
}

log_summary() {
    echo -e "\n===========================================" | tee -a "$LOG_FILE"
    echo "TRAINING RUN SUMMARY" | tee -a "$LOG_FILE"
    echo "===========================================" | tee -a "$LOG_FILE"
    
    for algo in idqn idrqn ippo mappo qmix ifac ifppo; do
        read -r model_path < "$BASE_DIR/scripts/most_recent_models/${algo}_path.txt"
        echo "$(echo $algo | tr '[:lower:]' '[:upper:]'): $model_path" | tee -a "$LOG_FILE"
    done
}

seed=1
total_timesteps=3000
track=no-track
save_model=save_model
env_id=Dec_Ablaincourt_Floris
scenario=constant
wandb_project_name=benchmark-wfcrl-test #wandb_project_name=benchmark-wfcrl-v2
episode_length=200

#set -x

# Initialize logging
echo "===========================================" | tee "$LOG_FILE"
echo "WFCRL BENCHMARK BATCH SCRIPT" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"
echo "Start time: $(date '+%d/%m/%y %H:%M')" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Script location: $(pwd)/scripts/ablaincourt_batch.sh" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "EXPERIMENT PARAMETERS:" | tee -a "$LOG_FILE"
echo "--------------------" | tee -a "$LOG_FILE"
echo "Environment ID: $env_id" | tee -a "$LOG_FILE"
echo "Seed: $seed" | tee -a "$LOG_FILE"
echo "Total timesteps: $total_timesteps" | tee -a "$LOG_FILE"
echo "Episode length: $episode_length" | tee -a "$LOG_FILE"
echo "Scenario: $scenario" | tee -a "$LOG_FILE"
echo "Tracking enabled: $track" | tee -a "$LOG_FILE"
echo "Save models: $save_model" | tee -a "$LOG_FILE"
echo "Wandb project: $wandb_project_name" | tee -a "$LOG_FILE"

# Train all algorithms
train_algorithm "IDQN" "baseline_idqn.py" "--episode_length $episode_length"
# Run the eval FAST.Farm command
#log_time "IDQN Evaluation on FAST.Farm" "START"
#python /home/reuben/code/wfcrl-benchmark/algorithms/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idqn_model --episode_length $episode_length --scenario $scenario --algo idqn --hidden_layer_nn 64
#log_time "IDQN Evaluation on FAST.Farm" "END"

train_algorithm "IDRQN" "baseline_idrqn.py" "--episode_length $episode_length"
#log_time "IDRQN Evaluation on FAST.Farm" "START"
#python /home/reuben/code/wfcrl-benchmark/algorithms/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idrqn_model --episode_length $episode_length --scenario $scenario --algo idrqn --hidden_layer_nn 64 64
#log_time "IDRQN Evaluation on FAST.Farm" "END"

train_algorithm "IPPO" "baseline_ippo.py" "--episode_length $episode_length"
#log_time "IPPO Evaluation on FAST.Farm" "START"
#python /home/reuben/code/wfcrl-benchmark/algorithms/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $ippo_model --episode_length $episode_length --scenario $scenario --algo ippo --hidden_layer_nn 64 64
#log_time "IPPO Evaluation on FAST.Farm" "END"

train_algorithm "MAPPO" "baseline_mappo.py" "--episode_length $episode_length"
#log_time "MAPPO Evaluation on FAST.Farm" "START"
#python /home/reuben/code/wfcrl-benchmark/algorithms/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $mappo_model --episode_length $episode_length --scenario $scenario --algo mappo --hidden_layer_nn 64 64
#log_time "MAPPO Evaluation on FAST.Farm" "END"

train_algorithm "QMIX" "baseline_qmix.py" "--episode_length $episode_length"
#log_time "QMIX Evaluation on FAST.Farm" "START"
#python /home/reuben/code/wfcrl-benchmark/algorithms/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $qmix_model --episode_length $episode_length --scenario $scenario --algo qmix --hidden_layer_nn 64
#log_time "QMIX Evaluation on FAST.Farm" "END"

train_algorithm "IFAC" "ifac.py" ""
#log_time "IFAC Evaluation on FAST.Farm" "START"
#python /home/reuben/code/wfcrl-benchmark/algorithms/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $ifac_model --episode_length $episode_length --scenario $scenario --algo ifac --hidden_layer_nn False
#log_time "IFAC Evaluation on FAST.Farm" "END"

train_algorithm "IFPPO" "ifppo.py" ""
#log_time "IFPPO Evaluation on FAST.Farm" "START"
#python /home/reuben/code/wfcrl-benchmark/algorithms/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $ifppo_model --episode_length $episode_length --scenario $scenario --algo ifppo --hidden_layer_nn False
#log_time "IFPPO Evaluation on FAST.Farm" "END"

# Training run summary
log_summary
echo "===========================================" | tee -a "$LOG_FILE"
echo "Batch script completed at $(date '+%d/%m/%y %H:%M')" | tee -a "$LOG_FILE"
echo "Log file saved: $LOG_FILE" | tee -a "$LOG_FILE"

#sudo shutdown -h +15
