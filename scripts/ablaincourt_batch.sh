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
total_timesteps=10000
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

# Train IDQN
log_time "IDQN Training" "START"
python "$BASE_DIR/algorithms/baseline_idqn.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
       --$track --$save_model --wandb_project_name $wandb_project_name --episode_length $episode_length
log_time "IDQN Training" "END"
# Log IDQN model path
read -r idqn_model_path < "$BASE_DIR/scripts/most_recent_models/idqn_path.txt"
echo "IDQN model saved to: $idqn_model_path" | tee -a "$LOG_FILE"


# Train IDRQN
log_time "IDRQN Training" "START"
python "$BASE_DIR/algorithms/baseline_idrqn.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
       --$track --$save_model --wandb_project_name $wandb_project_name --episode_length $episode_length
log_time "IDRQN Training" "END"
# Log IDRQN model path
read -r idrqn_model_path < "$BASE_DIR/scripts/most_recent_models/idrqn_path.txt"
echo "IDRQN model saved to: $idrqn_model_path" | tee -a "$LOG_FILE"


# Train IPPO
log_time "IPPO Training" "START"
python "$BASE_DIR/algorithms/baseline_ippo.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
       --$track --$save_model --wandb_project_name $wandb_project_name --episode_length $episode_length
log_time "IPPO Training" "END"
# Log IPPO model path
read -r ippo_model_path < "$BASE_DIR/scripts/most_recent_models/ippo_path.txt"
echo "IPPO model saved to: $ippo_model_path" | tee -a "$LOG_FILE"


# Train MAPPO
log_time "MAPPO Training" "START"
python "$BASE_DIR/algorithms/baseline_mappo.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
       --$track --$save_model --wandb_project_name $wandb_project_name --episode_length $episode_length
log_time "MAPPO Training" "END"
# Log MAPPO model path
read -r mappo_model_path < "$BASE_DIR/scripts/most_recent_models/mappo_path.txt"
echo "MAPPO model saved to: $mappo_model_path" | tee -a "$LOG_FILE"


# Train QMIX
log_time "QMIX Training" "START"
python "$BASE_DIR/algorithms/baseline_qmix.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
       --$track --$save_model --wandb_project_name $wandb_project_name --episode_length $episode_length
log_time "QMIX Training" "END"
# Log QMIX model path
read -r qmix_model_path < "$BASE_DIR/scripts/most_recent_models/qmix_path.txt"
echo "QMIX model saved to: $qmix_model_path" | tee -a "$LOG_FILE"


# Train IFAC
log_time "IFAC Training" "START"
python "$BASE_DIR/algorithms/ifac.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
       --$track --$save_model --wandb_project_name $wandb_project_name
log_time "IFAC Training" "END"
# Log IFAC model path
read -r ifac_model_path < "$BASE_DIR/scripts/most_recent_models/ifac_path.txt"
echo "IFAC model saved to: $ifac_model_path" | tee -a "$LOG_FILE"


# Train IFPPO
log_time "IFPPO Training" "START"
python "$BASE_DIR/algorithms/ifppo.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
       --$track --$save_model --wandb_project_name $wandb_project_name
log_time "IFPPO Training" "END"
# Log IFPPO model path
read -r ifppo_model_path < "$BASE_DIR/scripts/most_recent_models/ifppo_path.txt"
echo "IFPPO model saved to: $ifppo_model_path" | tee -a "$LOG_FILE"


# Training run summary
log_summary
echo "===========================================" | tee -a "$LOG_FILE"
echo "Batch script completed at $(date '+%d/%m/%y %H:%M')" | tee -a "$LOG_FILE"
echo "Log file saved: $LOG_FILE" | tee -a "$LOG_FILE"

#sudo shutdown -h +15
