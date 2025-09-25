#!/bin/bash

## This file runs each of the different benchmark algorithms using Floris, and then evaluates their performance on FAST.Farm

# Create log file with timestamp
LOG_FILE="/home/reuben/code/wfcrl-benchmark/logs/batch_run_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p /home/reuben/code/wfcrl-benchmark/logs

# Function to log time with task name
log_time() {
    local task_name="$1"
    local event_type="$2"  # "START" or "END"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_message="==========================================
[$event_type] $task_name
Time: $timestamp"
    
    if [ "$event_type" = "START" ]; then
        export task_start_time=$(date +%s)
    elif [ "$event_type" = "END" ]; then
        local task_end_time=$(date +%s)
        local duration=$((task_end_time - task_start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        log_message="$log_message
Duration: ${hours}h ${minutes}m ${seconds}s"
    fi
    
    log_message="$log_message
=========================================="
    
    # Display on console
    echo "$log_message"
    # Save to log file
    echo "$log_message" >> "$LOG_FILE"
}

seed=1
total_timesteps=2000
track=track
save_model=save_model
env_id=Dec_Ablaincourt_Floris
scenario=constant
wandb_project_name=benchmark-wfcrl-test #wandb_project_name=benchmark-wfcrl-v2
episode_length=100

#set -x

# Initialize logging
echo "Batch script started at $(date '+%Y-%m-%d %H:%M:%S')" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Script parameters: seed=$seed, total_timesteps=$total_timesteps, episode_length=$episode_length" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Train the idqn algorithm on Ablaincourt with Floris
log_time "IDQN Training" "START"
python /home/reuben/code/wfcrl-benchmark/algos/baseline_idqn.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
log_time "IDQN Training" "END"

# Get the path to the trained model
read -r idqn_model < /home/reuben/code/wfcrl-benchmark/scripts/most_recent_models/idqn_path.txt
# Run the eval FAST.Farm command
log_time "IDQN Evaluation on FAST.Farm" "START"
python /home/reuben/code/wfcrl-benchmark/algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idqn_model --episode_length $episode_length --scenario $scenario --algo idqn --hidden_layer_nn 64
log_time "IDQN Evaluation on FAST.Farm" "END"

log_time "IDRQN Training" "START"
python /home/reuben/code/wfcrl-benchmark/algos/baseline_idrqn.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
log_time "IDRQN Training" "END"
read -r idrqn_model < /home/reuben/code/wfcrl-benchmark/scripts/most_recent_models/idrqn_path.txt
log_time "IDRQN Evaluation on FAST.Farm" "START"
python /home/reuben/code/wfcrl-benchmark/algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idrqn_model --episode_length $episode_length --scenario $scenario --algo idrqn --hidden_layer_nn 64 64
log_time "IDRQN Evaluation on FAST.Farm" "END"

log_time "IPPO Training" "START"
python /home/reuben/code/wfcrl-benchmark/algos/baseline_ippo.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name --episode_length $episode_length
log_time "IPPO Training" "END"
read -r ippo_model < /home/reuben/code/wfcrl-benchmark/scripts/most_recent_models/ippo_path.txt
log_time "IPPO Evaluation on FAST.Farm" "START"
python /home/reuben/code/wfcrl-benchmark/algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $ippo_model --episode_length $episode_length --scenario $scenario --algo ippo --hidden_layer_nn 64 64
log_time "IPPO Evaluation on FAST.Farm" "END"

log_time "MAPPO Training" "START"
python /home/reuben/code/wfcrl-benchmark/algos/baseline_mappo.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
log_time "MAPPO Training" "END"
read -r mappo_model < /home/reuben/code/wfcrl-benchmark/scripts/most_recent_models/mappo_path.txt
log_time "MAPPO Evaluation on FAST.Farm" "START"
python /home/reuben/code/wfcrl-benchmark/algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $mappo_model --episode_length $episode_length --scenario $scenario --algo mappo --hidden_layer_nn 64 64
log_time "MAPPO Evaluation on FAST.Farm" "END"

log_time "QMIX Training" "START"
python /home/reuben/code/wfcrl-benchmark/algos/baseline_qmix.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
log_time "QMIX Training" "END"
read -r qmix_model < /home/reuben/code/wfcrl-benchmark/scripts/most_recent_models/qmix_path.txt
log_time "QMIX Evaluation on FAST.Farm" "START"
python /home/reuben/code/wfcrl-benchmark/algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $qmix_model --episode_length $episode_length --scenario $scenario --algo qmix --hidden_layer_nn 64
log_time "QMIX Evaluation on FAST.Farm" "END"

# Final log entry
echo "" | tee -a "$LOG_FILE"
echo "Batch script completed at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "Log file saved: $LOG_FILE" | tee -a "$LOG_FILE"

#sudo shutdown -h +15
