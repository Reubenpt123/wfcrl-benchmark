#!/bin/bash

## This file runs each of the different benchmark algorithms using Floris, and then evaluates their performance on FAST.Farm

seed=1
total_timesteps=150000
track=track
save_model=save_model
env_id=Dec_Ablaincourt_Floris
scenario=constant
wandb_project_name=benchmark-wfcrl-test #wandb_project_name=benchmark-wfcrl-v2
episode_length=10

# Creates a single line of '|' separated messages to filter out from fastfarm output, based on the txt file
messages=$(cat /home/reuben/code/wfcrl-benchmark/scripts/filter_messages.txt | tr -s '\r\n' '|')

set -x

# Train the idqn algorithm on Ablaincourt with Floris
#python /home/reuben/code/wfcrl-benchmark/algos/baseline_idqn.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
# Get the path to the trained model
read -r idqn_model < /home/reuben/code/wfcrl-benchmark/scripts/most_recent_models/idqn_path.txt
# Create the eval FAST.Farm command
idqn_eval="python /home/reuben/code/wfcrl-benchmark/algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idqn_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario"
# Filter fastfarm output
$idqn_eval | grep -Ev "$messages"

#python /home/reuben/code/wfcrl-benchmark/algos/baseline_idrqn.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
#read -r idrqn_model < /home/reuben/code/wfcrl-benchmark/scripts/most_recent_models/idrqn_path.txt
#idrqn_eval="python /home/reuben/code/wfcrl-benchmark/algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idrqn_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario"
#$idrqn_eval | grep -Ev "$messages"

#python /home/reuben/code/wfcrl-benchmark/algos/baseline_ippo.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name --episode_length $episode_length
#read -r ippo_model < /home/reuben/code/wfcrl-benchmark/scripts/most_recent_models/ippo_path.txt
#ippo_eval="python /home/reuben/code/wfcrl-benchmark/algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $ippo_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario"
#$ippo_eval | grep -Ev "$messages"

#python /home/reuben/code/wfcrl-benchmark/algos/baseline_mappo.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
#read -r mappo_model < /home/reuben/code/wfcrl-benchmark/scripts/most_recent_models/mappo_path.txt
#mappo_eval="python /home/reuben/code/wfcrl-benchmark/algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $mappo_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario"
#$mappo_eval | grep -Ev "$messages"

#python /home/reuben/code/wfcrl-benchmark/algos/baseline_qmix.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
#read -r qmix_model < /home/reuben/code/wfcrl-benchmark/scripts/most_recent_models/qmix_path.txt
#qmix_eval="python /home/reuben/code/wfcrl-benchmark/algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $qmix_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario"
#$qmix_eval | grep -Ev "$messages"

#sudo shutdown -h +15
