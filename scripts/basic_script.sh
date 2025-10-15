#!/bin/bash

BASE_DIR="/home/reuben/code/wfcrl-benchmark"
seed=1
total_timesteps=1000
track=no-track
save_model=save_model
env_id=Dec_Ablaincourt_Floris
scenario=constant
wandb_project_name=benchmark-wfcrl-test #wandb_project_name=benchmark-wfcrl-v2
episode_length=500
plot_power_ylim="7 9.5"
plot_load_ylim="5 8"
q_nn=64
ppo_nn="64 64"

set -x

# Train IPPO
#python "$BASE_DIR/algorithms/baseline_ippo.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
#       --$track --$save_model --wandb_project_name $wandb_project_name --episode_length $episode_length
#read -r ippo_model_path < "$BASE_DIR/scripts/most_recent_models/ippo_path.txt"
#
# Evaluate with FLORIS
#python "$BASE_DIR/algorithms/evaluate.py" --env_id $env_id --pretrained_models $ippo_model_path --hidden_layer_nn 64 64\
#        --episode_length $episode_length --scenario $scenario --algorithm ippo

python "$BASE_DIR/algorithms/baseline_ippo.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
      --$track --$save_model --wandb_project_name $wandb_project_name --hidden_layer_nn $ppo_nn \
      --episode_length $episode_length --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

read -r ippo_model_path < "$BASE_DIR/scripts/most_recent_models/ippo_path.txt"
python "$BASE_DIR/algorithms/evaluate.py" --env_id $env_id --pretrained_models $ippo_model_path --hidden_layer_nn $ppo_nn \
        --scenario $scenario --algorithm ippo --training_timesteps $total_timesteps --training_episode_length $episode_length \
        --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

