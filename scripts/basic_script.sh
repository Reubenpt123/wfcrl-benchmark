#!/bin/bash

BASE_DIR="/home/reuben/code/wfcrl-benchmark"
seed=1
total_timesteps=150
track=no-track
save_model=save_model
env_id=Dec_Ablaincourt_Floris
scenario=constant
wandb_project_name=benchmark-wfcrl-test #wandb_project_name=benchmark-wfcrl-v2
episode_length=50
plot_power_ylim="7 9.5"
plot_load_ylim="5 8"
set -x

##############################################################################
# IDQN
##############################################################################
# python "$BASE_DIR/algorithms/baseline_idqn.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
#       --$track --$save_model --wandb_project_name $wandb_project_name --hidden_layer_nn 64 \
#       --episode_length $episode_length --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

# read -r idqn_model_path < "$BASE_DIR/scripts/most_recent_models/idqn_path.txt"
# python "$BASE_DIR/algorithms/evaluate.py" --env_id $env_id --pretrained_models $idqn_model_path --hidden_layer_nn 64 \
#         --scenario $scenario --algorithm idqn --training_timesteps $total_timesteps --training_episode_length $episode_length \
#         --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

##############################################################################
# IDRQN
##############################################################################
# python "$BASE_DIR/algorithms/baseline_idrqn.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
#       --$track --$save_model --wandb_project_name $wandb_project_name --hidden_layer_nn 64 64 \
#       --episode_length $episode_length --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

# read -r idrqn_model_path < "$BASE_DIR/scripts/most_recent_models/idrqn_path.txt"
# python "$BASE_DIR/algorithms/evaluate.py" --env_id $env_id --pretrained_models $idrqn_model_path --hidden_layer_nn 64 64 \
#         --scenario $scenario --algorithm idrqn --training_timesteps $total_timesteps --training_episode_length $episode_length \
#         --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

##############################################################################
# IPPO
##############################################################################
# python "$BASE_DIR/algorithms/baseline_ippo.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
#       --$track --$save_model --wandb_project_name $wandb_project_name --hidden_layer_nn 64 64 \
#       --episode_length $episode_length --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

# read -r ippo_model_path < "$BASE_DIR/scripts/most_recent_models/ippo_path.txt"
# python "$BASE_DIR/algorithms/evaluate.py" --env_id $env_id --pretrained_models $ippo_model_path --hidden_layer_nn $ppo_nn \
#         --scenario $scenario --algorithm ippo --training_timesteps $total_timesteps --training_episode_length $episode_length \
#         --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

##############################################################################
# MAPPO
##############################################################################
# python "$BASE_DIR/algorithms/baseline_mappo.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
#       --$track --$save_model --wandb_project_name $wandb_project_name --hidden_layer_nn 64 64 \
#       --episode_length $episode_length --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

# read -r mappo_model_path < "$BASE_DIR/scripts/most_recent_models/mappo_path.txt"
# python "$BASE_DIR/algorithms/evaluate.py" --env_id $env_id --pretrained_models $mappo_model_path --hidden_layer_nn 64 64 \
#         --scenario $scenario --algorithm mappo --training_timesteps $total_timesteps --training_episode_length $episode_length \
#         --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

##############################################################################
# Qmix
##############################################################################
# python "$BASE_DIR/algorithms/baseline_qmix.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
#       --$track --$save_model --wandb_project_name $wandb_project_name --hidden_layer_nn 64 \
#       --episode_length $episode_length --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

# read -r qmix_model_path < "$BASE_DIR/scripts/most_recent_models/qmix_path.txt"
# python "$BASE_DIR/algorithms/evaluate.py" --env_id $env_id --pretrained_models $qmix_model_path --hidden_layer_nn 64 \
#         --scenario $scenario --algorithm qmix --training_timesteps $total_timesteps --training_episode_length $episode_length \
#         --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

##############################################################################
# IFAC
##############################################################################
# python "$BASE_DIR/algorithms/ifac.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
#       --$track --$save_model --wandb_project_name $wandb_project_name --hidden_layer_nn False \

# read -r ifac_model_path < "$BASE_DIR/scripts/most_recent_models/ifac_path.txt"
# python "$BASE_DIR/algorithms/evaluate.py" --env_id $env_id --pretrained_models $ifac_model_path --hidden_layer_nn False \
#         --scenario $scenario --algorithm ifac --training_timesteps $total_timesteps \
#         --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim


##############################################################################
# IFPPO 
##############################################################################
# python "$BASE_DIR/algorithms/ifppo.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
#       --$track --$save_model --wandb_project_name $wandb_project_name --hidden_layer_nn False \
#       --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim

# read -r ifppo_model_path < "$BASE_DIR/scripts/most_recent_models/ifppo_path.txt"
# python "$BASE_DIR/algorithms/evaluate.py" --env_id $env_id --pretrained_models $ifppo_model_path --hidden_layer_nn False \
#         --scenario $scenario --algorithm ifppo --training_timesteps $total_timesteps \
#         --plot_power_ylim $plot_power_ylim --plot_load_ylim $plot_load_ylim
