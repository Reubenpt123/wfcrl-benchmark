#!/bin/bash
seed=1
total_timesteps=3000
track=no-track
save_model=save_model
env_id=Dec_Ablaincourt_Floris
scenario=constant
wandb_project_name=benchmark-wfcrl-test #wandb_project_name=benchmark-wfcrl-v2
episode_length=200
BASE_DIR="/home/reuben/code/wfcrl-benchmark"

set -x
python "$BASE_DIR/algorithms/ifac.py" --seed $seed --env_id $env_id --total_timesteps $total_timesteps \
       --$track --$save_model --wandb_project_name $wandb_project_name
