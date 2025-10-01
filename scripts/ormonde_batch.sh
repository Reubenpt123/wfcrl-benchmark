#!/bin/bash

seed=67
total_timesteps=150000
track=track
save_model=save_model
env_id_floris=Dec_Ormonde_Floris
env_id_fastfarm=Dec_Ormonde_Fastfarm
scenario=constant
wandb_project_name=benchmark-wfcrl-ormonde #wandb_project_name=benchmark-wfcrl-v2
episode_length=300

# The newline characters are \r\n or ^M
# tr replaces both \r and \n; the squeeze option removes the extra |'s
messages=$(cat scripts/filter_messages.txt | tr -s '\r\n' '|')

# for debugging
#echo "$messages"

set -x

#fastfarm /home/mep23rt/code/wfcrl-benchmark/__simul__/fastfarm/FastFarm__1102s__30T_1741260950.201606/FarmInputs/Case.fstf

#python ~/code/wfcrl-benchmark/algorithms/baseline_ippo.py --seed $seed --env_id $env_id_floris --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
#read -r ippo_model < ~/code/wfcrl-benchmark/scripts/ippo_path.txt
#python algorithms/eval.py --seed 0 --env_id $env_id_fastfarm --pretrained_models $ippo_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario

#python ~/code/wfcrl-benchmark/algorithms/baseline_mappo.py --seed $seed --env_id $env_id_floris --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
read -r mappo_model < ~/code/wfcrl-benchmark/scripts/mappo_path.txt
python algorithms/eval.py --seed 0 --env_id $env_id_fastfarm --pretrained_models $mappo_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario

#python ~/code/wfcrl-benchmark/algorithms/baseline_qmix.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
#read -r qmix_model < ~/code/wfcrl-benchmark/scripts/qmix_path.txt
#python algorithms/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $qmix_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario

sudo shutdown -h +15
