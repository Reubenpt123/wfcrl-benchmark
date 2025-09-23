#!/bin/bash

seed=1
total_timesteps=150000
track=no-track
save_model=save_model
env_id=Dec_Ablaincourt_Floris
scenario=constant
wandb_project_name=benchmark-wfcrl-test #wandb_project_name=benchmark-wfcrl-v2
episode_length=10

# The newline characters are \r\n or ^M
# tr replaces both \r and \n; the squeeze option removes the extra |'s
messages=$(cat scripts/filter_messages.txt | tr -s '\r\n' '|')

# for debugging
#echo "$messages"

set -x
#
#python ~/code/wfcrl-benchmark/algos/baseline_idqn.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
#read -r idqn_model < ~/code/wfcrl-benchmark/scripts/idqn_path.txt
#idqn_eval="python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idqn_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario"
# Filter fastfarm output
#$idqn_eval | grep -Ev "$messages"
#python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idqn_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario
#
#python ~/code/wfcrl-benchmark/algos/baseline_idrqn.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
#read -r idrqn_model < ~/code/wfcrl-benchmark/scripts/idrqn_path.txt
#idrqn_eval="python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idrqn_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario"
# Filter fastfarm output
#$idrqn_eval | grep -Ev "$messages"
#python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idrqn_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario
#
python ~/code/wfcrl-benchmark/algos/baseline_ippo.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name --episode_length $episode_length
#read -r ippo_model < ~/code/wfcrl-benchmark/scripts/ippo_path.txt
#ippo_eval="python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $ippo_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario"
# Filter fastfarm output
#$ippo_eval | grep -Ev "$messages"
#python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $ippo_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario
#
#python ~/code/wfcrl-benchmark/algos/baseline_mappo.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
#read -r mappo_model < ~/code/wfcrl-benchmark/scripts/mappo_path.txt
#mappo_eval="python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $mappo_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario"
# Filter fastfarm output
#$mappo_eval | grep -Ev "$messages"
#python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $mappo_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario
#
#python ~/code/wfcrl-benchmark/algos/baseline_qmix.py --seed $seed --env_id $env_id --total_timesteps $total_timesteps --$track --$save_model --wandb_project_name $wandb_project_name
#read -r qmix_model < ~/code/wfcrl-benchmark/scripts/qmix_path.txt
#qmix_eval="python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $qmix_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario"
# Filter fastfarm output
#$qmix_eval | grep -Ev "$messages"
#python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $qmix_model --$track --wandb_project_name $wandb_project_name --episode_length $episode_length --scenario $scenario
#
#sudo shutdown -h +15
