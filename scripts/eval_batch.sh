#!/bin/bash

seed=2
total_timesteps=1000000
track=no-track
env_id=Dec_Ablaincourt_Floris

set -x

# Read the first line of the file into a variable
read -r idqn_model < ~/code/wfcrl-benchmark/scripts/most_recent_models/idqn_path.txt
python algorithms/evaluate.py --seed 0 --env_id $env_id --pretrained_models $idqn_model --$track

read -r idrqn_model < ~/code/wfcrl-benchmark/scripts/most_recent_models/idrqn_path.txt
python algorithms/evaluate.py --seed 0 --env_id $env_id --pretrained_models $idrqn_model --$track

read -r ippo_model < ~/code/wfcrl-benchmark/scripts/most_recent_models/ippo_path.txt
python algorithms/evaluate.py --seed 0 --env_id $env_id --pretrained_models $ippo_model --$track

read -r mappo_model < ~/code/wfcrl-benchmark/scripts/most_recent_models/mappo_path.txt
python algorithms/evaluate.py --seed 0 --env_id $env_id --pretrained_models $mappo_model --$track

read -r qmix_model < ~/code/wfcrl-benchmark/scripts/most_recent_models/qmix_path.txt
python algorithms/evaluate.py --seed 0 --env_id $env_id --pretrained_models $qmix_model --$track

read -r ifac_model < ~/code/wfcrl-benchmark/scripts/most_recent_models/ifac_path.txt
python algorithms/evaluate.py --seed 0 --env_id $env_id --pretrained_models $ifac_model --$track

read -r ifppo_model < ~/code/wfcrl-benchmark/scripts/most_recent_models/ifppo_path.txt
python algorithms/evaluate.py --seed 0 --env_id $env_id --pretrained_models $ifppo_model --$track
