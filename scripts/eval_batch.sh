#!/bin/bash

seed=2
total_timesteps=1000000
track=no-track
env_id=Dec_Ablaincourt_Floris

set -x

# Read the first line of the file into a variable
read -r idqn_model < ~/code/wfcrl-benchmark/scripts/ippo_path.txt
env_id=Dec_Ablaincourt_FastFarm
python algos/eval.py --seed 0 --env_id Dec_Ablaincourt_Fastfarm --pretrained_models $idqn_model --$track

#python algos/eval.py --seed $seed --env_id Dec_Ablaincourt_Fastfarm --$track --pretrained_models runs/Dec_Ablaincourt_Floris__baseline_ippo__constant_2__1740754177
