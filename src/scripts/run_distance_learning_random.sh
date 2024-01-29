#!/bin/bash

path=''

while getopts 'f:' flag; do
  case "${flag}" in
    f) path="${OPTARG}" ;;
    *) echo "1" ;;
  esac
done
 

env="hopper-medium-expert-v2"

timesteps=(
    "1000"
    "10000"
    "20000"
    "50000"
    "100000"
    "500000"
    "1000000")


for t in ${timesteps[*]}
do
    python3 src/algorithm/learn_distance_random.py \
    --env $env\
    --seed $"0"\
    --checkpoint_path $checkpoint_path\
    --checkpoint_timestep $t --timesteps 10000 --eval_freq 1000
done&

for env in ${timesteps[*]}
do
    python3 src/algorithm/learn_distance_random.py \
    --env $env \
    --seed $"1"\
    --checkpoint_path $checkpoint_path\
    --checkpoint_timestep $t --timesteps 10000 --eval_freq 1000

done&

for env in ${timesteps[*]}
do
    python3 src/algorithm/learn_distance_random.py \
    --env $env \
    --seed $"2"\
    --checkpoint_path $checkpoint_path\
    --checkpoint_timestep $t --timesteps 10000 --eval_freq 1000

done&

for env in ${timesteps[*]}
do
    python3 src/algorithm/learn_distance_random.py \
    --env $env \
    --seed $"3"\
    --checkpoint_path $checkpoint_path\
    --checkpoint_timestep $t --timesteps 10000 --eval_freq 1000

done&

for env in ${timesteps[*]}
do
    python3 src/algorithm/learn_distance_random.py \
    --env $env \
    --seed $"4" \
    --checkpoint_path $checkpoint_path\
    --checkpoint_timestep $t --timesteps 10000 --eval_freq 1000

done&

